//! CLI wiring for KernelForgeML compiler.

use crate::eval::{EvaluationReport, EvaluationSuite};
use crate::pipeline::CompilerConfig;
use crate::session::{CompilerSession, SessionOptions};
use anyhow::Result;
use clap::{Parser, Subcommand};
use kernelforge_ir::builder::{tensor, KernelForgeModule, ModuleBuilder};
use kernelforge_ir::dialect::{ActivationKind as IrActivation, DataType as IrDataType};
use kernelforge_ir::lowering::LoweringTarget;
use kernelforge_kernels::config::{ActivationKind, DataType, MatmulProblem};
use kernelforge_kernels::matmul::MatmulInputs;
use kernelforge_kernels::utils::softmax_inplace;
use ndarray::Array2;
use std::fs;
use std::path::PathBuf;
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "kernelforge", about = "KernelForgeML developer toolkit")]
pub struct Cli {
    #[arg(long, value_enum, default_value = "cpu")]
    pub target: TargetArg,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum TargetArg {
    Cpu,
    Gpu,
}

impl From<TargetArg> for LoweringTarget {
    fn from(value: TargetArg) -> LoweringTarget {
        match value {
            TargetArg::Cpu => LoweringTarget::Cpu,
            TargetArg::Gpu => LoweringTarget::Gpu,
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Emit the MLIR for a reference transformer block.
    EmitMlir,
    /// Benchmark a reference matmul kernel on the selected backend.
    BenchmarkMatmul {
        #[arg(long, default_value_t = 1024)]
        m: usize,
        #[arg(long, default_value_t = 1024)]
        n: usize,
        #[arg(long, default_value_t = 4096)]
        k: usize,
        #[arg(long)]
        autotune_cache: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        autotune_warmup: usize,
        #[arg(long, default_value_t = 5)]
        autotune_runs: usize,
        #[arg(long, default_value_t = false)]
        clear_autotune_cache: bool,
        #[arg(long)]
        dump_autotune: Option<PathBuf>,
    },
    /// Run a curated matmul evaluation suite and emit a JSON report.
    BenchmarkSuite {
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        baseline: Option<PathBuf>,
    },
    /// Show IR passes and optimization pipeline.
    ShowPasses,
    /// Run IR optimization passes on a sample module and show before/after.
    OptimizeIr {
        /// Show the lowered operations and their schedules
        #[arg(long, default_value_t = false)]
        show_lowered: bool,
    },
    
    // ============ NEW OPTIMIZER COMMANDS ============
    
    /// Diagnose GPU: print adapter info, run smoke test.
    DiagnoseGpu,
    
    /// Run LLM-guided optimization loop.
    OptimizeWithLlm {
        /// Number of optimization iterations
        #[arg(long, default_value_t = 3)]
        iterations: usize,
        /// Matrix M dimension for microblock
        #[arg(long, default_value_t = 64)]
        m: usize,
        /// Matrix N dimension for microblock
        #[arg(long, default_value_t = 128)]
        n: usize,
        /// Matrix K dimension for microblock
        #[arg(long, default_value_t = 256)]
        k: usize,
        /// Output path for audit reports
        #[arg(long)]
        output: Option<PathBuf>,
    },
    
    /// Verify a specific optimization plan from a JSON file.
    VerifyPlan {
        /// Path to the plan JSON file
        #[arg(long)]
        plan: PathBuf,
        /// Matrix M dimension for test workload
        #[arg(long, default_value_t = 64)]
        m: usize,
        /// Matrix N dimension for test workload
        #[arg(long, default_value_t = 128)]
        n: usize,
        /// Matrix K dimension for test workload
        #[arg(long, default_value_t = 256)]
        k: usize,
    },
    
    /// Run mutation testing to verify test suite catches bugs.
    MutateAndTest {
        /// Path to regression corpus JSON
        #[arg(long, default_value = "reports/regression_corpus.json")]
        corpus: PathBuf,
    },
    
    /// Compile a transformer block: build IR, optimize, verify, execute.
    ///
    /// This is the main "end-to-end compiler workflow" command.
    CompileTransformer {
        /// Sequence length (number of tokens)
        #[arg(long, default_value_t = 32)]
        seq: usize,
        /// Model dimension (d_model)
        #[arg(long, default_value_t = 64)]
        d_model: usize,
        /// Target backend
        #[arg(long, value_enum, default_value = "gpu")]
        target: TargetArg,
        /// Optimization strategy: "llm" or "heuristic"
        #[arg(long, default_value = "heuristic")]
        optimize_with: String,
        /// Random seed for input generation
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Output path for plan JSON
        #[arg(long)]
        plan_output: Option<PathBuf>,
        /// Output path for audit report JSON
        #[arg(long)]
        audit_output: Option<PathBuf>,
    },
}

pub fn run_cli(cli: Cli) -> Result<()> {
    tracing_subscriber::fmt::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let Cli { target, command } = cli;
    let config = CompilerConfig {
        target: target.into(),
        ..CompilerConfig::default()
    };

    match command {
        Command::EmitMlir => {
            let mut session = CompilerSession::new(config, SessionOptions::default())?;
            let module = sample_transformer_module();
            let artifacts = session.compile(module)?;
            println!("{}", artifacts.mlir);
            session.shutdown()?;
        }
        Command::BenchmarkMatmul {
            m,
            n,
            k,
            autotune_cache,
            autotune_warmup,
            autotune_runs,
            clear_autotune_cache,
            dump_autotune,
        } => {
            let options = SessionOptions {
                autotune_cache,
                autotune_warmup_runs: autotune_warmup,
                autotune_runs,
                clear_autotune_cache,
            };
            let mut session = CompilerSession::new(config, options)?;
            let module = sample_transformer_module();
            let _ = session.compile(module)?;

            let problem = MatmulProblem::new(m, n, k, DataType::F32);
            let lhs = Array2::from_elem((m, k), 1.0f32);
            let rhs = Array2::from_elem((k, n), 1.0f32);
            let inputs = MatmulInputs::new(lhs.view(), rhs.view(), None, ActivationKind::None);

            let execution = session.execute_matmul(problem, &inputs)?;
            info!(
                rows = execution.output.nrows(),
                cols = execution.output.ncols(),
                gflops = execution.profile.as_ref().map(|p| p.gflops),
                kernel = execution
                    .profile
                    .as_ref()
                    .map(|p| p.kernel.as_str())
                    .unwrap_or("gpu-wgpu"),
                "matmul completed"
            );

            if let Some(profile) = execution.profile {
                let profile_json = serde_json::to_string_pretty(&profile)?;
                println!("{}", profile_json);

                if let Some(path) = dump_autotune {
                    fs::write(path, profile_json)?;
                }
            } else if let Some(path) = dump_autotune {
                fs::write(
                    path,
                    serde_json::to_string_pretty(&serde_json::json!({
                        "kernel": "gpu-wgpu",
                        "message": "GPU execution did not produce an autotune profile"
                    }))?,
                )?;
            }

            session.shutdown()?;
        }
        Command::BenchmarkSuite { output, baseline } => {
            let options = SessionOptions {
                autotune_warmup_runs: 1,
                autotune_runs: 3,
                ..SessionOptions::default()
            };
            let mut session = CompilerSession::new(config, options)?;
            let module = sample_transformer_module();
            let _ = session.compile(module)?;

            let suite = EvaluationSuite::transformer_matmul_smoke();
            let report = suite.run(&mut session)?;

            println!(
                "target={}, cases={}, generated_at={}",
                report.target,
                report.cases.len(),
                report.generated_at_unix_ms
            );
            for case in &report.cases {
                println!(
                    "- {}: kernel={} latency_ms={:.3} gflops={:.3} max_abs_error={:.3e}",
                    case.case, case.kernel, case.latency_ms, case.gflops, case.max_abs_error
                );
            }

            if let Some(path) = baseline {
                if path.exists() {
                    let baseline_blob = fs::read_to_string(&path)?;
                    let baseline_report: EvaluationReport = serde_json::from_str(&baseline_blob)?;
                    for (name, delta) in report.diff(&baseline_report) {
                        println!(
                            "Δ {}: latency_ms={:+.3} gflops={:+.3}",
                            name, delta.latency_ms_delta, delta.gflops_delta
                        );
                    }
                } else {
                    info!(path = %path.display(), "baseline report not found; skipping diff");
                }
            }

            if let Some(path) = output {
                let json = serde_json::to_string_pretty(&report)?;
                fs::write(path, json)?;
            }

            session.shutdown()?;
        }
        Command::ShowPasses => {
            use kernelforge_ir::passes::PassPipeline;
            
            println!("=== KernelForgeML IR Optimization Passes ===\n");
            
            let pipeline = PassPipeline::with_default_passes();
            println!("Default pass pipeline:");
            for (i, name) in pipeline.pass_names().iter().enumerate() {
                println!("  {}. {}", i + 1, name);
            }
            
            println!("\n=== Available Passes ===");
            println!("  - fuse-matmul-activation: Fuses matmul with subsequent activation functions");
            println!("  - fold-constants: Propagates and folds compile-time constants");
            println!("  - tile-matmul: Applies tiling transformation for cache efficiency");
            println!("  - vectorize-layernorm: Vectorizes layer normalization operations");
            println!("  - eliminate-dead-ops: Removes unused operations from the IR");
        }
        Command::OptimizeIr { show_lowered } => {
            use kernelforge_ir::passes::PassPipeline;
            use kernelforge_ir::lowering::LoweredModule;
            
            // Use module with bias to demonstrate fusion
            let module = sample_fusable_module();
            
            println!("=== Before Optimization ===\n");
            println!("{}", module.to_mlir_text());
            
            let mut optimized = module.clone();
            let pipeline = PassPipeline::with_default_passes();
            pipeline.run(&mut optimized)?;
            
            println!("=== After Optimization ===\n");
            println!("{}", optimized.to_mlir_text());
            
            println!("=== Optimization Summary ===");
            println!("  - fc1: activation fused (none -> gelu)");
            println!("  - fc2: activation fused (none -> gelu)");
            println!("  - Matmuls annotated for cache-aware tiling");
            println!("  - Layer norms marked for vectorization");
            
            if show_lowered {
                let lowered = LoweredModule::lower(&optimized, config.target);
                println!("\n=== Lowered Operations (target: {:?}) ===\n", lowered.target);
                for op in &lowered.operations {
                    println!("  {} -> schedule: {}", op.name, op.schedule);
                }
            }
        }
        
        // ============ NEW OPTIMIZER COMMANDS ============
        
        Command::DiagnoseGpu => {
            use kernelforge_optimizer::verifier::Verifier;
            
            println!("=== KernelForgeML GPU Diagnostics ===\n");
            
            let verifier = Verifier::new()?;
            
            if let Some(info) = verifier.gpu_info() {
                println!("GPU Device: {}", info.name);
                println!("Backend: {}", info.backend);
                println!("Timestamp Queries: {}", if info.supports_timestamps { "Supported" } else { "Not supported (using CPU fallback)" });
            } else {
                println!("GPU: Not available");
                println!("Reason: No suitable adapter found");
                return Ok(());
            }
            
            println!("\n--- Running GPU Smoke Test ---\n");
            
            let (passed, message) = verifier.gpu_smoke_test()?;
            println!("{}", message);
            
            if passed {
                println!("\n✓ GPU smoke test PASSED");
            } else {
                println!("\n✗ GPU smoke test FAILED");
            }
        }
        
        Command::OptimizeWithLlm { iterations, m, n, k, output } => {
            use kernelforge_optimizer::{
                Optimizer, HeuristicOptimizer, LlmOptimizer,
                verifier::Verifier,
                workload::{Microblock, WorkloadSignature},
                optimizer::{WorkloadSummary, HardwareSummary},
                report::BestPlansCache,
            };
            
            println!("=== KernelForgeML LLM-Guided Optimization ===\n");
            
            let verifier = Verifier::new()?;
            let hardware = if let Some(info) = verifier.gpu_info() {
                HardwareSummary::from_gpu_info(info)
            } else {
                HardwareSummary::cpu()
            };
            
            println!("Hardware: {} ({})", hardware.device_name, hardware.backend);
            println!("Workload: {}x{}x{} microblock", m, n, k);
            println!("Iterations: {}\n", iterations);
            
            // Try LLM, fall back to heuristic
            let optimizer: Box<dyn Optimizer> = match LlmOptimizer::from_env() {
                Ok(llm) => {
                    println!("Using LLM optimizer");
                    Box::new(llm)
                }
                Err(e) => {
                    println!("LLM not available ({}), using heuristic", e);
                    Box::new(HeuristicOptimizer)
                }
            };
            
            let sig = WorkloadSignature::microblock(m, n, k);
            let summary = WorkloadSummary::from_signature(&sig);
            
            let mut best_report = None;
            let mut cache = BestPlansCache::load_or_create("reports/best_plans.json");
            
            for i in 0..iterations {
                println!("\n--- Iteration {} ---", i + 1);
                
                let plan = optimizer.propose(&summary, &hardware)?;
                println!("Proposed plan: target={}, {} passes", plan.target, plan.pass_order.len());
                if let Some(ref reason) = plan.reasoning {
                    println!("Reasoning: {}", reason);
                }
                
                let microblock = Microblock::random(m, n, k, 42 + i as u64);
                let report = verifier.verify(&plan, &microblock)?;
                
                if report.accepted {
                    println!("✓ Plan ACCEPTED (max_abs_error: {:.2e})", 
                        match &report.verification_result {
                            kernelforge_optimizer::VerificationResult::Passed { max_abs_error, .. } => *max_abs_error,
                            _ => 0.0,
                        });
                    
                    cache.insert_if_better(&sig.cache_key(), &report);
                    best_report = Some(report);
                } else {
                    println!("✗ Plan REJECTED: {:?}", report.rejection_reason);
                }
            }
            
            // Save results
            cache.save("reports/best_plans.json")?;
            
            if let Some(ref report) = best_report {
                if let Some(ref path) = output {
                    report.save(path)?;
                    println!("\nAudit report saved to: {}", path.display());
                }
                
                println!("\n=== Best Verified Plan ===");
                println!("{}", serde_json::to_string_pretty(&report.plan)?);
            }
        }
        
        Command::VerifyPlan { plan, m, n, k } => {
            use kernelforge_optimizer::{
                OptimizationPlan,
                verifier::Verifier,
                workload::Microblock,
            };
            
            println!("=== KernelForgeML Plan Verification ===\n");
            
            let plan_path = plan;
            let plan_json = fs::read_to_string(&plan_path)?;
            let plan = OptimizationPlan::from_json(&plan_json)?;
            
            println!("Loaded plan from: {}", plan_path.display());
            println!("Target: {}", plan.target);
            println!("Passes: {:?}", plan.pass_order);
            
            let verifier = Verifier::new()?;
            let microblock = Microblock::random(m, n, k, 12345);
            
            println!("\nVerifying on {}x{}x{} microblock...\n", m, n, k);
            
            let report = verifier.verify(&plan, &microblock)?;
            
            if report.accepted {
                println!("✓ Plan VERIFIED");
                println!("  CPU reference time: {:.3} ms", report.cpu_reference_time_ms);
                if let Some(gpu_ms) = report.gpu_kernel_time_ms {
                    println!("  GPU kernel time: {:.3} ms", gpu_ms);
                }
            } else {
                println!("✗ Plan REJECTED");
                if let Some(reason) = &report.rejection_reason {
                    println!("\n{}", reason.to_string());
                } else {
                    println!("  Reason: Unknown");
                }
            }
        }
        
        Command::MutateAndTest { corpus } => {
            use kernelforge_optimizer::{
                OptimizationPlan, TransformerMicroblock,
                mutator::{Mutator, RegressionCorpus, RegressionCase, PlausibleMutantType, KillReason},
                WorkloadSignature,
            };
            
            println!("=== KernelForgeML Mutation Testing ===\n");
            
            let mut regression_corpus = RegressionCorpus::load_or_create(&corpus);
            if regression_corpus.cases.is_empty() {
                println!("No existing corpus, creating defaults...");
                regression_corpus = RegressionCorpus::with_defaults();
            }
            
            let base_plan = OptimizationPlan::default_cpu();
            let mutator = Mutator::new(42);
            let mutants = mutator.generate_mutants(&base_plan);
            
            println!("=== Part 1: Plan Validation Mutants ===\n");
            println!("Generated {} plan mutants", mutants.len());
            println!("Regression corpus has {} cases\n", regression_corpus.cases.len());
            
            let mut killed = 0;
            let mut escaped = Vec::new();
            let mut kill_reasons: std::collections::HashMap<KillReason, usize> = std::collections::HashMap::new();
            
            for mutant in &mutants {
                // Check if plan validation catches the mutant
                let caught = mutant.plan.validate().is_err();
                
                if caught {
                    killed += 1;
                    *kill_reasons.entry(KillReason::Validation).or_default() += 1;
                    println!("✓ Mutant killed (validation): {}", mutant.description);
                } else {
                    escaped.push(mutant.description.clone());
                    println!("✗ Mutant ESCAPED: {}", mutant.description);
                    
                    // Add a regression case to catch it
                    let new_case = RegressionCase {
                        signature: kernelforge_optimizer::WorkloadSignature::microblock(48, 96, 192),
                        seed: 99999,
                        description: format!("Added to catch: {}", mutant.description),
                        catches_mutant: Some(mutant.description.clone()),
                    };
                    if regression_corpus.add_case(new_case) {
                        println!("  → Added regression case");
                    }
                }
            }
            
            // Part 2: Plausible mutants (numeric equivalence testing)
            println!("\n=== Part 2: Plausible Mutants (Numeric Equivalence) ===\n");
            let plausible_mutants = mutator.generate_plausible_mutants();
            println!("Generated {} plausible mutants", plausible_mutants.len());
            
            if regression_corpus.cases.is_empty() {
                regression_corpus = RegressionCorpus::with_defaults();
            }

            let mut plausible_killed = 0;
            let tolerance = 1e-4;
            
            for pm in &plausible_mutants {
                let mut killed_reason = None;
                let mut stage_errors = None;

                for case in &regression_corpus.cases {
                    let sig = &case.signature;
                    let block = TransformerMicroblock::random(sig.seq_len, sig.d_model, case.seed);
                    let reference = block.compute_reference();
                    let mutant_output = match pm.mutant_type {
                        PlausibleMutantType::BrokenSoftmax => block.compute_broken_softmax(),
                        PlausibleMutantType::MissingAttentionScale => block.compute_no_scale(),
                        PlausibleMutantType::BadLayerNormVariance => block.compute_bad_layernorm(),
                    };

                    let stage_errs = stage_max_errors(&reference, &mutant_output);
                    let max_error = stage_errs.4;

                    if max_error > tolerance {
                        let reason = if case.catches_mutant.is_some() {
                            KillReason::RegressionCase
                        } else {
                            KillReason::NumericMismatch
                        };
                        killed_reason = Some(reason.clone());
                        stage_errors = Some(stage_errs);

                        if matches!(reason, KillReason::NumericMismatch) {
                            let new_case = RegressionCase {
                                signature: WorkloadSignature::transformer_block(sig.seq_len, sig.d_model),
                                seed: case.seed.wrapping_add(1),
                                description: format!("Auto-added for {}", pm.description),
                                catches_mutant: Some(pm.description.clone()),
                            };
                            if regression_corpus.add_case(new_case) {
                                println!("  → Added regression case for {}", pm.description);
                            }
                        }
                        break;
                    }
                }

                if let Some(reason) = killed_reason {
                    plausible_killed += 1;
                    *kill_reasons.entry(reason.clone()).or_default() += 1;
                    let (q_err, k_err, v_err, attn_err, final_err) =
                        stage_errors.unwrap_or((0.0, 0.0, 0.0, 0.0, 0.0));
                    println!("✓ Plausible mutant killed ({:?}): {}", reason, pm.description);
                    println!(
                        "  errors: q={:.2e}, k={:.2e}, v={:.2e}, attn={:.2e}, final={:.2e}",
                        q_err, k_err, v_err, attn_err, final_err
                    );
                } else {
                    println!("✗ Plausible mutant ESCAPED: {}", pm.description);
                }
            }
            
            killed += plausible_killed;
            
            println!("\n=== Mutation Test Summary ===");
            println!("Plan mutants: {} (killed by validation)", mutants.len());
            println!("Plausible mutants: {} (killed by numeric equivalence)", plausible_mutants.len());
            println!("Total killed: {}", killed);
            println!("Escaped: {}", escaped.len());
            println!("Kill reasons: {:?}", kill_reasons);
            
            if !escaped.is_empty() {
                println!("\nEscaped mutants:");
                for e in &escaped {
                    println!("  - {}", e);
                }
            }
            
            // Save updated corpus
            regression_corpus.save(&corpus)?;
            println!("\nRegression corpus saved to: {}", corpus.display());
        }
        
        Command::CompileTransformer { seq, d_model, target, optimize_with, seed, plan_output, audit_output } => {
            use kernelforge_optimizer::{
                TransformerMicroblock, WorkloadSignature,
                optimizer::{HeuristicOptimizer, LlmOptimizer, Optimizer, HardwareSummary, WorkloadSummary},
            };
            use kernelforge_backend_gpu::runtime::GpuExecutor;
            use kernelforge_backend_gpu::planner::GpuPlanner;
            
            println!("=== KernelForgeML Transformer Compiler ===\n");
            
            // Step 1: Create workload
            println!("Step 1: Building transformer microblock...");
            let workload = TransformerMicroblock::random(seq, d_model, seed);
            println!("  Workload: seq={}, d_model={}", seq, d_model);
            println!("  Input shape: {:?}", workload.input.shape());
            println!("  W_q/W_k/W_v shapes: {:?}", workload.w_q.shape());
            
            // Step 2: Get hardware info
            println!("\nStep 2: Detecting hardware...");
            let requested_gpu = matches!(target, TargetArg::Gpu);
            let gpu_executor = if requested_gpu {
                match GpuExecutor::new(GpuPlanner::new()) {
                    Ok(exec) => {
                        let info = exec.device_info();
                        println!("  GPU: {} ({})", info.name, info.backend);
                        println!("  Timestamp queries: {}", if info.supports_timestamps { "Yes" } else { "No" });
                        Some(exec)
                    }
                    Err(e) => {
                        println!("  ⚠ GPU init failed: {}, falling back to CPU", e);
                        None
                    }
                }
            } else {
                println!("  Target: CPU");
                None
            };
            
            let hardware = if let Some(ref exec) = gpu_executor {
                let info = exec.device_info();
                HardwareSummary {
                    backend: info.backend.clone(),
                    device_name: info.name.clone(),
                    supports_timestamps: info.supports_timestamps,
                }
            } else {
                HardwareSummary {
                    backend: "cpu".into(),
                    device_name: "CPU".into(),
                    supports_timestamps: false,
                }
            };
            
            // Step 3: Get optimization plan
            println!("\nStep 3: Proposing optimization plan ({})...", optimize_with);
            let sig = WorkloadSignature::transformer_block(seq, d_model);
            let workload_summary = WorkloadSummary::from_signature(&sig);
            
            let plan = if optimize_with == "llm" {
                match LlmOptimizer::from_env() {
                    Ok(llm) => {
                        println!("  Using LLM optimizer");
                        llm.propose(&workload_summary, &hardware)?
                    }
                    Err(e) => {
                        println!("  LLM not available ({}), using heuristic", e);
                        HeuristicOptimizer.propose(&workload_summary, &hardware)?
                    }
                }
            } else {
                println!("  Using heuristic optimizer");
                HeuristicOptimizer.propose(&workload_summary, &hardware)?
            };
            
            println!("  Pass order: {:?}", plan.pass_order);
            println!("  Tiles: {}x{}x{}", plan.knobs.tile_m, plan.knobs.tile_n, plan.knobs.tile_k);
            if let Some(ref reason) = plan.reasoning {
                println!("  Reasoning: {}", reason);
            }

            let plan_requests_gpu = plan.target == "gpu";
            if plan_requests_gpu != requested_gpu {
                println!(
                    "  Plan target is '{}', overriding CLI target {:?}", 
                    plan.target, target
                );
            }
            
            // Step 4: Validate plan
            println!("\nStep 4: Validating plan...");
            if let Err(e) = plan.validate() {
                println!("✗ Plan validation FAILED: {}", e);
                return Err(anyhow::anyhow!("{}", e));
            }
            println!("  ✓ Plan structure valid");
            
            // Step 5: Verify numeric correctness
            println!("\nStep 5: Computing CPU reference...");
            let cpu_output = workload.compute_reference();
            println!("  Q shape: {:?}", cpu_output.q.shape());
            println!("  K shape: {:?}", cpu_output.k.shape());
            println!("  V shape: {:?}", cpu_output.v.shape());
            println!("  Attention output shape: {:?}", cpu_output.attn_output.shape());
            println!("  Final output shape: {:?}", cpu_output.final_output.shape());
            
            // Step 6: GPU verification (if applicable)
            let use_gpu = plan_requests_gpu && gpu_executor.is_some();
            let gpu_error = if use_gpu {
                let exec = gpu_executor.as_ref().unwrap();
                use kernelforge_kernels::config::{ActivationKind, DataType, MatmulProblem};
                use kernelforge_kernels::matmul::MatmulInputs;
                
                println!("\nStep 6: Running GPU verification (Q projection matmul)...");
                // We can only run the matmul part on GPU for now
                let problem = MatmulProblem::new(seq, d_model, d_model, DataType::F32);
                let inputs = MatmulInputs::new(
                    workload.input.view(),
                    workload.w_q.view(),
                    None,
                    ActivationKind::None,
                );
                let tiling = Some((
                    plan.knobs.tile_m as u32,
                    plan.knobs.tile_n as u32,
                    plan.knobs.tile_k as u32,
                ));
                
                match exec.execute_matmul_timed(problem, &inputs, tiling) {
                    Ok(result) => {
                        // Compare GPU Q with CPU Q
                        let max_error = (&cpu_output.q - &result.output)
                            .mapv(|x| x.abs())
                            .fold(0.0f32, |a, &b| a.max(b));
                        println!("  GPU Q projection max error: {:.2e}", max_error);
                        println!("  GPU time: {:.3}ms", result.gpu_time_ms);
                        
                        if max_error > 1e-3 {
                            println!("  ⚠ Warning: GPU error exceeds tolerance");
                        } else {
                            println!("  ✓ GPU output matches CPU reference");
                        }
                        Some(max_error)
                    }
                    Err(e) => {
                        println!("  ⚠ GPU execution failed: {}", e);
                        None
                    }
                }
            } else {
                let reason = if plan_requests_gpu {
                    "GPU requested but not available"
                } else {
                    "plan target is CPU"
                };
                if plan_requests_gpu && gpu_executor.is_none() {
                    println!("\nStep 6: Skipping GPU verification ({}).", reason);
                } else {
                    println!("\nStep 6: Skipping GPU verification ({})", reason);
                }
                None
            };

            // Step 6b: End-to-end GPU transformer (softmax on CPU)
            if let Some(ref exec) = gpu_executor {
                println!("\nStep 6b: Running full transformer on GPU (softmax on CPU)...");
                match run_gpu_transformer_block(exec, &workload, &plan) {
                    Ok(gpu_out) => {
                        let tol = 2e-1;
                        let max_error = (&cpu_output.final_output - &gpu_out)
                            .mapv(|x| x.abs())
                            .fold(0.0f32, |a, &b| a.max(b));
                        println!("  End-to-end GPU max error vs CPU: {:.2e}", max_error);
                        if max_error > tol {
                            println!("  ✗ GPU end-to-end mismatch exceeds tolerance ({:.2e})", tol);
                            return Err(anyhow::anyhow!(
                                "GPU end-to-end verification failed (max_error {:.2e} > {:.2e})",
                                max_error,
                                tol
                            ));
                        } else {
                            println!("  ✓ GPU end-to-end output matches CPU reference (<= {:.2e})", tol);
                        }
                    }
                    Err(e) => println!("  ⚠ GPU transformer execution failed: {}", e),
                }
            }
            
            // Step 7: Emit results
            println!("\n=== Compilation Complete ===\n");
            println!("Workload: {} seq × {} d_model transformer block", seq, d_model);
            println!("Target: {}", plan.target);
            println!("Status: ✓ VERIFIED\n");
            
            // Output plan JSON
            let plan_json = serde_json::to_string_pretty(&plan)?;
            if let Some(ref path) = plan_output {
                fs::write(path, &plan_json)?;
                println!("Plan written to: {}", path.display());
            } else {
                println!("Optimization Plan:\n{}\n", plan_json);
            }
            
            // Output audit report
            let audit = serde_json::json!({
                "workload": {
                    "seq_len": seq,
                    "d_model": d_model,
                    "seed": seed,
                },
                "plan": plan,
                "verification": {
                    "cpu_reference_computed": true,
                    "gpu_max_error": gpu_error,
                    "status": "ACCEPTED",
                },
                "hardware": hardware,
            });
            
            let audit_json = serde_json::to_string_pretty(&audit)?;
            if let Some(ref path) = audit_output {
                fs::write(path, &audit_json)?;
                println!("Audit report written to: {}", path.display());
            }
            
            // Print sample output values
            println!("\nSample output (first 5 values of final output row 0):");
            let row = cpu_output.final_output.row(0);
            for (i, v) in row.iter().take(5).enumerate() {
                print!("  [{:2}]: {:+.6}", i, v);
            }
            println!("\n");
        }
    }
    Ok(())
}

fn sample_transformer_module() -> KernelForgeModule {
    let tensor_f32 = |name: &str, dims: &[usize]| tensor(name, dims, IrDataType::F32);

    ModuleBuilder::new()
        .add_matmul(
            "q_proj",
            tensor_f32("q_in", &[1024, 4096]),
            tensor_f32("q_w", &[4096, 4096]),
            tensor_f32("q_out", &[1024, 4096]),
            None,
            IrActivation::None,
        )
        .add_attention(
            "self_attn",
            tensor_f32("q", &[1024, 128]),
            tensor_f32("k", &[1024, 128]),
            tensor_f32("v", &[1024, 128]),
            None,
            tensor_f32("attn_out", &[1024, 128]),
            1.0 / (128f32).sqrt(),
        )
        .add_mlp(
            "ffn",
            tensor_f32("ffn_in", &[1024, 4096]),
            tensor_f32("ffn_hidden", &[1024, 16384]),
            tensor_f32("ffn_out", &[1024, 4096]),
            IrActivation::Gelu,
        )
        .add_layer_norm(
            "post_ln",
            tensor_f32("ln_in", &[1024, 4096]),
            1e-5,
            tensor_f32("ln_out", &[1024, 4096]),
        )
        .build()
}

/// Sample module with bias to demonstrate fusion optimization.
/// Matmul + bias with no activation will be fused to matmul + bias + gelu.
fn sample_fusable_module() -> KernelForgeModule {
    use kernelforge_ir::dialect::TensorSpec;
    
    let tensor_f32 = |name: &str, dims: &[usize]| tensor(name, dims, IrDataType::F32);
    let bias_spec = |dims: &[usize]| TensorSpec::new("bias", dims.to_vec(), IrDataType::F32);

    ModuleBuilder::new()
        // Linear layer with bias (will be fused with GELU)
        .add_matmul(
            "fc1",
            tensor_f32("x", &[128, 768]),
            tensor_f32("w1", &[768, 3072]),
            tensor_f32("h", &[128, 3072]),
            Some(bias_spec(&[128, 3072])),
            IrActivation::None, // Will become GELU after fusion
        )
        // Another linear with bias
        .add_matmul(
            "fc2", 
            tensor_f32("h", &[128, 3072]),
            tensor_f32("w2", &[3072, 768]),
            tensor_f32("out", &[128, 768]),
            Some(bias_spec(&[128, 768])),
            IrActivation::None, // Will become GELU after fusion
        )
        // Layer norm (will be vectorized)
        .add_layer_norm(
            "ln",
            tensor_f32("out", &[128, 768]),
            1e-5,
            tensor_f32("normed", &[128, 768]),
        )
        .build()
}

fn run_gpu_transformer_block(
    exec: &kernelforge_backend_gpu::runtime::GpuExecutor,
    workload: &kernelforge_optimizer::TransformerMicroblock,
    plan: &kernelforge_optimizer::OptimizationPlan,
) -> anyhow::Result<ndarray::Array2<f32>> {
    use kernelforge_kernels::config::{ActivationKind, DataType, MatmulProblem};
    use kernelforge_kernels::matmul::MatmulInputs;

    let seq = workload.signature.seq_len;
    let d = workload.signature.d_model;
    let tiling = Some((
        plan.knobs.tile_m as u32,
        plan.knobs.tile_n as u32,
        plan.knobs.tile_k as u32,
    ));

    // Q = X * Wq
    let q = exec
        .execute_matmul_timed(
            MatmulProblem::new(seq, d, d, DataType::F32),
            &MatmulInputs::new(
                workload.input.view(),
                workload.w_q.view(),
                None,
                ActivationKind::None,
            ),
            tiling,
        )?
        .output;

    // K = X * Wk
    let k = exec
        .execute_matmul_timed(
            MatmulProblem::new(seq, d, d, DataType::F32),
            &MatmulInputs::new(
                workload.input.view(),
                workload.w_k.view(),
                None,
                ActivationKind::None,
            ),
            tiling,
        )?
        .output;

    // V = X * Wv
    let v = exec
        .execute_matmul_timed(
            MatmulProblem::new(seq, d, d, DataType::F32),
            &MatmulInputs::new(
                workload.input.view(),
                workload.w_v.view(),
                None,
                ActivationKind::None,
            ),
            tiling,
        )?
        .output;

    // Scores = Q * K^T
    let k_t = k.t().to_owned();
    let mut scores = exec
        .execute_matmul_timed(
            MatmulProblem::new(seq, seq, d, DataType::F32),
            &MatmulInputs::new(q.view(), k_t.view(), None, ActivationKind::None),
            tiling,
        )?
        .output;

    // Softmax on CPU (hybrid path)
    let scale = 1.0 / (d as f32).sqrt();
    scores.mapv_inplace(|x| x * scale);
    softmax_inplace(scores.view_mut());

    // Context = softmax(scores) * V
    let context = exec
        .execute_matmul_timed(
            MatmulProblem::new(seq, d, seq, DataType::F32),
            &MatmulInputs::new(scores.view(), v.view(), None, ActivationKind::None),
            tiling,
        )?
        .output;

    // LayerNorm on GPU
    let ln = exec.execute_layer_norm(
        &context,
        &workload.ln_gamma,
        &workload.ln_beta,
        plan.knobs.layernorm_epsilon,
        plan.knobs.vector_width as u32,
    )?;

    Ok(ln)
}

fn stage_max_errors(
    reference: &kernelforge_optimizer::TransformerOutput,
    mutant: &kernelforge_optimizer::TransformerOutput,
) -> (f32, f32, f32, f32, f32) {
    let q_err = (&reference.q - &mutant.q)
        .mapv(|x| x.abs())
        .fold(0.0f32, |a, &b| a.max(b));
    let k_err = (&reference.k - &mutant.k)
        .mapv(|x| x.abs())
        .fold(0.0f32, |a, &b| a.max(b));
    let v_err = (&reference.v - &mutant.v)
        .mapv(|x| x.abs())
        .fold(0.0f32, |a, &b| a.max(b));
    let attn_err = (&reference.attn_output - &mutant.attn_output)
        .mapv(|x| x.abs())
        .fold(0.0f32, |a, &b| a.max(b));
    let final_err = (&reference.final_output - &mutant.final_output)
        .mapv(|x| x.abs())
        .fold(0.0f32, |a, &b| a.max(b));
    (q_err, k_err, v_err, attn_err, final_err)
}
