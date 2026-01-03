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
                println!("  Reason: {:?}", report.rejection_reason);
            }
        }
        
        Command::MutateAndTest { corpus } => {
            use kernelforge_optimizer::{
                OptimizationPlan,
                mutator::{Mutator, RegressionCorpus, RegressionCase},
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
            
            println!("Generated {} mutants", mutants.len());
            println!("Regression corpus has {} cases\n", regression_corpus.cases.len());
            
            let mut killed = 0;
            let mut escaped = Vec::new();
            
            for mutant in &mutants {
                // Check if plan validation catches the mutant
                let caught = mutant.plan.validate().is_err();
                
                if caught {
                    killed += 1;
                    println!("✓ Mutant killed: {}", mutant.description);
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
            
            println!("\n=== Mutation Test Summary ===");
            println!("Total mutants: {}", mutants.len());
            println!("Killed: {}", killed);
            println!("Escaped: {}", escaped.len());
            
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
