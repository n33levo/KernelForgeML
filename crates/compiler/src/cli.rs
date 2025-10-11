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
    /// Run LLM inference (text generation with KV-cache).
    LlmInference {
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 20)]
        max_tokens: usize,
        #[arg(long, default_value_t = 1.0)]
        temperature: f32,
        #[arg(long)]
        weights: Option<PathBuf>,
        #[arg(long)]
        cerebras_api_key: Option<String>,
        #[arg(long, default_value = "llama3.1-8b")]
        cerebras_model: String,
        #[arg(long, default_value_t = false)]
        compare_with_cerebras: bool,
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
        Command::LlmInference {
            prompt,
            max_tokens,
            temperature,
            weights,
            cerebras_api_key,
            cerebras_model,
            compare_with_cerebras,
        } => {
            use kernelforge_llm::tokenizer::SimpleTokenizer;
            use kernelforge_llm::{
                CerebrasClient, InferenceComparison, LLMModel, ModelConfig, ModelWeights,
            };

            info!("loading model (tiny config for demo)");
            let config = ModelConfig::tiny();
            let model_weights = if let Some(path) = weights {
                info!(path = %path.display(), "loading weights from safetensors");
                ModelWeights::load_safetensors(path)?
            } else {
                info!("using random weights (not pretrained)");
                ModelWeights::random(
                    config.vocab_size,
                    config.d_model,
                    config.n_layers,
                    config.d_ff,
                )
            };

            let mut model = LLMModel::new(config.clone(), model_weights);
            let tokenizer = SimpleTokenizer::new(config.vocab_size);

            info!("tokenizing prompt");
            let input_ids = tokenizer.encode(&prompt);
            info!(prompt = %prompt, tokens = input_ids.len(), "prompt encoded");

            info!("running local inference");
            let (generated_ids, metrics) = model.generate(&input_ids, max_tokens, temperature)?;
            let output_text = tokenizer.decode(&generated_ids);
            let local_total_time = metrics.prefill_ms + metrics.decode_ms;

            println!("\n=== Local Generation Results ===");
            println!("Prompt: {}", prompt);
            println!("Output: {}", output_text);
            println!("\n=== Local Metrics ===");
            println!("Prefill time: {:.2} ms", metrics.prefill_ms);
            println!("Decode time: {:.2} ms", metrics.decode_ms);
            println!("Total time: {:.2} ms", local_total_time);
            println!("Tokens generated: {}", metrics.tokens_generated);
            println!("Tokens/sec: {:.2}", metrics.tokens_per_sec);
            println!("KV-cache: {} bytes/token", metrics.kv_cache_bytes_per_token);

            // Cerebras comparison if enabled
            if compare_with_cerebras {
                let api_key = cerebras_api_key.or_else(|| std::env::var("CEREBRAS_API_KEY").ok());
                if let Some(api_key) = api_key {
                    info!("running Cerebras comparison");
                    let cerebras_client = CerebrasClient::new(api_key, Some(cerebras_model));

                    let rt = tokio::runtime::Runtime::new()?;
                    match rt.block_on(cerebras_client.generate(&prompt, max_tokens, temperature)) {
                        Ok((cerebras_output, cerebras_metrics)) => {
                            let comparison = InferenceComparison::new(
                                output_text,
                                metrics.tokens_per_sec,
                                local_total_time,
                                cerebras_output,
                                cerebras_metrics.tokens_per_sec,
                                cerebras_metrics.response_time_ms,
                            );
                            comparison.print_comparison();
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to run Cerebras comparison: {}", e);
                            eprintln!("Local inference completed successfully.");
                        }
                    }
                } else {
                    eprintln!(
                        "Warning: --compare-with-cerebras enabled but no CEREBRAS_API_KEY provided"
                    );
                    eprintln!(
                        "Set CEREBRAS_API_KEY environment variable or use --cerebras-api-key flag"
                    );
                }
            }
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
