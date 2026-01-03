//! Compiler session orchestration.

use crate::pipeline::{CompileArtifacts, CompilerConfig, CompilerPipeline};
use anyhow::Result;
use kernelforge_backend_cpu::runtime::{CpuExecutor, CpuExecutorOptions};
use kernelforge_backend_gpu::{GpuExecutor, GpuPlanner};
use kernelforge_ir::builder::KernelForgeModule;
use kernelforge_ir::lowering::LoweringTarget;
use kernelforge_kernels::config::{KernelProfile, MatmulProblem, MatmulTilingConfig};
use kernelforge_kernels::matmul::MatmulInputs;
use kernelforge_optimizer::OptimizationPlan;
use std::path::PathBuf;
use tracing::info;

pub struct CompilerSession {
    pipeline: CompilerPipeline,
    cpu_executor: CpuExecutor,
    gpu_executor: GpuExecutor,
    plan: Option<OptimizationPlan>,
}

#[derive(Debug, Clone)]
pub struct SessionOptions {
    pub autotune_cache: Option<PathBuf>,
    pub autotune_warmup_runs: usize,
    pub autotune_runs: usize,
    pub clear_autotune_cache: bool,
}

impl SessionOptions {
    pub fn to_cpu_options(&self) -> CpuExecutorOptions {
        CpuExecutorOptions {
            autotune_cache: self.autotune_cache.clone(),
            warmup_runs: self.autotune_warmup_runs,
            runs: self.autotune_runs,
            clear_cache: self.clear_autotune_cache,
        }
    }
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            autotune_cache: None,
            autotune_warmup_runs: 1,
            autotune_runs: 5,
            clear_autotune_cache: false,
        }
    }
}

impl CompilerSession {
    pub fn new(config: CompilerConfig, options: SessionOptions) -> Result<Self> {
        let pipeline = CompilerPipeline::new(config);
        let cpu_executor = CpuExecutor::with_options(options.to_cpu_options())?;
        let gpu_executor = GpuExecutor::new(GpuPlanner::new())?;
        Ok(Self {
            pipeline,
            cpu_executor,
            gpu_executor,
            plan: None,
        })
    }

    pub fn compile(&self, module: KernelForgeModule) -> Result<CompileArtifacts> {
        self.pipeline.compile(module, self.plan.as_ref())
    }

    /// Attach an optimization plan so lowering and runtime honor knobs.
    pub fn set_plan(&mut self, plan: OptimizationPlan) {
        info!(
            target = plan.target,
            tile_m = plan.knobs.tile_m,
            tile_n = plan.knobs.tile_n,
            tile_k = plan.knobs.tile_k,
            vector_width = plan.knobs.vector_width,
            fuse_matmul_activation = plan.knobs.enable_fuse_matmul_activation,
            fuse_mlp = plan.knobs.enable_fuse_mlp,
            "applying optimization plan to session"
        );
        self.plan = Some(plan);
    }

    pub fn execute_matmul(
        &mut self,
        problem: MatmulProblem,
        inputs: &MatmulInputs<'_>,
    ) -> Result<MatmulResult> {
        let target = self
            .plan
            .as_ref()
            .map(|p| match p.target.as_str() {
                "gpu" => LoweringTarget::Gpu,
                "cpu" => LoweringTarget::Cpu,
                _ => self.pipeline.config().target,
            })
            .unwrap_or(self.pipeline.config().target);

        match target {
            LoweringTarget::Cpu => {
                let overrides = self.plan.as_ref().map(|p| MatmulTilingConfig {
                    tile_m: p.knobs.tile_m,
                    tile_n: p.knobs.tile_n,
                    tile_k: p.knobs.tile_k,
                    unroll: p.knobs.vector_width.max(1),
                });

                let execution = if let Some(config) = overrides {
                    info!(
                        "using plan-driven CPU matmul kernel ({}x{}x{}, vw={})",
                        config.tile_m, config.tile_n, config.tile_k, config.unroll
                    );
                    self.cpu_executor
                        .execute_matmul_with_overrides(problem, inputs, config, true)?
                } else {
                    self.cpu_executor.execute_matmul(problem, inputs)?
                };
                Ok(MatmulResult {
                    output: execution.output,
                    profile: Some(execution.profile),
                })
            }
            LoweringTarget::Gpu => {
                let tiling = self.plan.as_ref().map(|p| {
                    (
                        p.knobs.tile_m as u32,
                        p.knobs.tile_n as u32,
                        p.knobs.tile_k as u32,
                    )
                });
                let output = self
                    .gpu_executor
                    .execute_matmul(problem, inputs, tiling)?;
                let profile = KernelProfile::new("gpu-wgpu", problem, 0.0);
                Ok(MatmulResult {
                    output,
                    profile: Some(profile),
                })
            }
        }
    }

    pub fn shutdown(&mut self) -> Result<()> {
        self.cpu_executor.shutdown()?;
        Ok(())
    }

    pub fn config(&self) -> &CompilerConfig {
        self.pipeline.config()
    }
}

pub struct MatmulResult {
    pub output: ndarray::Array2<f32>,
    pub profile: Option<KernelProfile>,
}
