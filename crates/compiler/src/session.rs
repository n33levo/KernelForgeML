//! Compiler session orchestration.

use crate::pipeline::{CompileArtifacts, CompilerConfig, CompilerPipeline};
use anyhow::Result;
use kernelforge_backend_cpu::runtime::{CpuExecutor, CpuExecutorOptions};
use kernelforge_backend_gpu::{GpuExecutor, GpuPlanner};
use kernelforge_ir::builder::KernelForgeModule;
use kernelforge_ir::lowering::LoweringTarget;
use kernelforge_kernels::config::{KernelProfile, MatmulProblem};
use kernelforge_kernels::matmul::MatmulInputs;
use std::path::PathBuf;

pub struct CompilerSession {
    pipeline: CompilerPipeline,
    cpu_executor: CpuExecutor,
    gpu_executor: GpuExecutor,
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
        })
    }

    pub fn compile(&self, module: KernelForgeModule) -> Result<CompileArtifacts> {
        self.pipeline.compile(module)
    }

    pub fn execute_matmul(
        &mut self,
        problem: MatmulProblem,
        inputs: &MatmulInputs<'_>,
    ) -> Result<MatmulResult> {
        match self.pipeline.config().target {
            LoweringTarget::Cpu => {
                let execution = self.cpu_executor.execute_matmul(problem, inputs)?;
                Ok(MatmulResult {
                    output: execution.output,
                    profile: Some(execution.profile),
                })
            }
            LoweringTarget::Gpu => {
                let output = self.gpu_executor.execute_matmul(problem, inputs)?;
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
