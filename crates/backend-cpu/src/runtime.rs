//! CPU runtime entrypoints.

use crate::planner::{CpuMatmulPlan, CpuPlanner};
use anyhow::Result;
use kernelforge_autotune::cache::AutotuneCache;
use kernelforge_autotune::tuner::Autotuner;
use kernelforge_kernels::config::{KernelProfile, MatmulProblem, MatmulTilingConfig};
use kernelforge_kernels::matmul::{MatmulInputs, PlannedMatmul};
use kernelforge_kernels::MatmulKernel;
use kernelforge_kernels::registry::KernelRegistry;
use ndarray::Array2;
use std::fs;
use std::path::PathBuf;
use tracing::info;

#[derive(Debug, Clone)]
pub struct CpuExecutorOptions {
    pub autotune_cache: Option<PathBuf>,
    pub warmup_runs: usize,
    pub runs: usize,
    pub clear_cache: bool,
}

impl Default for CpuExecutorOptions {
    fn default() -> Self {
        Self {
            autotune_cache: None,
            warmup_runs: 1,
            runs: 5,
            clear_cache: false,
        }
    }
}

pub struct MatmulExecution {
    pub output: Array2<f32>,
    pub profile: KernelProfile,
}

pub struct CpuExecutor {
    planner: CpuPlanner,
}

impl CpuExecutor {
    pub fn new(planner: CpuPlanner) -> Self {
        Self { planner }
    }

    pub fn with_options(mut options: CpuExecutorOptions) -> Result<Self> {
        if let Some(path) = &options.autotune_cache {
            if options.clear_cache && path.exists() {
                fs::remove_file(path)?;
            }
        }

        let registry = KernelRegistry::with_default_matmul_kernels();
        let cache = if let Some(path) = &options.autotune_cache {
            AutotuneCache::load_from_file(path)?
        } else {
            AutotuneCache::new()
        };

        let autotuner = Autotuner::new(cache).with_runs(options.warmup_runs, options.runs);
        let mut planner = CpuPlanner::new(registry, autotuner);

        if let Some(path) = options.autotune_cache.take() {
            planner = planner.with_cache_path(path);
        }

        Ok(Self::new(planner))
    }

    pub fn execute_matmul(
        &mut self,
        problem: MatmulProblem,
        inputs: &MatmulInputs<'_>,
    ) -> Result<MatmulExecution> {
        self.execute_matmul_with_overrides(problem, inputs, MatmulTilingConfig::default(), false)
    }

    /// Execute matmul while honoring explicit tiling/vectorization knobs.
    /// When `use_plan_kernel` is true, bypass autotuning and run the plan-driven kernel.
    pub fn execute_matmul_with_overrides(
        &mut self,
        problem: MatmulProblem,
        inputs: &MatmulInputs<'_>,
        tiling: MatmulTilingConfig,
        use_plan_kernel: bool,
    ) -> Result<MatmulExecution> {
        if use_plan_kernel {
            let kernel = PlannedMatmul::new(tiling);
            let output = kernel.run(&problem, inputs)?;
            let profile = KernelProfile::new(kernel.name(), problem, 0.0);
            return Ok(MatmulExecution { output, profile });
        }

        let plan = self.planner.plan_matmul(problem, inputs)?;
        self.dispatch_matmul(plan, problem, inputs)
    }

    pub fn planner_mut(&mut self) -> &mut CpuPlanner {
        &mut self.planner
    }

    pub fn shutdown(&mut self) -> Result<()> {
        self.planner.persist_cache()
    }

    fn dispatch_matmul(
        &self,
        plan: CpuMatmulPlan,
        problem: MatmulProblem,
        inputs: &MatmulInputs<'_>,
    ) -> Result<MatmulExecution> {
        info!(
            kernel = plan.kernel.name(),
            m = problem.m,
            n = problem.n,
            k = problem.k,
            time_ms = plan.profile.average_time_ms,
            "executing matmul plan"
        );
        let output = plan.kernel.run(&problem, inputs)?;
        Ok(MatmulExecution {
            output,
            profile: plan.profile,
        })
    }
}

impl Default for CpuExecutor {
    fn default() -> Self {
        Self::with_options(CpuExecutorOptions::default()).expect("default CPU executor")
    }
}
