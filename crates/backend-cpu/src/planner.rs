//! Planning logic for CPU execution.

use anyhow::Result;
use kernelforge_autotune::cache::AutotuneCache;
use kernelforge_autotune::tuner::Autotuner;
use kernelforge_kernels::config::{KernelProfile, MatmulProblem};
use kernelforge_kernels::matmul::{DynMatmulKernel, MatmulInputs};
use kernelforge_kernels::registry::KernelRegistry;
use std::path::PathBuf;

pub struct CpuMatmulPlan {
    pub problem: MatmulProblem,
    pub profile: KernelProfile,
    pub kernel: DynMatmulKernel,
}

pub struct CpuPlanner {
    registry: KernelRegistry,
    autotuner: Autotuner,
    cache_path: Option<PathBuf>,
}

impl CpuPlanner {
    pub fn new(registry: KernelRegistry, autotuner: Autotuner) -> Self {
        Self {
            registry,
            autotuner,
            cache_path: None,
        }
    }

    pub fn with_cache_path(mut self, cache_path: PathBuf) -> Self {
        self.cache_path = Some(cache_path);
        self
    }

    pub fn from_cache_path(registry: KernelRegistry, cache_path: PathBuf) -> Result<Self> {
        let cache = AutotuneCache::load_from_file(&cache_path)?;
        let autotuner = Autotuner::new(cache);
        Ok(Self::new(registry, autotuner).with_cache_path(cache_path))
    }

    pub fn plan_matmul(
        &mut self,
        problem: MatmulProblem,
        inputs: &MatmulInputs<'_>,
    ) -> Result<CpuMatmulPlan> {
        let (profile, kernel) =
            self.autotuner
                .select_matmul_kernel(&self.registry, problem, inputs)?;
        Ok(CpuMatmulPlan {
            problem,
            profile,
            kernel,
        })
    }

    pub fn persist_cache(&mut self) -> Result<()> {
        if let Some(path) = self.cache_path.clone() {
            self.autotuner.cache().save_to_file(&path)?;
        }
        Ok(())
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut KernelRegistry {
        &mut self.registry
    }

    pub fn autotuner_mut(&mut self) -> &mut Autotuner {
        &mut self.autotuner
    }
}
