//! Core autotuning logic.

use crate::cache::AutotuneCache;
use anyhow::{anyhow, Result};
use kernelforge_kernels::config::{KernelProfile, MatmulProblem};
use kernelforge_kernels::matmul::{DynMatmulKernel, MatmulInputs};
use kernelforge_kernels::registry::KernelRegistry;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct Autotuner {
    cache: AutotuneCache,
    runs: usize,
    warmup_runs: usize,
}

impl Autotuner {
    pub fn new(cache: AutotuneCache) -> Self {
        Self {
            cache,
            runs: 5,
            warmup_runs: 1,
        }
    }

    pub fn with_runs(mut self, warmup_runs: usize, runs: usize) -> Self {
        self.warmup_runs = warmup_runs;
        self.runs = runs.max(1);
        self
    }

    pub fn cache(&self) -> &AutotuneCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut AutotuneCache {
        &mut self.cache
    }

    pub fn select_matmul_kernel(
        &mut self,
        registry: &KernelRegistry,
        problem: MatmulProblem,
        inputs: &MatmulInputs<'_>,
    ) -> Result<(KernelProfile, DynMatmulKernel)> {
        if let Some(profile) = self.cache.get_matmul(&problem) {
            if let Some(kernel) = registry
                .matmul_kernels()
                .iter()
                .find(|kernel| kernel.name() == profile.kernel)
            {
                return Ok((profile.clone(), Arc::clone(kernel)));
            }
        }

        let mut best: Option<(KernelProfile, DynMatmulKernel)> = None;

        for kernel in registry.matmul_kernels() {
            if !kernel.supports(&problem) {
                continue;
            }

            // Warmup runs to avoid cold-start noise.
            for _ in 0..self.warmup_runs {
                let _ = kernel.run(&problem, inputs)?;
            }

            let mut total = Duration::default();
            for _ in 0..self.runs {
                let start = Instant::now();
                let _ = kernel.run(&problem, inputs)?;
                total += start.elapsed();
            }

            let avg_ms = total.as_secs_f64() * 1000.0 / self.runs as f64;
            let profile = KernelProfile::new(kernel.name(), problem, avg_ms);

            match &best {
                Some((best_profile, _))
                    if profile.average_time_ms >= best_profile.average_time_ms =>
                {
                    continue;
                }
                _ => {
                    best = Some((profile, Arc::clone(kernel)));
                }
            }
        }

        let (profile, kernel) = best.ok_or_else(|| {
            anyhow!(
                "no registered matmul kernels support problem m={} n={} k={}",
                problem.m,
                problem.n,
                problem.k
            )
        })?;

        self.cache.insert_matmul(profile.clone());
        Ok((profile, kernel))
    }
}
