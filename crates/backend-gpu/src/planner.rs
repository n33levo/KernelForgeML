//! GPU planning logic.

use anyhow::{bail, Result};
use kernelforge_kernels::config::MatmulProblem;

#[derive(Debug, Clone, Copy)]
pub struct GpuMatmulPlan {
    pub problem: MatmulProblem,
}

pub struct GpuPlanner;

impl GpuPlanner {
    pub fn new() -> Self {
        Self
    }

    pub fn plan_matmul(&self, problem: MatmulProblem) -> Result<GpuMatmulPlan> {
        if problem.batch > 1 {
            bail!("GPU backend currently supports only single-batch matmul");
        }
        Ok(GpuMatmulPlan { problem })
    }
}

impl Default for GpuPlanner {
    fn default() -> Self {
        Self::new()
    }
}
