//! GPU planning logic.

use anyhow::{bail, Result};
use kernelforge_kernels::config::MatmulProblem;

#[derive(Debug, Clone, Copy)]
pub struct GpuMatmulPlan {
    pub problem: MatmulProblem,
    pub workgroup_m: u32,
    pub workgroup_n: u32,
    pub tile_k: u32,
}

pub struct GpuPlanner;

impl GpuPlanner {
    pub fn new() -> Self {
        Self
    }

    pub fn plan_matmul(
        &self,
        problem: MatmulProblem,
        tiling: Option<(u32, u32, u32)>,
    ) -> Result<GpuMatmulPlan> {
        if problem.batch > 1 {
            bail!("GPU backend currently supports only single-batch matmul");
        }
        let (wg_m, wg_n, tile_k) = tiling.unwrap_or((16, 16, 16));
        Ok(GpuMatmulPlan {
            problem,
            workgroup_m: wg_m.max(1),
            workgroup_n: wg_n.max(1),
            tile_k: tile_k.max(1),
        })
    }
}

impl Default for GpuPlanner {
    fn default() -> Self {
        Self::new()
    }
}
