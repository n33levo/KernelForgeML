//! Kernel configuration structures.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    F32,
    F16,
    BF16,
}

impl DataType {
    pub fn element_size_bytes(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatmulProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    pub dtype: DataType,
}

impl MatmulProblem {
    pub fn new(m: usize, n: usize, k: usize, dtype: DataType) -> Self {
        Self {
            m,
            n,
            k,
            batch: 1,
            dtype,
        }
    }

    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = batch;
        self
    }

    pub fn flops(&self) -> f64 {
        2.0 * self.m as f64 * self.n as f64 * self.k as f64 * self.batch as f64
    }

    pub fn data_footprint_bytes(&self) -> usize {
        let elem = self.dtype.element_size_bytes();
        self.batch * (self.m * self.k + self.k * self.n + self.m * self.n) * elem
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MatmulTilingConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub unroll: usize,
}

impl Default for MatmulTilingConfig {
    fn default() -> Self {
        Self {
            tile_m: 64,
            tile_n: 64,
            tile_k: 32,
            unroll: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum ActivationKind {
    #[default]
    None,
    Relu,
    Gelu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelProfile {
    pub kernel: String,
    pub problem: MatmulProblem,
    pub average_time_ms: f64,
    pub gflops: f64,
}

impl KernelProfile {
    pub fn new(kernel: &str, problem: MatmulProblem, average_time_ms: f64) -> Self {
        let gflops = if average_time_ms > 0.0 {
            problem.flops() / (average_time_ms * 1.0e6)
        } else {
            0.0
        };

        Self {
            kernel: kernel.to_string(),
            problem,
            average_time_ms,
            gflops,
        }
    }
}
