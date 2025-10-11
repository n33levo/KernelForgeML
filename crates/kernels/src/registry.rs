//! Kernel registry for lookup and discovery.

use crate::matmul::{DynMatmulKernel, MatmulKernel};
use std::sync::Arc;

#[derive(Default)]
pub struct KernelRegistry {
    matmul_kernels: Vec<DynMatmulKernel>,
}

impl Clone for KernelRegistry {
    fn clone(&self) -> Self {
        Self {
            matmul_kernels: self.matmul_kernels.clone(),
        }
    }
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self {
            matmul_kernels: Vec::new(),
        }
    }

    pub fn with_default_matmul_kernels() -> Self {
        let mut registry = Self::new();
        registry.register_matmul_kernel(crate::matmul::ReferenceMatmul::new());
        registry.register_matmul_kernel(crate::matmul::BlockedMatmul::new());
        registry.register_matmul_kernel(crate::matmul::ParallelMatmul::new());
        registry
    }

    pub fn register_matmul_kernel<K>(&mut self, kernel: K)
    where
        K: MatmulKernel + 'static,
    {
        self.matmul_kernels.push(Arc::new(kernel));
    }

    pub fn matmul_kernels(&self) -> &[DynMatmulKernel] {
        &self.matmul_kernels
    }

    pub fn find_matmul_kernel(&self, name: &str) -> Option<DynMatmulKernel> {
        self.matmul_kernels
            .iter()
            .find(|kernel| kernel.name() == name)
            .map(Arc::clone)
    }
}
