//! Transformation passes for KernelForgeML IR.

use crate::builder::KernelForgeModule;
use crate::dialect::{ActivationKind, Operation};
use anyhow::Result;
use tracing::debug;

pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, module: &mut KernelForgeModule) -> Result<()>;
}

pub struct FuseMatmulBias;

impl Pass for FuseMatmulBias {
    fn name(&self) -> &str {
        "fuse-matmul-bias"
    }

    fn run(&self, module: &mut KernelForgeModule) -> Result<()> {
        let mut fused = 0;
        for op in module.operations.iter_mut() {
            if let Operation::Matmul(matmul) = op {
                if matmul.bias.is_some() && matches!(matmul.activation, ActivationKind::None) {
                    matmul.activation = ActivationKind::Gelu;
                    fused += 1;
                }
            }
        }
        debug!(pass = self.name(), fused, "applied matmul bias fusion");
        Ok(())
    }
}

pub struct PromoteLayerNorm;

impl Pass for PromoteLayerNorm {
    fn name(&self) -> &str {
        "promote-layer-norm"
    }

    fn run(&self, module: &mut KernelForgeModule) -> Result<()> {
        let mut promoted = 0;
        for op in module.operations.iter_mut() {
            if let Operation::LayerNorm(_) = op {
                promoted += 1;
            }
        }
        debug!(pass = self.name(), promoted, "promoted layer norms");
        Ok(())
    }
}

pub struct PassPipeline {
    passes: Vec<Box<dyn Pass + Send + Sync>>,
}

impl PassPipeline {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn with_default_passes() -> Self {
        Self {
            passes: vec![Box::new(FuseMatmulBias), Box::new(PromoteLayerNorm)],
        }
    }

    pub fn add_pass<P>(&mut self, pass: P)
    where
        P: Pass + Send + Sync + 'static,
    {
        self.passes.push(Box::new(pass));
    }

    pub fn run(&self, module: &mut KernelForgeModule) -> Result<()> {
        for pass in &self.passes {
            pass.run(module)?;
        }
        Ok(())
    }
}

impl Default for PassPipeline {
    fn default() -> Self {
        Self::new()
    }
}
