//! Lowering strategies from high-level dialects to backend-friendly forms.

use crate::builder::KernelForgeModule;
use crate::dialect::Operation;
use crate::passes::{PassPipeline, PassPlan};
use tracing::warn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoweringTarget {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone)]
pub struct LoweredOp {
    pub name: String,
    pub target: LoweringTarget,
    pub operation: Operation,
    pub schedule: String,
}

#[derive(Debug, Clone)]
pub struct LoweredModule {
    pub target: LoweringTarget,
    pub operations: Vec<LoweredOp>,
}

impl LoweredModule {
    pub fn lower(source: &KernelForgeModule, target: LoweringTarget) -> Self {
        Self::lower_with_plan(source, target, None)
    }

    pub fn lower_with_plan(
        source: &KernelForgeModule,
        target: LoweringTarget,
        plan: Option<&PassPlan>,
    ) -> Self {
        let mut module = source.clone();
        let pipeline = match plan {
            Some(plan) => PassPipeline::from_plan(plan),
            None => PassPipeline::with_default_passes(),
        };
        if let Err(err) = pipeline.run(&mut module) {
            warn!(error = %err, "failed to run lowering pass pipeline");
        }

        let operations = module
            .operations
            .into_iter()
            .map(|operation| LoweredOp {
                name: operation.name().to_string(),
                schedule: default_schedule_for(&operation, target),
                target,
                operation,
            })
            .collect();

        Self { target, operations }
    }
}

fn default_schedule_for(operation: &Operation, target: LoweringTarget) -> String {
    match (target, operation) {
        (LoweringTarget::Cpu, Operation::Matmul(_)) => "cpu-blocked".to_string(),
        (LoweringTarget::Cpu, Operation::Attention(_)) => "cpu-tiled".to_string(),
        (LoweringTarget::Gpu, Operation::Matmul(_)) => "gpu-wmma".to_string(),
        (LoweringTarget::Gpu, Operation::Attention(_)) => "gpu-warp-specialized".to_string(),
        (_, Operation::Mlp(_)) => "fused-mlp".to_string(),
        (_, Operation::LayerNorm(_)) => "vectorized-layernorm".to_string(),
    }
}
