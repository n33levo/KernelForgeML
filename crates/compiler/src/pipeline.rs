//! Compiler pass pipeline assembly.

use anyhow::Result;
use kernelforge_ir::builder::KernelForgeModule;
use kernelforge_ir::lowering::{LoweredModule, LoweringTarget};
use kernelforge_ir::passes::PassPlan;
use kernelforge_optimizer::OptimizationPlan;

#[derive(Debug, Clone, Copy)]
pub struct CompilerConfig {
    pub target: LoweringTarget,
    pub validate_mlir: bool,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            target: LoweringTarget::Cpu,
            validate_mlir: true,
        }
    }
}

#[derive(Debug)]
pub struct CompileArtifacts {
    pub module: KernelForgeModule,
    pub mlir: String,
    pub lowered: LoweredModule,
}

pub struct CompilerPipeline {
    config: CompilerConfig,
}

impl CompilerPipeline {
    pub fn new(config: CompilerConfig) -> Self {
        Self { config }
    }

    pub fn compile(
        &self,
        module: KernelForgeModule,
        plan: Option<&OptimizationPlan>,
    ) -> Result<CompileArtifacts> {
        let mlir = module.to_mlir_text();

        if self.config.validate_mlir {
            module.validate_mlir()?;
        }

        let target = plan
            .and_then(|p| target_from_plan(p))
            .unwrap_or(self.config.target);

        let pass_plan = plan.map(to_pass_plan);
        let lowered = LoweredModule::lower_with_plan(&module, target, pass_plan.as_ref());

        Ok(CompileArtifacts {
            module,
            mlir,
            lowered,
        })
    }

    pub fn config(&self) -> &CompilerConfig {
        &self.config
    }
}

fn target_from_plan(plan: &OptimizationPlan) -> Option<LoweringTarget> {
    match plan.target.as_str() {
        "cpu" => Some(LoweringTarget::Cpu),
        "gpu" => Some(LoweringTarget::Gpu),
        _ => None,
    }
}

fn to_pass_plan(plan: &OptimizationPlan) -> PassPlan {
    PassPlan {
        pass_order: plan.pass_order.clone(),
        enable_fuse_matmul_activation: plan.knobs.enable_fuse_matmul_activation,
        enable_fuse_mlp: plan.knobs.enable_fuse_mlp,
        tile_m: plan.knobs.tile_m,
        tile_n: plan.knobs.tile_n,
        vector_width: plan.knobs.vector_width,
    }
}
