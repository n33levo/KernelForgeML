//! Compiler pass pipeline assembly.

use anyhow::Result;
use kernelforge_ir::builder::KernelForgeModule;
use kernelforge_ir::lowering::{LoweredModule, LoweringTarget};

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

    pub fn compile(&self, module: KernelForgeModule) -> Result<CompileArtifacts> {
        let mlir = module.to_mlir_text();

        if self.config.validate_mlir {
            module.validate_mlir()?;
        }

        let lowered = LoweredModule::lower(&module, self.config.target);

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
