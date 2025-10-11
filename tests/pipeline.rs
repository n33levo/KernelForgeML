use anyhow::Result;
use kernelforge_compiler::pipeline::{CompilerConfig, CompilerPipeline};
use kernelforge_ir::builder::{tensor, ModuleBuilder};
use kernelforge_ir::dialect::DataType;

fn ensure_llvm_prefix() {
    if std::env::var("LLVM_SYS_170_PREFIX").is_err() {
        let candidate = "/opt/homebrew/opt/llvm@17";
        if std::path::Path::new(candidate).exists() {
            std::env::set_var("LLVM_SYS_170_PREFIX", candidate);
        }
    }
}

#[test]
fn pipeline_emits_valid_mlir() -> Result<()> {
    ensure_llvm_prefix();

    let module = ModuleBuilder::new()
        .add_matmul(
            "unit_matmul",
            tensor("lhs", &[8, 16], DataType::F32),
            tensor("rhs", &[16, 4], DataType::F32),
            tensor("out", &[8, 4], DataType::F32),
            None,
            kernelforge_ir::dialect::ActivationKind::None,
        )
        .build();

    let config = CompilerConfig::default();
    let pipeline = CompilerPipeline::new(config);
    let artifacts = pipeline.compile(module)?;
    assert!(artifacts.mlir.contains("func.func"));
    Ok(())
}
