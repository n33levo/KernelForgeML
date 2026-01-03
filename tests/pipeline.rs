use anyhow::Result;
use kernelforge_compiler::pipeline::{CompilerConfig, CompilerPipeline};
use kernelforge_ir::builder::{tensor, ModuleBuilder};
use kernelforge_ir::dialect::{ActivationKind, DataType};

fn ensure_llvm_prefix() {
    // Check for LLVM 18 (current supported version)
    if std::env::var("LLVM_SYS_180_PREFIX").is_err() {
        let candidate = "/opt/homebrew/opt/llvm@18";
        if std::path::Path::new(candidate).exists() {
            std::env::set_var("LLVM_SYS_180_PREFIX", candidate);
            std::env::set_var("MLIR_SYS_180_PREFIX", candidate);
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
            ActivationKind::None,
        )
        .build();

    let config = CompilerConfig::default();
    let pipeline = CompilerPipeline::new(config);
    let artifacts = pipeline.compile(module)?;
    assert!(artifacts.mlir.contains("func.func"));
    assert!(artifacts.mlir.contains("linalg.matmul"));
    Ok(())
}

#[test]
fn pipeline_lowering_assigns_schedules() -> Result<()> {
    ensure_llvm_prefix();

    let module = ModuleBuilder::new()
        .add_matmul(
            "test_matmul",
            tensor("lhs", &[1024, 4096], DataType::F32),
            tensor("rhs", &[4096, 4096], DataType::F32),
            tensor("out", &[1024, 4096], DataType::F32),
            None,
            ActivationKind::None,
        )
        .add_layer_norm(
            "test_ln",
            tensor("in", &[1024, 768], DataType::F32),
            1e-5,
            tensor("out", &[1024, 768], DataType::F32),
        )
        .build();

    let config = CompilerConfig::default();
    let pipeline = CompilerPipeline::new(config);
    let artifacts = pipeline.compile(module)?;
    
    // Check that lowering assigned schedules
    assert!(!artifacts.lowered.operations.is_empty());
    assert!(artifacts.lowered.operations[0].schedule.contains("cpu"));
    Ok(())
}
