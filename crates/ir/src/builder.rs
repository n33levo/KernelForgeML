//! IR builder entrypoints and helpers.

use crate::dialect::{
    ActivationKind, AttentionOp, DataType, LayerNormOp, MatmulOp, MlpOp, Operation, TensorSpec,
};
use anyhow::{bail, Result};
use melior::dialect::DialectRegistry;
use melior::ir::Module;
use melior::utility::register_all_dialects;
use melior::Context;
use std::fmt::Write;

#[derive(Debug, Default, Clone)]
pub struct ModuleBuilder {
    operations: Vec<Operation>,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    pub fn add_matmul<N: Into<String>>(
        mut self,
        name: N,
        lhs: TensorSpec,
        rhs: TensorSpec,
        result: TensorSpec,
        bias: Option<TensorSpec>,
        activation: ActivationKind,
    ) -> Self {
        let op = MatmulOp {
            name: name.into(),
            lhs,
            rhs,
            result,
            bias,
            activation,
        };
        self.operations.push(Operation::Matmul(op));
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_attention<N: Into<String>>(
        mut self,
        name: N,
        query: TensorSpec,
        key: TensorSpec,
        value: TensorSpec,
        mask: Option<TensorSpec>,
        result: TensorSpec,
        scale: f32,
    ) -> Self {
        let op = AttentionOp {
            name: name.into(),
            query,
            key,
            value,
            mask,
            result,
            scale,
        };
        self.operations.push(Operation::Attention(op));
        self
    }

    pub fn add_mlp<N: Into<String>>(
        mut self,
        name: N,
        input: TensorSpec,
        hidden: TensorSpec,
        output: TensorSpec,
        activation: ActivationKind,
    ) -> Self {
        let op = MlpOp {
            name: name.into(),
            input,
            hidden,
            output,
            activation,
        };
        self.operations.push(Operation::Mlp(op));
        self
    }

    pub fn add_layer_norm<N: Into<String>>(
        mut self,
        name: N,
        input: TensorSpec,
        epsilon: f32,
        result: TensorSpec,
    ) -> Self {
        let op = LayerNormOp {
            name: name.into(),
            input,
            epsilon,
            result,
        };
        self.operations.push(Operation::LayerNorm(op));
        self
    }

    pub fn build(self) -> KernelForgeModule {
        KernelForgeModule {
            operations: self.operations,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KernelForgeModule {
    pub operations: Vec<Operation>,
}

impl KernelForgeModule {
    pub fn builder() -> ModuleBuilder {
        ModuleBuilder::new()
    }

    pub fn to_mlir_text(&self) -> String {
        let mut text = String::from("module {\n");

        for op in &self.operations {
            match op {
                Operation::Matmul(op) => {
                    let _ = writeln!(text, "{}", emit_matmul_function(op));
                }
                Operation::Attention(op) => {
                    let _ = writeln!(text, "{}", emit_attention_function(op));
                }
                Operation::Mlp(op) => {
                    let _ = writeln!(text, "{}", emit_mlp_function(op));
                }
                Operation::LayerNorm(op) => {
                    let _ = writeln!(text, "{}", emit_layer_norm_function(op));
                }
            }
        }

        text.push_str("}\n");
        text
    }

    pub fn validate_mlir(&self) -> Result<()> {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let text = self.to_mlir_text();
        if Module::parse(&context, &text).is_none() {
            bail!("failed to parse MLIR module");
        }

        Ok(())
    }
}

pub fn tensor(name: &str, shape: &[usize], dtype: DataType) -> TensorSpec {
    TensorSpec::new(name, shape.to_vec(), dtype)
}

fn emit_matmul_function(op: &MatmulOp) -> String {
    let lhs_ty = op.lhs.mlir_tensor_type();
    let rhs_ty = op.rhs.mlir_tensor_type();
    let res_ty = op.result.mlir_tensor_type();

    let mut body = String::new();
    let attributes = format!(
        " attributes {{kernelforge.activation = \"{}\", kernelforge.has_bias = {}}}",
        op.activation.as_str(),
        if op.bias.is_some() { "true" } else { "false" }
    );

    let _ = writeln!(
        body,
        "  func.func @{}(%arg0: {}, %arg1: {}) -> {}{} {{",
        op.name, lhs_ty, rhs_ty, res_ty, attributes
    );
    body.push_str("    %c0 = arith.constant 0.0 : f32\n");
    body.push_str(&format!("    %empty = tensor.empty() : {}\n", res_ty));
    body.push_str(&format!(
        "    %init = linalg.fill ins(%c0 : f32) outs(%empty : {}) -> {}\n",
        res_ty, res_ty
    ));
    body.push_str(&format!(
        "    %0 = linalg.matmul ins(%arg0, %arg1 : {}, {}) outs(%init : {}) -> {}\n",
        lhs_ty, rhs_ty, res_ty, res_ty
    ));
    body.push_str(&format!("    return %0 : {}\n", res_ty));
    body.push_str("  }\n");
    body
}

fn emit_attention_function(op: &AttentionOp) -> String {
    let mut arg_types = vec![
        op.query.mlir_tensor_type(),
        op.key.mlir_tensor_type(),
        op.value.mlir_tensor_type(),
    ];
    if let Some(mask) = &op.mask {
        arg_types.push(mask.mlir_tensor_type());
    }

    let result_ty = op.result.mlir_tensor_type();
    let attributes = format!(" attributes {{kernelforge.scale = {:.6}}}", op.scale);

    let mut body = String::new();
    let arg_list = arg_types
        .iter()
        .enumerate()
        .map(|(index, ty)| format!("%arg{}: {}", index, ty))
        .collect::<Vec<_>>()
        .join(", ");

    let stub_name = format!("{}_impl", op.name);

    let _ = writeln!(
        body,
        "  func.func @{}({}) -> {}{} {{",
        op.name, arg_list, result_ty, attributes
    );

    let call_args = (0..arg_types.len())
        .map(|index| format!("%arg{}", index))
        .collect::<Vec<_>>()
        .join(", ");

    body.push_str(&format!(
        "    %0 = func.call @{}({}) : ({}) -> {}\n",
        stub_name,
        call_args,
        arg_types.join(", "),
        result_ty
    ));
    body.push_str(&format!("    return %0 : {}\n", result_ty));
    body.push_str("  }\n");

    body.push_str(&format!(
        "  func.func private @{}({}) -> {}\n",
        stub_name,
        arg_types.join(", "),
        result_ty
    ));

    body
}

fn emit_mlp_function(op: &MlpOp) -> String {
    let input_ty = op.input.mlir_tensor_type();
    let _hidden_ty = op.hidden.mlir_tensor_type();
    let output_ty = op.output.mlir_tensor_type();

    let hidden_shape = op
        .hidden
        .shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join("x");

    let attributes = format!(
        " attributes {{kernelforge.activation = \"{}\", kernelforge.hidden = \"{}\"}}",
        op.activation.as_str(),
        hidden_shape
    );

    let stub_name = format!("{}_impl", op.name);

    let mut body = String::new();
    let _ = writeln!(
        body,
        "  func.func @{}(%arg0: {}) -> {}{} {{",
        op.name, input_ty, output_ty, attributes
    );
    body.push_str(&format!(
        "    %0 = func.call @{}(%arg0) : ({}) -> {}\n",
        stub_name, input_ty, output_ty
    ));
    body.push_str(&format!("    return %0 : {}\n", output_ty));
    body.push_str("  }\n");
    body.push_str(&format!(
        "  func.func private @{}(%arg0: {}) -> {}\n",
        stub_name, input_ty, output_ty
    ));

    body
}

fn emit_layer_norm_function(op: &LayerNormOp) -> String {
    let input_ty = op.input.mlir_tensor_type();
    let result_ty = op.result.mlir_tensor_type();

    let attributes = format!(" attributes {{kernelforge.epsilon = {:.6}}}", op.epsilon);

    let stub_name = format!("{}_impl", op.name);

    let mut body = String::new();
    let _ = writeln!(
        body,
        "  func.func @{}(%arg0: {}) -> {}{} {{",
        op.name, input_ty, result_ty, attributes
    );
    body.push_str(&format!(
        "    %0 = func.call @{}(%arg0) : ({}) -> {}\n",
        stub_name, input_ty, result_ty
    ));
    body.push_str(&format!("    return %0 : {}\n", result_ty));
    body.push_str("  }\n");
    body.push_str(&format!(
        "  func.func private @{}(%arg0: {}) -> {}\n",
        stub_name, input_ty, result_ty
    ));

    body
}
