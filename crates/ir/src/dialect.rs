//! Custom KernelForgeML dialect definitions.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    F16,
    BF16,
}

impl DataType {
    pub fn element_type(&self) -> &'static str {
        match self {
            DataType::F32 => "f32",
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DataType,
}

impl TensorSpec {
    pub fn new<N: Into<String>>(name: N, shape: Vec<usize>, dtype: DataType) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype,
        }
    }

    pub fn mlir_tensor_type(&self) -> String {
        if self.shape.is_empty() {
            return format!("tensor<{}>", self.dtype.element_type());
        }
        let dims = self
            .shape
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join("x");
        format!("tensor<{}x{}>", dims, self.dtype.element_type())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum ActivationKind {
    #[default]
    None,
    Relu,
    Gelu,
}

impl ActivationKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ActivationKind::None => "none",
            ActivationKind::Relu => "relu",
            ActivationKind::Gelu => "gelu",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulOp {
    pub name: String,
    pub lhs: TensorSpec,
    pub rhs: TensorSpec,
    pub result: TensorSpec,
    pub bias: Option<TensorSpec>,
    pub activation: ActivationKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionOp {
    pub name: String,
    pub query: TensorSpec,
    pub key: TensorSpec,
    pub value: TensorSpec,
    pub mask: Option<TensorSpec>,
    pub result: TensorSpec,
    pub scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpOp {
    pub name: String,
    pub input: TensorSpec,
    pub hidden: TensorSpec,
    pub output: TensorSpec,
    pub activation: ActivationKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormOp {
    pub name: String,
    pub input: TensorSpec,
    pub epsilon: f32,
    pub result: TensorSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    Matmul(MatmulOp),
    Attention(AttentionOp),
    Mlp(MlpOp),
    LayerNorm(LayerNormOp),
}

impl Operation {
    pub fn name(&self) -> &str {
        match self {
            Operation::Matmul(op) => &op.name,
            Operation::Attention(op) => &op.name,
            Operation::Mlp(op) => &op.name,
            Operation::LayerNorm(op) => &op.name,
        }
    }
}
