//! Weight structures and safetensors loader.

use anyhow::{Context, Result};
use ndarray::Array2;
use safetensors::SafeTensors;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct DecoderWeights {
    // Attention
    pub q_proj: Array2<f32>,
    pub k_proj: Array2<f32>,
    pub v_proj: Array2<f32>,
    pub out_proj: Array2<f32>,

    // LayerNorm (pre-attention)
    pub ln1_weight: Array2<f32>,
    pub ln1_bias: Array2<f32>,

    // MLP
    pub mlp_up: Array2<f32>,
    pub mlp_down: Array2<f32>,

    // LayerNorm (pre-MLP)
    pub ln2_weight: Array2<f32>,
    pub ln2_bias: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub token_embeddings: Array2<f32>, // (vocab_size, d_model)
    pub layers: Vec<DecoderWeights>,
    pub final_ln_weight: Array2<f32>,
    pub final_ln_bias: Array2<f32>,
    pub lm_head: Array2<f32>, // (d_model, vocab_size)
}

impl ModelWeights {
    /// Load weights from safetensors format.
    /// Expected tensor naming: layers.{i}.{component}.weight
    pub fn load_safetensors(path: impl AsRef<Path>) -> Result<Self> {
        let data = fs::read(path).context("failed to read safetensors file")?;
        let tensors = SafeTensors::deserialize(&data).context("failed to parse safetensors")?;

        // Extract metadata (TODO: read from file metadata when safetensors supports it)
        let n_layers: usize = 12; // Default, should be passed or read from config

        // Load embeddings
        let token_embeddings = load_tensor_2d(&tensors, "token_embeddings")?;
        let lm_head = load_tensor_2d(&tensors, "lm_head")?;
        let final_ln_weight = load_tensor_2d(&tensors, "final_ln.weight")?;
        let final_ln_bias = load_tensor_2d(&tensors, "final_ln.bias")?;

        // Load layers
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let prefix = format!("layers.{}", i);
            layers.push(DecoderWeights {
                q_proj: load_tensor_2d(&tensors, &format!("{}.q_proj", prefix))?,
                k_proj: load_tensor_2d(&tensors, &format!("{}.k_proj", prefix))?,
                v_proj: load_tensor_2d(&tensors, &format!("{}.v_proj", prefix))?,
                out_proj: load_tensor_2d(&tensors, &format!("{}.out_proj", prefix))?,
                ln1_weight: load_tensor_2d(&tensors, &format!("{}.ln1.weight", prefix))?,
                ln1_bias: load_tensor_2d(&tensors, &format!("{}.ln1.bias", prefix))?,
                mlp_up: load_tensor_2d(&tensors, &format!("{}.mlp_up", prefix))?,
                mlp_down: load_tensor_2d(&tensors, &format!("{}.mlp_down", prefix))?,
                ln2_weight: load_tensor_2d(&tensors, &format!("{}.ln2.weight", prefix))?,
                ln2_bias: load_tensor_2d(&tensors, &format!("{}.ln2.bias", prefix))?,
            });
        }

        Ok(Self {
            token_embeddings,
            layers,
            final_ln_weight,
            final_ln_bias,
            lm_head,
        })
    }

    /// Create random weights for testing (not for real inference)
    pub fn random(vocab_size: usize, d_model: usize, n_layers: usize, d_ff: usize) -> Self {
        // Simple initialization with small random values
        fn random_array(shape: (usize, usize)) -> Array2<f32> {
            Array2::from_shape_fn(shape, |_| (fastrand::f32() - 0.5) * 0.04)
        }

        let token_embeddings = random_array((vocab_size, d_model));
        let lm_head = random_array((d_model, vocab_size));
        let final_ln_weight = Array2::ones((1, d_model));
        let final_ln_bias = Array2::zeros((1, d_model));

        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(DecoderWeights {
                q_proj: random_array((d_model, d_model)),
                k_proj: random_array((d_model, d_model)),
                v_proj: random_array((d_model, d_model)),
                out_proj: random_array((d_model, d_model)),
                ln1_weight: Array2::ones((1, d_model)),
                ln1_bias: Array2::zeros((1, d_model)),
                mlp_up: random_array((d_model, d_ff)),
                mlp_down: random_array((d_ff, d_model)),
                ln2_weight: Array2::ones((1, d_model)),
                ln2_bias: Array2::zeros((1, d_model)),
            });
        }

        Self {
            token_embeddings,
            layers,
            final_ln_weight,
            final_ln_bias,
            lm_head,
        }
    }
}

fn load_tensor_2d(tensors: &SafeTensors, name: &str) -> Result<Array2<f32>> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("tensor '{}' not found", name))?;

    let shape = view.shape();
    anyhow::ensure!(shape.len() == 2, "expected 2D tensor for {}", name);

    let data = view.data();
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Array2::from_shape_vec((shape[0], shape[1]), floats).context("failed to reshape tensor")
}
