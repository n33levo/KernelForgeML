//! Decoder block implementing self-attention + MLP with KV-cache support.

use crate::config::ModelConfig;
use crate::kv_cache::KVCache;
use crate::rope::RoPECache;
use crate::weights::DecoderWeights;
use anyhow::Result;
use kernelforge_kernels::attention::scaled_dot_product_attention;
use kernelforge_kernels::config::ActivationKind;
use kernelforge_kernels::layernorm::layer_norm;
use kernelforge_kernels::matmul::MatmulInputs;
use kernelforge_kernels::matmul::ReferenceMatmul;
use kernelforge_kernels::MatmulKernel;
use ndarray::{Array2, Array3, Axis};

pub struct DecoderBlock {
    config: ModelConfig,
    rope: RoPECache,
}

impl DecoderBlock {
    pub fn new(config: ModelConfig) -> Self {
        let rope = RoPECache::new(config.max_seq_len, config.head_dim(), config.rope_theta);
        Self { config, rope }
    }

    /// Forward pass for a single decoder layer.
    ///
    /// Inputs:
    /// - x: (batch=1, seq_len, d_model) hidden states
    /// - weights: layer weights
    /// - kv_cache: cache for this layer
    /// - position_offset: starting position (0 for prefill, >0 for decode)
    /// - is_prefill: whether this is prefill (true) or decode (false)
    ///
    /// Returns: (batch=1, seq_len, d_model) output
    pub fn forward(
        &self,
        x: &Array3<f32>,
        weights: &DecoderWeights,
        kv_cache: &mut KVCache,
        layer_idx: usize,
        position_offset: usize,
        is_prefill: bool,
    ) -> Result<Array3<f32>> {
        let (batch, _seq_len, d_model) = x.dim();
        assert!(batch == 1, "Only batch size 1 is supported, got {}", batch);
        assert_eq!(d_model, self.config.d_model);

        // 1. Pre-attention layernorm
        let x_norm = self.layer_norm_3d(x, &weights.ln1_weight, &weights.ln1_bias)?;

        // 2. Self-attention with KV-cache
        let attn_out = self.self_attention(
            &x_norm,
            weights,
            kv_cache,
            layer_idx,
            position_offset,
            is_prefill,
        )?;

        // Residual
        let x = x + &attn_out;

        // 3. Pre-MLP layernorm
        let x_norm = self.layer_norm_3d(&x, &weights.ln2_weight, &weights.ln2_bias)?;

        // 4. MLP
        let mlp_out = self.mlp(&x_norm, weights)?;

        // Residual
        let x = x + &mlp_out;

        Ok(x)
    }

    fn self_attention(
        &self,
        x: &Array3<f32>,
        weights: &DecoderWeights,
        kv_cache: &mut KVCache,
        layer_idx: usize,
        position_offset: usize,
        _is_prefill: bool,
    ) -> Result<Array3<f32>> {
        let (_batch, _seq_len, _d_model) = x.dim();
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim();

        // Flatten to 2D for matmul
        let x_flat = x.index_axis(Axis(0), 0).to_owned(); // (seq_len, d_model)

        // QKV projection
        let q = self.matmul(&x_flat, &weights.q_proj)?; // (seq_len, d_model)
        let k_new = self.matmul(&x_flat, &weights.k_proj)?;
        let v_new = self.matmul(&x_flat, &weights.v_proj)?;

        // Apply RoPE to Q and K per head
        let mut q_rope = q.clone();
        let mut k_rope = k_new.clone();

        // Apply RoPE per head
        for h in 0..n_heads {
            let start_idx = h * head_dim;
            let end_idx = (h + 1) * head_dim;

            let q_head = q_rope.slice_mut(ndarray::s![.., start_idx..end_idx]);
            let k_head = k_rope.slice_mut(ndarray::s![.., start_idx..end_idx]);

            self.rope.apply(q_head, position_offset);
            self.rope.apply(k_head, position_offset);
        }

        // Update KV-cache
        let k_3d = k_rope.insert_axis(Axis(0)); // (1, seq_len, d_model)
        let v_3d = v_new.insert_axis(Axis(0));
        kv_cache.append(layer_idx, k_3d, v_3d);

        // Get full K/V from cache
        let (k_cached, v_cached) = kv_cache.get(layer_idx);
        let k_full = k_cached.index_axis(Axis(0), 0).to_owned(); // (cached_len, d_model)
        let v_full = v_cached.index_axis(Axis(0), 0).to_owned();

        // Compute attention
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn_output = scaled_dot_product_attention(
            q_rope.view(),
            k_full.view(),
            v_full.view(),
            None, // No mask for now (can add causal mask if needed)
            scale,
        )?;

        // Output projection
        let out = self.matmul(&attn_output, &weights.out_proj)?;

        // Reshape back to 3D
        Ok(out.insert_axis(Axis(0)))
    }

    fn mlp(&self, x: &Array3<f32>, weights: &DecoderWeights) -> Result<Array3<f32>> {
        let x_flat = x.index_axis(Axis(0), 0).to_owned();

        // Up projection + GELU
        let hidden = self.matmul(&x_flat, &weights.mlp_up)?;
        let hidden = self.gelu(hidden);

        // Down projection
        let out = self.matmul(&hidden, &weights.mlp_down)?;

        Ok(out.insert_axis(Axis(0)))
    }

    fn matmul(&self, lhs: &Array2<f32>, rhs: &Array2<f32>) -> Result<Array2<f32>> {
        let kernel = ReferenceMatmul::new();
        let problem = kernelforge_kernels::config::MatmulProblem::new(
            lhs.nrows(),
            rhs.ncols(),
            lhs.ncols(),
            kernelforge_kernels::config::DataType::F32,
        );
        let inputs = MatmulInputs::new(lhs.view(), rhs.view(), None, ActivationKind::None);
        kernel.run(&problem, &inputs)
    }

    fn layer_norm_3d(
        &self,
        x: &Array3<f32>,
        weight: &Array2<f32>,
        bias: &Array2<f32>,
    ) -> Result<Array3<f32>> {
        let x_flat = x.index_axis(Axis(0), 0).to_owned();
        let normed = layer_norm(x_flat.view(), weight.view(), bias.view(), 1e-5)?;
        Ok(normed.insert_axis(Axis(0)))
    }

    fn gelu(&self, mut x: Array2<f32>) -> Array2<f32> {
        let c = (2.0 / std::f32::consts::PI).sqrt();
        x.mapv_inplace(|v| 0.5 * v * (1.0 + (c * (v + 0.044715 * v.powi(3))).tanh()));
        x
    }
}
