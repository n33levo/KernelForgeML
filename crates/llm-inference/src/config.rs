//! Model configuration matching GPT-2 / TinyLlama architectures.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
}

impl ModelConfig {
    /// GPT-2 small config (124M params)
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            d_model: 768,
            n_layers: 12,
            n_heads: 12,
            d_ff: 3072,
            max_seq_len: 1024,
            rope_theta: 10000.0,
        }
    }

    /// Tiny test config for quick iteration
    pub fn tiny() -> Self {
        Self {
            vocab_size: 1000,
            d_model: 128,
            n_layers: 2,
            n_heads: 4,
            d_ff: 512,
            max_seq_len: 256,
            rope_theta: 10000.0,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}
