//! Rotary Position Embeddings (RoPE) as used in LLaMA/TinyLlama.
//!
//! Reference: https://arxiv.org/abs/2104.09864

use ndarray::{Array2, ArrayViewMut2};

pub struct RoPECache {
    cos: Array2<f32>,
    sin: Array2<f32>,
}

impl RoPECache {
    pub fn new(max_seq_len: usize, head_dim: usize, theta: f32) -> Self {
        let mut cos = Array2::<f32>::zeros((max_seq_len, head_dim));
        let mut sin = Array2::<f32>::zeros((max_seq_len, head_dim));

        for pos in 0..max_seq_len {
            for i in (0..head_dim).step_by(2) {
                let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos[[pos, i]] = angle.cos();
                cos[[pos, i + 1]] = angle.cos();
                sin[[pos, i]] = angle.sin();
                sin[[pos, i + 1]] = angle.sin();
            }
        }

        Self { cos, sin }
    }

    /// Apply RoPE to a query or key tensor of shape (seq_len, n_heads, head_dim).
    /// Modifies in-place.
    pub fn apply(&self, mut qk: ArrayViewMut2<f32>, position_offset: usize) {
        let (seq_len, feat_dim) = qk.dim();
        assert!(position_offset + seq_len <= self.cos.nrows());

        for pos in 0..seq_len {
            let abs_pos = position_offset + pos;
            for d in (0..feat_dim).step_by(2) {
                let x0 = qk[[pos, d]];
                let x1 = qk[[pos, d + 1]];
                let c = self.cos[[abs_pos, d]];
                let s = self.sin[[abs_pos, d]];

                qk[[pos, d]] = x0 * c - x1 * s;
                qk[[pos, d + 1]] = x0 * s + x1 * c;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn rope_cache_basic() {
        let cache = RoPECache::new(128, 64, 10000.0);
        assert_eq!(cache.cos.dim(), (128, 64));
        assert_eq!(cache.sin.dim(), (128, 64));
    }

    #[test]
    fn rope_apply_shape() {
        let cache = RoPECache::new(128, 64, 10000.0);
        let mut qk = Array2::<f32>::ones((4, 64));
        cache.apply(qk.view_mut(), 0);
        // Should not panic, shape unchanged
        assert_eq!(qk.dim(), (4, 64));
    }
}
