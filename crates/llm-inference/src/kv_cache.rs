//! KV-cache for incremental decoding.
//!
//! During prefill, we compute full attention over the prompt.
//! During decode, we only compute new K/V for the current token
//! and concatenate with cached past keys/values.

use ndarray::{Array3, Axis};

pub struct KVCache {
    /// Cached keys: (n_layers, current_len, d_model)
    pub keys: Vec<Array3<f32>>,
    /// Cached values: (n_layers, current_len, d_model)
    pub values: Vec<Array3<f32>>,
    /// Current sequence length in cache
    pub current_len: usize,
    /// Maximum capacity
    pub max_len: usize,
}

impl KVCache {
    pub fn new(n_layers: usize, max_len: usize, d_model: usize) -> Self {
        let keys = (0..n_layers)
            .map(|_| Array3::<f32>::zeros((1, 0, d_model)))
            .collect();
        let values = (0..n_layers)
            .map(|_| Array3::<f32>::zeros((1, 0, d_model)))
            .collect();

        Self {
            keys,
            values,
            current_len: 0,
            max_len,
        }
    }

    /// Append new keys/values for a given layer.
    /// new_k, new_v: (batch=1, new_seq_len, d_model)
    pub fn append(&mut self, layer: usize, new_k: Array3<f32>, new_v: Array3<f32>) {
        assert_eq!(new_k.dim().0, 1, "only batch=1 supported");
        let new_len = new_k.dim().1;

        if self.current_len == 0 {
            // First append (prefill)
            self.keys[layer] = new_k;
            self.values[layer] = new_v;
            self.current_len = new_len;
        } else {
            // Incremental decode
            self.keys[layer] =
                ndarray::concatenate(Axis(1), &[self.keys[layer].view(), new_k.view()]).unwrap();
            self.values[layer] =
                ndarray::concatenate(Axis(1), &[self.values[layer].view(), new_v.view()]).unwrap();
            self.current_len += new_len;
        }

        assert!(
            self.current_len <= self.max_len,
            "exceeded max sequence length"
        );
    }

    /// Get cached K/V for a layer: returns (batch, current_len, d_model)
    pub fn get(&self, layer: usize) -> (&Array3<f32>, &Array3<f32>) {
        (&self.keys[layer], &self.values[layer])
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
        for layer in 0..self.keys.len() {
            self.keys[layer] = Array3::<f32>::zeros((1, 0, self.keys[layer].dim().2));
            self.values[layer] = Array3::<f32>::zeros((1, 0, self.values[layer].dim().2));
        }
    }

    /// Bytes per token for this cache
    pub fn bytes_per_token(&self) -> usize {
        let d_model = self.keys[0].dim().2;
        let n_layers = self.keys.len();
        // 2 (K+V) * n_layers * d_model * sizeof(f32)
        2 * n_layers * d_model * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_prefill_decode() {
        let mut cache = KVCache::new(2, 128, 64);

        // Prefill: 10 tokens
        let k_prefill = Array3::<f32>::ones((1, 10, 64));
        let v_prefill = Array3::<f32>::ones((1, 10, 64));
        cache.append(0, k_prefill, v_prefill);
        assert_eq!(cache.current_len, 10);

        // Decode: 1 token
        let k_decode = Array3::<f32>::ones((1, 1, 64));
        let v_decode = Array3::<f32>::ones((1, 1, 64));
        cache.append(0, k_decode, v_decode);
        assert_eq!(cache.current_len, 11);

        let (k, v) = cache.get(0);
        assert_eq!(k.dim(), (1, 11, 64));
        assert_eq!(v.dim(), (1, 11, 64));
    }

    #[test]
    fn kv_cache_bytes_per_token() {
        let cache = KVCache::new(12, 1024, 768);
        // 2 * 12 layers * 768 * 4 bytes = 73728 bytes/token
        assert_eq!(cache.bytes_per_token(), 73728);
    }
}
