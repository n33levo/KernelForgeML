//! Full model with embeddings, decoder stack, and LM head.

use crate::config::ModelConfig;
use crate::decoder::DecoderBlock;
use crate::kv_cache::KVCache;
use crate::weights::ModelWeights;
use anyhow::Result;
use kernelforge_kernels::layernorm::layer_norm;
use ndarray::{Array1, Array3, Axis};
use std::time::Instant;

pub struct LLMModel {
    config: ModelConfig,
    weights: ModelWeights,
    decoder: DecoderBlock,
    kv_cache: KVCache,
}

#[derive(Debug)]
pub struct GenerationMetrics {
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub tokens_generated: usize,
    pub tokens_per_sec: f64,
    pub kv_cache_bytes_per_token: usize,
}

impl LLMModel {
    pub fn new(config: ModelConfig, weights: ModelWeights) -> Self {
        let kv_cache = KVCache::new(config.n_layers, config.max_seq_len, config.d_model);
        let decoder = DecoderBlock::new(config.clone());

        Self {
            config,
            weights,
            decoder,
            kv_cache,
        }
    }

    /// Generate tokens auto-regressively.
    ///
    /// Returns (generated_tokens, metrics)
    pub fn generate(
        &mut self,
        input_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<(Vec<usize>, GenerationMetrics)> {
        self.kv_cache.reset();

        // Prefill phase
        let prefill_start = Instant::now();
        let mut logits = self.forward(input_ids, true)?;
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut current_token = self.sample(&logits, temperature);
        generated.push(current_token);

        // Decode phase
        let decode_start = Instant::now();
        for _ in 1..max_new_tokens {
            logits = self.forward(&[current_token], false)?;
            current_token = self.sample(&logits, temperature);
            generated.push(current_token);

            if current_token == 0 {
                // EOS (assuming token 0 is EOS)
                break;
            }
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        let tokens_per_sec = if decode_ms > 0.0 {
            (generated.len() as f64 / decode_ms) * 1000.0
        } else {
            0.0
        };

        let metrics = GenerationMetrics {
            prefill_ms,
            decode_ms,
            tokens_generated: generated.len(),
            tokens_per_sec,
            kv_cache_bytes_per_token: self.kv_cache.bytes_per_token(),
        };

        Ok((generated, metrics))
    }

    /// Forward pass: embed → decoder stack → LM head
    fn forward(&mut self, token_ids: &[usize], is_prefill: bool) -> Result<Array1<f32>> {
        let seq_len = token_ids.len();

        // Embedding lookup
        let mut hidden = Array3::<f32>::zeros((1, seq_len, self.config.d_model));
        for (i, &token_id) in token_ids.iter().enumerate() {
            hidden
                .slice_mut(ndarray::s![0, i, ..])
                .assign(&self.weights.token_embeddings.row(token_id));
        }

        // Decoder stack
        let position_offset = if is_prefill {
            0
        } else {
            self.kv_cache.current_len
        };

        for (layer_idx, layer_weights) in self.weights.layers.iter().enumerate() {
            hidden = self.decoder.forward(
                &hidden,
                layer_weights,
                &mut self.kv_cache,
                layer_idx,
                position_offset,
                is_prefill,
            )?;
        }

        // Final layernorm
        let hidden_flat = hidden.index_axis(Axis(0), 0).to_owned();
        let normed = layer_norm(
            hidden_flat.view(),
            self.weights.final_ln_weight.view(),
            self.weights.final_ln_bias.view(),
            1e-5,
        )?;

        // LM head: (seq_len, d_model) @ (d_model, vocab_size) -> (seq_len, vocab_size)
        let logits = normed.dot(&self.weights.lm_head);

        // Return logits for last token
        Ok(logits.row(seq_len - 1).to_owned())
    }

    fn sample(&self, logits: &Array1<f32>, temperature: f32) -> usize {
        // Simple greedy sampling (can extend to top-k/top-p)
        if temperature < 1e-5 {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        } else {
            // Temperature sampling
            let scaled: Vec<f32> = logits.iter().map(|&x| (x / temperature).exp()).collect();
            let sum: f32 = scaled.iter().sum();
            let probs: Vec<f32> = scaled.iter().map(|&x| x / sum).collect();

            // Sample from distribution (simple version)
            let rand_val: f32 = fastrand::f32();
            let mut cumsum = 0.0;
            for (idx, &prob) in probs.iter().enumerate() {
                cumsum += prob;
                if rand_val < cumsum {
                    return idx;
                }
            }
            probs.len() - 1
        }
    }
}
