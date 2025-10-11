//! Minimal decoder-only LLM inference with KV-cache.
//!
//! This crate implements a real transformer decoder with:
//! - KV-cache for incremental decoding
//! - RoPE (Rotary Position Embeddings)
//! - Safetensors weight loading
//! - Prefill vs. decode metrics
//!
//! Designed to work with GPT-2 or TinyLlama architectures.

pub mod cerebras;
pub mod config;
pub mod decoder;
pub mod kv_cache;
pub mod model;
pub mod rope;
pub mod tokenizer;
pub mod weights;

pub use cerebras::{CerebrasClient, CerebrasMetrics, InferenceComparison};
pub use config::ModelConfig;
pub use model::{GenerationMetrics, LLMModel};
pub use weights::ModelWeights;
