//! Cerebras API integration for high-quality inference comparison.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Serialize)]
struct CerebrasMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct CerebrasRequest {
    model: String,
    messages: Vec<CerebrasMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CerebrasChoice {
    message: CerebrasResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CerebrasResponseMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CerebrasUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct CerebrasResponse {
    choices: Vec<CerebrasChoice>,
    usage: CerebrasUsage,
}

#[derive(Debug)]
pub struct CerebrasMetrics {
    pub response_time_ms: f64,
    pub tokens_generated: usize,
    pub tokens_per_sec: f64,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

pub struct CerebrasClient {
    api_key: String,
    model: String,
    base_url: String,
}

impl CerebrasClient {
    pub fn new(api_key: String, model: Option<String>) -> Self {
        Self {
            api_key,
            model: model.unwrap_or_else(|| "llama3.1-8b".to_string()),
            base_url: "https://api.cerebras.ai/v1".to_string(),
        }
    }

    /// Generate text using Cerebras API for comparison with local inference
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<(String, CerebrasMetrics)> {
        let client = reqwest::Client::new();

        let request = CerebrasRequest {
            model: self.model.clone(),
            messages: vec![CerebrasMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: Some(max_tokens),
            temperature: Some(temperature),
            stream: false,
        };

        let start_time = Instant::now();

        let response = client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to send request to Cerebras API: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Cerebras API error: {} - {}", status, error_text));
        }

        let cerebras_response: CerebrasResponse = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse Cerebras response: {}", e))?;

        let response_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        let generated_text = cerebras_response
            .choices
            .first()
            .map(|choice| choice.message.content.clone())
            .unwrap_or_default();

        let tokens_per_sec = if response_time_ms > 0.0 {
            (cerebras_response.usage.completion_tokens as f64 / response_time_ms) * 1000.0
        } else {
            0.0
        };

        let metrics = CerebrasMetrics {
            response_time_ms,
            tokens_generated: cerebras_response.usage.completion_tokens,
            tokens_per_sec,
            prompt_tokens: cerebras_response.usage.prompt_tokens,
            completion_tokens: cerebras_response.usage.completion_tokens,
        };

        Ok((generated_text, metrics))
    }

    /// Get available models (for future enhancement)
    pub fn available_models() -> Vec<&'static str> {
        vec![
            "llama3.1-8b",
            "llama-3.3-70b",
            "llama-4-scout-17b-16e-instruct",
            "qwen-3-32b",
            "gpt-oss-120b",
        ]
    }
}

/// Comparison between local and Cerebras inference
#[derive(Debug)]
pub struct InferenceComparison {
    pub local_output: String,
    pub local_tokens_per_sec: f64,
    pub local_time_ms: f64,
    pub cerebras_output: String,
    pub cerebras_tokens_per_sec: f64,
    pub cerebras_time_ms: f64,
    pub speedup_factor: f64,
}

impl InferenceComparison {
    pub fn new(
        local_output: String,
        local_tokens_per_sec: f64,
        local_time_ms: f64,
        cerebras_output: String,
        cerebras_tokens_per_sec: f64,
        cerebras_time_ms: f64,
    ) -> Self {
        let speedup_factor = if local_tokens_per_sec > 0.0 {
            cerebras_tokens_per_sec / local_tokens_per_sec
        } else {
            0.0
        };

        Self {
            local_output,
            local_tokens_per_sec,
            local_time_ms,
            cerebras_output,
            cerebras_tokens_per_sec,
            cerebras_time_ms,
            speedup_factor,
        }
    }

    pub fn print_comparison(&self) {
        println!("\n=== Inference Comparison ===");
        println!("Local Output: {}", self.local_output);
        println!("Cerebras Output: {}", self.cerebras_output);
        println!("\n=== Performance Comparison ===");
        println!(
            "Local: {:.2} tokens/sec ({:.2} ms total)",
            self.local_tokens_per_sec, self.local_time_ms
        );
        println!(
            "Cerebras: {:.2} tokens/sec ({:.2} ms total)",
            self.cerebras_tokens_per_sec, self.cerebras_time_ms
        );
        println!(
            "Speedup Factor: {:.2}x (positive = Cerebras faster)",
            self.speedup_factor
        );
    }
}
