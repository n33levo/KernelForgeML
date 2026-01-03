//! Optimizer trait and implementations (Heuristic + LLM).

use crate::plan::OptimizationPlan;
use crate::knobs::OptimizationKnobs;
use crate::workload::WorkloadSignature;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Summary of hardware capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSummary {
    pub backend: String,
    pub device_name: String,
    pub supports_timestamps: bool,
}

impl HardwareSummary {
    pub fn cpu() -> Self {
        Self {
            backend: "cpu".into(),
            device_name: "CPU (Rayon)".into(),
            supports_timestamps: true,
        }
    }

    pub fn from_gpu_info(info: &kernelforge_backend_gpu::runtime::GpuDeviceInfo) -> Self {
        Self {
            backend: info.backend.clone(),
            device_name: info.name.clone(),
            supports_timestamps: info.supports_timestamps,
        }
    }
}

/// Summary of the workload for the optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadSummary {
    pub signature: WorkloadSignature,
    pub total_flops: u64,
    pub memory_bytes: u64,
}

impl WorkloadSummary {
    pub fn from_signature(sig: &WorkloadSignature) -> Self {
        let (m, n, k) = sig.mnk();
        let m = m as u64;
        let n = n as u64;
        let k = k as u64;
        
        // 2*M*N*K for matmul
        let matmul_flops = 2 * m * n * k;
        // LayerNorm: ~5*M*N (mean, var, normalize, scale, bias)
        let ln_flops = if sig.has_layernorm { 5 * m * n } else { 0 };
        // Attention adds ~4*seq^2*d_k flops
        let attn_flops = if sig.has_attention { 4 * m * m * k } else { 0 };
        
        // Memory: input (M*K) + weight (K*N) + output (M*N) in f32
        let memory_bytes = ((m * k + k * n + m * n) * 4) as u64;

        Self {
            signature: sig.clone(),
            total_flops: matmul_flops + ln_flops + attn_flops,
            memory_bytes,
        }
    }
}

/// Trait for optimization proposers.
pub trait Optimizer: Send + Sync {
    /// Name of this optimizer.
    fn name(&self) -> &str;

    /// Propose an optimization plan for the given workload and hardware.
    fn propose(
        &self,
        workload: &WorkloadSummary,
        hardware: &HardwareSummary,
    ) -> Result<OptimizationPlan>;
}

/// Heuristic-based optimizer (no network, deterministic).
pub struct HeuristicOptimizer;

impl Optimizer for HeuristicOptimizer {
    fn name(&self) -> &str {
        "heuristic"
    }

    fn propose(
        &self,
        workload: &WorkloadSummary,
        hardware: &HardwareSummary,
    ) -> Result<OptimizationPlan> {
        let is_gpu = hardware.backend != "cpu";
        let sig = &workload.signature;
        let (m, n, k) = sig.mnk();

        // Heuristic: choose tile sizes based on problem size
        let (tile_m, tile_n, tile_k) = if m * n * k < 100_000 {
            // Small problem: smaller tiles
            (32, 32, 16)
        } else if m * n * k < 10_000_000 {
            // Medium problem
            (64, 64, 32)
        } else {
            // Large problem
            (128, 128, 64)
        };

        let knobs = OptimizationKnobs {
            tile_m,
            tile_n,
            tile_k,
            vector_width: if is_gpu { 4 } else { 8 },
            enable_fuse_matmul_activation: !is_gpu, // GPU shader doesn't support fusion
            enable_fuse_mlp: !is_gpu,
            enable_fold_constants: true,
            layernorm_epsilon: 1e-5,
        };

        let pass_order = if is_gpu {
            vec![
                "fold-constants".into(),
                "tile-matmul".into(),
                "eliminate-dead-ops".into(),
            ]
        } else {
            vec![
                "fold-constants".into(),
                "fuse-matmul-activation".into(),
                "fuse-mlp-block".into(),
                "tile-matmul".into(),
                "vectorize-layernorm".into(),
                "eliminate-dead-ops".into(),
            ]
        };

        Ok(OptimizationPlan {
            pass_order,
            knobs,
            target: if is_gpu { "gpu" } else { "cpu" }.into(),
            reasoning: Some(format!(
                "Heuristic: {}x{}x{} → tiles {}x{}x{}, {}",
                m, n, k, tile_m, tile_n, tile_k,
                if is_gpu { "GPU mode" } else { "CPU mode" }
            )),
        })
    }
}

/// LLM-based optimizer that calls an OpenAI-compatible API.
pub struct LlmOptimizer {
    api_endpoint: String,
    api_key: String,
    model: String,
}

impl LlmOptimizer {
    /// Create from environment variables.
    /// Expects: KERNELFORGE_LLM_ENDPOINT, KERNELFORGE_LLM_API_KEY, KERNELFORGE_LLM_MODEL
    pub fn from_env() -> Result<Self> {
        let api_endpoint = std::env::var("KERNELFORGE_LLM_ENDPOINT")
            .unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".into());
        let api_key = std::env::var("KERNELFORGE_LLM_API_KEY")
            .map_err(|_| anyhow::anyhow!("KERNELFORGE_LLM_API_KEY not set"))?;
        let model = std::env::var("KERNELFORGE_LLM_MODEL")
            .unwrap_or_else(|_| "gpt-4o-mini".into());

        Ok(Self {
            api_endpoint,
            api_key,
            model,
        })
    }

    /// Create with explicit configuration.
    pub fn new(api_endpoint: String, api_key: String, model: String) -> Self {
        Self {
            api_endpoint,
            api_key,
            model,
        }
    }

    fn build_prompt(&self, workload: &WorkloadSummary, hardware: &HardwareSummary) -> String {
        let (m, n, k) = workload.signature.mnk();
        format!(
            r#"You are a compiler optimization expert. Given a workload and hardware, propose an optimization plan.

WORKLOAD:
- Sequence length: {}, Model dim: {}, Head dim: {}
- Has Attention: {}
- Has LayerNorm: {}
- Total FLOPs: {}
- Memory: {} bytes

HARDWARE:
- Backend: {}
- Device: {}
- Supports GPU timestamps: {}

AVAILABLE PASSES (in order you can choose):
1. fold-constants - Fold compile-time constant expressions
2. fuse-matmul-activation - Fuse activation into matmul ops
3. fuse-mlp-block - Fuse MLP block pattern
4. tile-matmul - Apply cache-aware tiling
5. vectorize-layernorm - Vectorize layer normalization
6. eliminate-dead-ops - Remove unused operations

CONSTRAINTS:
- GPU backend does NOT support fused activation or bias yet
- tile_m, tile_n, tile_k must be powers of 2 between 16 and 128
- vector_width must be 4 or 8
- target must be exactly "cpu" or "gpu" (do NOT return "metal", "mps", or other variants)

Respond with ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "pass_order": ["pass-name-1", "pass-name-2", ...],
  "knobs": {{
    "tile_m": 64,
    "tile_n": 64,
    "tile_k": 32,
    "vector_width": 8,
    "enable_fuse_matmul_activation": true,
    "enable_fuse_mlp": true,
    "enable_fold_constants": true,
    "layernorm_epsilon": 1e-5
  }},
  "target": "cpu",
  "reasoning": "Brief explanation"
}}"#,
            m, n, k,
            workload.signature.has_attention,
            workload.signature.has_layernorm,
            workload.total_flops,
            workload.memory_bytes,
            hardware.backend,
            hardware.device_name,
            hardware.supports_timestamps
        )
    }
}

impl Optimizer for LlmOptimizer {
    fn name(&self) -> &str {
        "llm"
    }

    fn propose(
        &self,
        workload: &WorkloadSummary,
        hardware: &HardwareSummary,
    ) -> Result<OptimizationPlan> {
        let prompt = self.build_prompt(workload, hardware);

        let request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a compiler optimization assistant. Respond only with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 500
        });

        let response = ureq::post(&self.api_endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(&request_body);

        match response {
            Ok(resp) => {
                let body: serde_json::Value = resp.into_json()?;
                let content = body["choices"][0]["message"]["content"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("No content in LLM response"))?;

                // Try to parse the JSON response
                match OptimizationPlan::from_json(content) {
                    Ok(mut plan) => {
                        plan.normalize_target();
                        // Validate the plan
                        if let Err(e) = plan.validate() {
                            tracing::warn!(
                                error = %e,
                                "LLM returned invalid plan, falling back to heuristic"
                            );
                            return HeuristicOptimizer.propose(workload, hardware);
                        }
                        Ok(plan)
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            content = %content,
                            "Failed to parse LLM response, falling back to heuristic"
                        );
                        HeuristicOptimizer.propose(workload, hardware)
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "LLM API call failed, falling back to heuristic");
                HeuristicOptimizer.propose(workload, hardware)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_optimizer() {
        let workload = WorkloadSummary::from_signature(&crate::workload::WorkloadSignature::microblock(64, 128, 256));
        let hardware = HardwareSummary::cpu();
        
        let plan = HeuristicOptimizer.propose(&workload, &hardware).unwrap();
        assert!(plan.validate().is_ok());
        assert_eq!(plan.target, "cpu");
    }

    #[test]
    fn test_workload_summary() {
        let sig = crate::workload::WorkloadSignature::matmul(64, 64, 64);
        let summary = WorkloadSummary::from_signature(&sig);
        
        // 2 * 64 * 64 * 64 = 524288
        assert_eq!(summary.total_flops, 524288);
    }
}
