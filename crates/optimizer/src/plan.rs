//! Optimization plan - complete strategy for optimizing a workload.

use crate::knobs::OptimizationKnobs;
use serde::{Deserialize, Serialize};

/// Complete optimization plan proposed by an optimizer.
///
/// Contains the pass order and tuning knobs to apply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPlan {
    /// Ordered list of pass names to run.
    pub pass_order: Vec<String>,

    /// Tuning knobs for the passes.
    pub knobs: OptimizationKnobs,

    /// Target backend: "cpu" or "gpu"
    pub target: String,

    /// Optional reasoning from the LLM (for debugging/logging).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

impl OptimizationPlan {
    /// Create a new plan with default pass order for CPU.
    pub fn default_cpu() -> Self {
        Self {
            pass_order: vec![
                "fold-constants".into(),
                "fuse-matmul-activation".into(),
                "fuse-mlp-block".into(),
                "tile-matmul".into(),
                "vectorize-layernorm".into(),
                "eliminate-dead-ops".into(),
            ],
            knobs: OptimizationKnobs::default(),
            target: "cpu".into(),
            reasoning: None,
        }
    }

    /// Create a minimal plan for GPU (no fusion, basic tiling).
    pub fn default_gpu() -> Self {
        Self {
            pass_order: vec![
                "fold-constants".into(),
                "tile-matmul".into(),
                "eliminate-dead-ops".into(),
            ],
            knobs: OptimizationKnobs::for_gpu(),
            target: "gpu".into(),
            reasoning: None,
        }
    }

    /// Create plan from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Validate the plan.
    pub fn validate(&self) -> Result<(), String> {
        // Validate knobs
        self.knobs.validate()?;

        // Validate pass names
        let valid_passes = [
            "fold-constants",
            "fuse-matmul-activation",
            "fuse-mlp-block",
            "tile-matmul",
            "vectorize-layernorm",
            "eliminate-dead-ops",
        ];

        for pass in &self.pass_order {
            if !valid_passes.contains(&pass.as_str()) {
                return Err(format!("Unknown pass: {}", pass));
            }
        }

        // Validate target
        if self.target != "cpu" && self.target != "gpu" {
            return Err(format!("Invalid target: {}", self.target));
        }

        Ok(())
    }

    /// Normalize common target aliases to accepted values.
    pub fn normalize_target(&mut self) {
        let lower = self.target.to_ascii_lowercase();
        self.target = match lower.as_str() {
            "cpu" => "cpu".into(),
            "gpu" | "metal" | "mps" => "gpu".into(),
            _ => self.target.clone(),
        };
    }
}

impl Default for OptimizationPlan {
    fn default() -> Self {
        Self::default_cpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_serialization() {
        let plan = OptimizationPlan::default_cpu();
        let json = plan.to_json().unwrap();
        let parsed = OptimizationPlan::from_json(&json).unwrap();
        assert_eq!(plan.pass_order, parsed.pass_order);
    }

    #[test]
    fn test_normalize_target_aliases() {
        let mut plan = OptimizationPlan::default_gpu();
        plan.target = "metal".into();
        plan.normalize_target();
        assert_eq!(plan.target, "gpu");
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn test_plan_validation() {
        let plan = OptimizationPlan::default_cpu();
        assert!(plan.validate().is_ok());

        let mut bad_plan = plan.clone();
        bad_plan.pass_order.push("nonexistent-pass".into());
        assert!(bad_plan.validate().is_err());
    }
}
