//! Audit reports and verification results.

use crate::optimizer::HardwareSummary;
use crate::plan::OptimizationPlan;
use crate::workload::WorkloadSignature;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Result of verification comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum VerificationResult {
    Passed {
        max_abs_error: f32,
        max_rel_error: f32,
    },
    Failed {
        reason: String,
    },
}

/// Complete audit report for a verification run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    /// The optimization plan that was verified.
    pub plan: OptimizationPlan,

    /// Workload signature for cache keying.
    pub workload_signature: WorkloadSignature,

    /// Hardware information (if GPU was used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_info: Option<HardwareSummary>,

    /// Verification result.
    pub verification_result: VerificationResult,

    /// CPU reference execution time (ms).
    pub cpu_reference_time_ms: f64,

    /// GPU kernel time (ms) if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_kernel_time_ms: Option<f64>,

    /// Total verification time (ms).
    pub total_verification_time_ms: f64,

    /// Whether the plan was accepted.
    pub accepted: bool,

    /// Reason for rejection if not accepted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejection_reason: Option<String>,
}

impl AuditReport {
    /// Create a rejection report.
    pub fn rejected(plan: OptimizationPlan, sig: WorkloadSignature, reason: String) -> Self {
        Self {
            plan,
            workload_signature: sig,
            hardware_info: None,
            verification_result: VerificationResult::Failed {
                reason: reason.clone(),
            },
            cpu_reference_time_ms: 0.0,
            gpu_kernel_time_ms: None,
            total_verification_time_ms: 0.0,
            accepted: false,
            rejection_reason: Some(reason),
        }
    }

    /// Save report to JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load report from JSON file.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let report = serde_json::from_str(&json)?;
        Ok(report)
    }
}

/// Cache of best verified plans per workload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BestPlansCache {
    /// Map from workload cache key to best plan + metrics.
    pub plans: HashMap<String, CachedPlan>,
}

/// A cached plan with performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPlan {
    pub plan: OptimizationPlan,
    pub cpu_time_ms: f64,
    pub gpu_time_ms: Option<f64>,
    pub max_abs_error: f32,
    pub verified_at: u64, // Unix timestamp
}

impl BestPlansCache {
    /// Load from file or create empty.
    pub fn load_or_create(path: impl AsRef<Path>) -> Self {
        if let Ok(json) = std::fs::read_to_string(path) {
            serde_json::from_str(&json).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    /// Save to file.
    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Insert a new plan if it's better than existing.
    pub fn insert_if_better(&mut self, key: &str, report: &AuditReport) -> bool {
        if !report.accepted {
            return false;
        }

        let max_abs_error = match &report.verification_result {
            VerificationResult::Passed { max_abs_error, .. } => *max_abs_error,
            _ => return false,
        };

        let new_plan = CachedPlan {
            plan: report.plan.clone(),
            cpu_time_ms: report.cpu_reference_time_ms,
            gpu_time_ms: report.gpu_kernel_time_ms,
            max_abs_error,
            verified_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };

        // Check if better than existing
        if let Some(existing) = self.plans.get(key) {
            // Prefer faster GPU time, or faster CPU time if no GPU
            let new_time = new_plan.gpu_time_ms.unwrap_or(new_plan.cpu_time_ms);
            let old_time = existing.gpu_time_ms.unwrap_or(existing.cpu_time_ms);
            if new_time >= old_time {
                return false;
            }
        }

        self.plans.insert(key.to_string(), new_plan);
        true
    }

    /// Get best plan for a workload signature.
    pub fn get(&self, sig: &WorkloadSignature) -> Option<&CachedPlan> {
        self.plans.get(&sig.cache_key())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_result_serialization() {
        let passed = VerificationResult::Passed {
            max_abs_error: 1e-5,
            max_rel_error: 1e-4,
        };
        let json = serde_json::to_string(&passed).unwrap();
        assert!(json.contains("Passed"));

        let failed = VerificationResult::Failed {
            reason: "too much error".into(),
        };
        let json = serde_json::to_string(&failed).unwrap();
        assert!(json.contains("Failed"));
    }
}
