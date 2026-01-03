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

/// Structured rejection reason for failed verifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RejectReason {
    /// Plan validation failed (invalid knobs, unknown passes, etc.)
    Validation {
        detail: String,
    },
    /// GPU execution failed at a specific stage
    GpuExecutionFailed {
        stage: String,
        shape: String,
        detail: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        plan_knobs: Option<String>,
    },
    /// Numeric mismatch between CPU reference and GPU output
    NumericMismatch {
        max_abs_error: f32,
        tolerance: f32,
        detail: String,
    },
    /// GPU not available
    GpuUnavailable {
        detail: String,
    },
}

impl RejectReason {
    pub fn to_string(&self) -> String {
        match self {
            RejectReason::Validation { detail } => {
                format!("Validation failed: {}", detail)
            }
            RejectReason::GpuExecutionFailed { stage, shape, detail, plan_knobs } => {
                let mut msg = format!("GPU execution failed at stage '{}' (shape {}): {}", stage, shape, detail);
                if let Some(knobs) = plan_knobs {
                    msg.push_str(&format!("\nPlan knobs: {}", knobs));
                }
                msg
            }
            RejectReason::NumericMismatch { max_abs_error, tolerance, detail } => {
                format!("Numeric mismatch: max_error={:.2e} > tolerance={:.2e}\n{}", max_abs_error, tolerance, detail)
            }
            RejectReason::GpuUnavailable { detail } => {
                format!("GPU unavailable: {}", detail)
            }
        }
    }
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

    /// Structured reason for rejection if not accepted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejection_reason: Option<RejectReason>,
}

impl AuditReport {
    /// Create a rejection report with structured reason.
    pub fn rejected_with_reason(plan: OptimizationPlan, sig: WorkloadSignature, reason: RejectReason) -> Self {
        Self {
            plan,
            workload_signature: sig,
            hardware_info: None,
            verification_result: VerificationResult::Failed {
                reason: reason.to_string(),
            },
            cpu_reference_time_ms: 0.0,
            gpu_kernel_time_ms: None,
            total_verification_time_ms: 0.0,
            accepted: false,
            rejection_reason: Some(reason),
        }
    }

    /// Create a rejection report (legacy string-based).
    pub fn rejected(plan: OptimizationPlan, sig: WorkloadSignature, reason: String) -> Self {
        Self::rejected_with_reason(plan, sig, RejectReason::Validation { detail: reason })
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
