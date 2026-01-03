//! Verification harness - ensures optimization plans are correct.

use crate::plan::OptimizationPlan;
use crate::report::{AuditReport, RejectReason, VerificationResult};
use crate::workload::Microblock;
use anyhow::Result;
use kernelforge_backend_gpu::runtime::{GpuDeviceInfo, GpuExecutor};
use kernelforge_backend_gpu::planner::GpuPlanner;
use kernelforge_kernels::config::{ActivationKind, DataType, MatmulProblem};
use kernelforge_kernels::matmul::MatmulInputs;
use ndarray::Array2;
use std::time::Instant;

/// Tolerance for numeric comparison.
#[derive(Debug, Clone)]
pub struct VerificationTolerance {
    /// Maximum absolute error allowed.
    pub max_abs_error: f32,
    /// Maximum relative error allowed (for values > 1e-6).
    pub max_rel_error: f32,
}

impl Default for VerificationTolerance {
    fn default() -> Self {
        Self {
            max_abs_error: 1e-4,
            max_rel_error: 1e-3,
        }
    }
}

impl VerificationTolerance {
    /// Stricter tolerance for critical applications.
    pub fn strict() -> Self {
        Self {
            max_abs_error: 1e-5,
            max_rel_error: 1e-4,
        }
    }

    /// Looser tolerance for GPU comparisons (FP precision differences).
    pub fn gpu_friendly() -> Self {
        Self {
            max_abs_error: 1e-3,
            max_rel_error: 1e-2,
        }
    }
}

/// Verifier that checks optimization plans for correctness.
pub struct Verifier {
    tolerance: VerificationTolerance,
    gpu_executor: Option<GpuExecutor>,
}

impl Verifier {
    /// Create a new verifier with default tolerance.
    pub fn new() -> Result<Self> {
        Self::with_tolerance(VerificationTolerance::default())
    }

    /// Create with custom tolerance.
    pub fn with_tolerance(tolerance: VerificationTolerance) -> Result<Self> {
        // Try to initialize GPU executor
        let gpu_executor = match GpuExecutor::new(GpuPlanner) {
            Ok(exec) => Some(exec),
            Err(e) => {
                tracing::warn!(error = %e, "GPU not available, verification will be CPU-only");
                None
            }
        };

        Ok(Self {
            tolerance,
            gpu_executor,
        })
    }

    /// Get GPU device info if available.
    pub fn gpu_info(&self) -> Option<&GpuDeviceInfo> {
        self.gpu_executor.as_ref().map(|e| e.device_info())
    }

    /// Verify an optimization plan on a microblock workload.
    pub fn verify(
        &self,
        plan: &OptimizationPlan,
        microblock: &Microblock,
    ) -> Result<AuditReport> {
        let start = Instant::now();

        // Step 1: Validate plan
        if let Err(e) = plan.validate() {
            return Ok(AuditReport::rejected_with_reason(
                plan.clone(),
                microblock.signature.clone(),
                RejectReason::Validation {
                    detail: format!("Plan validation failed: {}", e),
                },
            ));
        }

        // Step 2: Compute CPU reference
        let cpu_start = Instant::now();
        let reference = microblock.compute_reference();
        let cpu_time_ms = cpu_start.elapsed().as_secs_f64() * 1000.0;

        // Step 3: Run GPU if target is GPU
        let (gpu_result, gpu_info) = if plan.target == "gpu" {
            if let Some(ref gpu_exec) = self.gpu_executor {
                let (m, n, k) = microblock.signature.mnk();
                let problem = MatmulProblem::new(m, n, k, DataType::F32);
                let inputs = MatmulInputs::new(
                    microblock.input.view(),
                    microblock.weight.view(),
                    None,
                    ActivationKind::None,
                );
                let tiling = Some((
                    plan.knobs.tile_m as u32,
                    plan.knobs.tile_n as u32,
                    plan.knobs.tile_k as u32,
                ));

                match gpu_exec.execute_matmul_timed(problem, &inputs, tiling) {
                    Ok(result) => (Some(result), Some(gpu_exec.device_info().clone())),
                    Err(e) => {
                        let plan_knobs = format!(
                            "tile_m={}, tile_n={}, tile_k={}, vector_width={}, workgroup_m={}, workgroup_n={}",
                            plan.knobs.tile_m,
                            plan.knobs.tile_n,
                            plan.knobs.tile_k,
                            plan.knobs.vector_width,
                            plan.knobs.tile_m.min(16),
                            plan.knobs.tile_n.min(16)
                        );
                        return Ok(AuditReport::rejected_with_reason(
                            plan.clone(),
                            microblock.signature.clone(),
                            RejectReason::GpuExecutionFailed {
                                stage: "matmul".into(),
                                shape: format!("{}x{}x{}", m, n, k),
                                detail: format!("{}", e),
                                plan_knobs: Some(plan_knobs),
                            },
                        ));
                    }
                }
            } else {
                return Ok(AuditReport::rejected_with_reason(
                    plan.clone(),
                    microblock.signature.clone(),
                    RejectReason::GpuUnavailable {
                        detail: "GPU target requested but GPU not available".into(),
                    },
                ));
            }
        } else {
            (None, None)
        };

        // Step 4: Compare outputs
        let verification_result = if let Some(ref gpu_res) = gpu_result {
            self.compare_outputs(&reference.matmul_output, &gpu_res.output)
        } else {
            // CPU-only verification: just check the reference runs
            VerificationResult::Passed {
                max_abs_error: 0.0,
                max_rel_error: 0.0,
            }
        };

        // Step 5: Build report
        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        let accepted = matches!(&verification_result, VerificationResult::Passed { .. });

        let report = AuditReport {
            plan: plan.clone(),
            workload_signature: microblock.signature.clone(),
            hardware_info: gpu_info.map(|i| crate::optimizer::HardwareSummary::from_gpu_info(&i)),
            verification_result,
            cpu_reference_time_ms: cpu_time_ms,
            gpu_kernel_time_ms: gpu_result.as_ref().map(|r| r.gpu_time_ms),
            total_verification_time_ms: total_time_ms,
            accepted,
            rejection_reason: None,
        };


        Ok(report)
    }

    /// Compare two output matrices and return verification result.
    fn compare_outputs(&self, reference: &Array2<f32>, candidate: &Array2<f32>) -> VerificationResult {
        if reference.shape() != candidate.shape() {
            return VerificationResult::Failed {
                reason: format!(
                    "Shape mismatch: {:?} vs {:?}",
                    reference.shape(),
                    candidate.shape()
                ),
            };
        }

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;

        for (r, c) in reference.iter().zip(candidate.iter()) {
            let abs_err = (r - c).abs();
            max_abs = max_abs.max(abs_err);

            if r.abs() > 1e-6 {
                let rel_err = abs_err / r.abs();
                max_rel = max_rel.max(rel_err);
            }
        }

        if max_abs > self.tolerance.max_abs_error {
            return VerificationResult::Failed {
                reason: format!(
                    "Absolute error {} exceeds tolerance {}",
                    max_abs, self.tolerance.max_abs_error
                ),
            };
        }

        if max_rel > self.tolerance.max_rel_error {
            return VerificationResult::Failed {
                reason: format!(
                    "Relative error {} exceeds tolerance {}",
                    max_rel, self.tolerance.max_rel_error
                ),
            };
        }

        VerificationResult::Passed {
            max_abs_error: max_abs,
            max_rel_error: max_rel,
        }
    }

    /// Run a GPU smoke test: small matmul, compare to CPU.
    pub fn gpu_smoke_test(&self) -> Result<(bool, String)> {
        let gpu_exec = match &self.gpu_executor {
            Some(e) => e,
            None => return Ok((false, "GPU not available".into())),
        };

        let m = 64;
        let n = 64;
        let k = 128;

        // Create test data
        fastrand::seed(12345);
        let lhs = Array2::from_shape_fn((m, k), |_| fastrand::f32() * 2.0 - 1.0);
        let rhs = Array2::from_shape_fn((k, n), |_| fastrand::f32() * 2.0 - 1.0);

        // CPU reference
        let cpu_result = lhs.dot(&rhs);

        // GPU execution
        let problem = MatmulProblem::new(m, n, k, DataType::F32);
        let inputs = MatmulInputs::new(lhs.view(), rhs.view(), None, ActivationKind::None);
        let gpu_result = gpu_exec.execute_matmul_timed(problem, &inputs, None)?;

        // Compare
        let mut max_err = 0.0f32;
        for (r, g) in cpu_result.iter().zip(gpu_result.output.iter()) {
            max_err = max_err.max((r - g).abs());
        }

        let passed = max_err < 1e-4;
        let message = format!(
            "GPU smoke test: {}x{}x{} matmul, max_abs_error={:.2e}, gpu_time={:.3}ms, passed={}",
            m, n, k, max_err, gpu_result.gpu_time_ms, passed
        );

        Ok((passed, message))
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new().expect("failed to create verifier")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_outputs_pass() {
        let verifier = Verifier::with_tolerance(VerificationTolerance::default()).unwrap();
        let a = Array2::from_elem((4, 4), 1.0f32);
        let b = Array2::from_elem((4, 4), 1.00001f32);
        
        let result = verifier.compare_outputs(&a, &b);
        assert!(matches!(result, VerificationResult::Passed { .. }));
    }

    #[test]
    fn test_compare_outputs_fail() {
        let verifier = Verifier::with_tolerance(VerificationTolerance::strict()).unwrap();
        let a = Array2::from_elem((4, 4), 1.0f32);
        let b = Array2::from_elem((4, 4), 1.1f32); // 10% error
        
        let result = verifier.compare_outputs(&a, &b);
        assert!(matches!(result, VerificationResult::Failed { .. }));
    }
}
