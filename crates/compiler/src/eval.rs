//! Evaluation helpers for end-to-end matmul benchmarking.
//!
//! The evaluation suite exercises a handful of representative matmul
//! configurations, checks numerical fidelity against a reference
//! implementation, and collects latency / throughput metrics in a
//! reproducible JSON report.

use crate::session::{CompilerSession, MatmulResult};
use anyhow::Result;
use kernelforge_ir::lowering::LoweringTarget;
use kernelforge_kernels::config::{ActivationKind, DataType, MatmulProblem};
use kernelforge_kernels::matmul::MatmulInputs;
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulEvalCase {
    pub name: String,
    pub problem: MatmulProblem,
}

impl MatmulEvalCase {
    pub fn new(name: impl Into<String>, problem: MatmulProblem) -> Self {
        Self {
            name: name.into(),
            problem,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatmulEvalResult {
    pub case: String,
    pub kernel: String,
    pub latency_ms: f64,
    pub gflops: f64,
    pub bandwidth_gbps: f64,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub l2_error: f64,
    pub problem: MatmulProblem,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub target: String,
    pub generated_at_unix_ms: u128,
    pub cases: Vec<MatmulEvalResult>,
}

impl EvaluationReport {
    pub fn as_map(&self) -> BTreeMap<&str, &MatmulEvalResult> {
        self.cases
            .iter()
            .map(|case| (case.case.as_str(), case))
            .collect()
    }

    pub fn diff<'a>(
        &'a self,
        baseline: &'a EvaluationReport,
    ) -> BTreeMap<&'a str, EvaluationDelta<'a>> {
        let mut deltas = BTreeMap::new();
        let current = self.as_map();
        let previous = baseline.as_map();

        for (case, result) in current {
            if let Some(&baseline_result) = previous.get(case) {
                deltas.insert(
                    case,
                    EvaluationDelta {
                        current: result,
                        baseline: baseline_result,
                        latency_ms_delta: result.latency_ms - baseline_result.latency_ms,
                        gflops_delta: result.gflops - baseline_result.gflops,
                    },
                );
            }
        }

        deltas
    }
}

#[derive(Debug)]
pub struct EvaluationDelta<'a> {
    pub current: &'a MatmulEvalResult,
    pub baseline: &'a MatmulEvalResult,
    pub latency_ms_delta: f64,
    pub gflops_delta: f64,
}

pub struct EvaluationSuite {
    cases: Vec<MatmulEvalCase>,
}

impl EvaluationSuite {
    pub fn new(cases: Vec<MatmulEvalCase>) -> Self {
        Self { cases }
    }

    pub fn transformer_matmul_smoke() -> Self {
        let cases = vec![
            MatmulEvalCase::new(
                "decoder_mha_qkv",
                MatmulProblem::new(64, 64, 128, DataType::F32),
            ),
            MatmulEvalCase::new(
                "decoder_mlp_up_proj",
                MatmulProblem::new(128, 192, 192, DataType::F32),
            ),
            MatmulEvalCase::new(
                "decoder_mlp_down_proj",
                MatmulProblem::new(128, 128, 256, DataType::F32),
            ),
        ];
        Self::new(cases)
    }

    pub fn run(&self, session: &mut CompilerSession) -> Result<EvaluationReport> {
        let mut results = Vec::with_capacity(self.cases.len());

        for case in &self.cases {
            let lhs = deterministic_tensor(case.problem.m, case.problem.k);
            let rhs = deterministic_tensor(case.problem.k, case.problem.n);
            let expected = lhs.dot(&rhs);

            let inputs = MatmulInputs::new(lhs.view(), rhs.view(), None, ActivationKind::None);

            let (lap_time, MatmulResult { output, profile }) =
                timed(|| session.execute_matmul(case.problem, &inputs))?;

            let profile_kernel = profile
                .as_ref()
                .map(|p| p.kernel.clone())
                .unwrap_or_else(|| "unknown".to_string());

            let profile_latency = profile
                .as_ref()
                .map(|p| p.average_time_ms)
                .filter(|lat| *lat > 0.0)
                .unwrap_or(lap_time.as_secs_f64() * 1000.0);

            let gflops = if profile_latency > 0.0 {
                case.problem.flops() / (profile_latency * 1.0e6)
            } else {
                0.0
            };

            let bandwidth_gbps = if profile_latency > 0.0 {
                let bytes = case.problem.data_footprint_bytes() as f64;
                (bytes / 1.0e9) / (profile_latency / 1000.0)
            } else {
                0.0
            };

            let diff = &output - &expected;
            let max_abs_error = diff
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs() as f64));
            let mean_abs_error =
                diff.iter().map(|value| value.abs() as f64).sum::<f64>() / diff.len() as f64;
            let l2_error = diff
                .iter()
                .map(|value| {
                    let v = *value as f64;
                    v * v
                })
                .sum::<f64>()
                .sqrt();

            results.push(MatmulEvalResult {
                case: case.name.clone(),
                kernel: profile_kernel,
                latency_ms: profile_latency,
                gflops,
                bandwidth_gbps,
                max_abs_error,
                mean_abs_error,
                l2_error,
                problem: case.problem,
            });
        }

        let generated_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_millis();

        let target = match session.config().target {
            LoweringTarget::Cpu => "cpu",
            LoweringTarget::Gpu => "gpu",
        }
        .to_string();

        Ok(EvaluationReport {
            target,
            generated_at_unix_ms,
            cases: results,
        })
    }
}

fn deterministic_tensor(rows: usize, cols: usize) -> Array2<f32> {
    let mut tensor = Array2::zeros((rows, cols));
    tensor
        .axis_iter_mut(Axis(0))
        .enumerate()
        .for_each(|(row_idx, mut row)| {
            row.iter_mut().enumerate().for_each(|(col_idx, value)| {
                let seed = ((row_idx * 1313) ^ (col_idx * 7331)) as f32;
                *value = 1.0 + (seed % 17.0) / 16.0;
            });
        });
    tensor
}

fn timed<F, T>(f: F) -> Result<(Duration, T)>
where
    F: FnOnce() -> Result<T>,
{
    let start = Instant::now();
    let value = f()?;
    Ok((start.elapsed(), value))
}
