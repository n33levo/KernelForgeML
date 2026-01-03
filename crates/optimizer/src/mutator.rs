//! Mutation testing for the optimizer.
//!
//! Generates "broken" optimization plans and workload variants to verify
//! the test suite catches bugs.
//!
//! There are two types of mutants:
//! 1. **Plan mutants** - Invalid optimization plans that should fail validation
//! 2. **Plausible mutants** - Valid-looking plans that produce wrong numeric results

use crate::knobs::OptimizationKnobs;
use crate::plan::OptimizationPlan;
use crate::workload::WorkloadSignature;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A mutant is a deliberately broken optimization plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mutant {
    /// Description of what was mutated.
    pub description: String,
    /// The mutated plan.
    pub plan: OptimizationPlan,
    /// Expected failure reason (for documentation).
    pub expected_failure: String,
    /// How this mutant is expected to be killed.
    #[serde(default)]
    pub killed_by: KillReason,
}

/// How a mutant is expected to be killed.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum KillReason {
    /// Killed by plan validation (shape/type checks)
    #[default]
    Validation,
    /// Killed by numeric equivalence checks (CPU vs GPU mismatch)
    NumericMismatch,
    /// Killed by a specific regression test case
    RegressionCase,
}

/// Mutator that generates broken variants.
pub struct Mutator {
    #[allow(dead_code)]
    seed: u64,
}

impl Mutator {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate mutants from a valid plan.
    /// 
    /// Returns two categories:
    /// - Mutants that fail validation (caught by plan.validate())
    /// - Mutants that pass validation but produce wrong numeric results
    pub fn generate_mutants(&self, valid_plan: &OptimizationPlan) -> Vec<Mutant> {
        let mut mutants = Vec::new();

        // =========================================================
        // Category 1: Invalid plans that should fail validation
        // =========================================================

        // Mutant 1: Wrong tile_k (0 or very large)
        mutants.push(Mutant {
            description: "tile_k = 0".into(),
            plan: OptimizationPlan {
                knobs: OptimizationKnobs {
                    tile_k: 0,
                    ..valid_plan.knobs.clone()
                },
                ..valid_plan.clone()
            },
            expected_failure: "Tile dimensions must be > 0".into(),
            killed_by: KillReason::Validation,
        });

        // Mutant 2: Non-power-of-2 vector width
        mutants.push(Mutant {
            description: "vector_width = 3 (not power of 2)".into(),
            plan: OptimizationPlan {
                knobs: OptimizationKnobs {
                    vector_width: 3,
                    ..valid_plan.knobs.clone()
                },
                ..valid_plan.clone()
            },
            expected_failure: "Vector width must be power of 2".into(),
            killed_by: KillReason::Validation,
        });

        // Mutant 3: Invalid epsilon
        mutants.push(Mutant {
            description: "layernorm_epsilon = 0.0".into(),
            plan: OptimizationPlan {
                knobs: OptimizationKnobs {
                    layernorm_epsilon: 0.0,
                    ..valid_plan.knobs.clone()
                },
                ..valid_plan.clone()
            },
            expected_failure: "LayerNorm epsilon must be > 0".into(),
            killed_by: KillReason::Validation,
        });

        // Mutant 4: Invalid pass name
        let mut bad_passes = valid_plan.pass_order.clone();
        bad_passes.push("nonexistent-pass".into());
        mutants.push(Mutant {
            description: "Unknown pass in pass_order".into(),
            plan: OptimizationPlan {
                pass_order: bad_passes,
                ..valid_plan.clone()
            },
            expected_failure: "Unknown pass: nonexistent-pass".into(),
            killed_by: KillReason::Validation,
        });

        // Mutant 5: Invalid target
        mutants.push(Mutant {
            description: "target = 'tpu' (unsupported)".into(),
            plan: OptimizationPlan {
                target: "tpu".into(),
                ..valid_plan.clone()
            },
            expected_failure: "Invalid target: tpu".into(),
            killed_by: KillReason::Validation,
        });

        // Mutant 6: Huge tile size
        mutants.push(Mutant {
            description: "tile_m = 1024 (too large)".into(),
            plan: OptimizationPlan {
                knobs: OptimizationKnobs {
                    tile_m: 1024,
                    ..valid_plan.knobs.clone()
                },
                ..valid_plan.clone()
            },
            expected_failure: "Tile dimensions too large".into(),
            killed_by: KillReason::Validation,
        });

        mutants
    }
    
    /// Generate "plausible" mutants that pass validation but produce wrong results.
    /// These test numeric equivalence checking.
    pub fn generate_plausible_mutants(&self) -> Vec<PlausibleMutant> {
        vec![
            PlausibleMutant {
                description: "uniform attention weights (ignores content)".into(),
                mutant_type: PlausibleMutantType::BrokenSoftmax,
                expected_max_error: 0.1, // Should have significant error
            },
            PlausibleMutant {
                description: "missing attention scale (1/sqrt(d_k))".into(),
                mutant_type: PlausibleMutantType::MissingAttentionScale,
                expected_max_error: 0.01, // Should cause drift
            },
            PlausibleMutant {
                description: "biased layernorm variance (no gamma/beta)".into(),
                mutant_type: PlausibleMutantType::BadLayerNormVariance,
                expected_max_error: 0.01,
            },
        ]
    }
}

/// A plausible mutant that passes validation but produces wrong numeric results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlausibleMutant {
    pub description: String,
    pub mutant_type: PlausibleMutantType,
    /// Expected minimum error when comparing to reference
    pub expected_max_error: f32,
}

/// Types of plausible mutations (at the kernel level, not plan level).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlausibleMutantType {
    /// Uniform attention weights (ignores Q·K content)
    BrokenSoftmax,
    /// Missing 1/sqrt(d_k) attention scaling
    MissingAttentionScale,
    /// Uses incorrect variance computation in layernorm
    BadLayerNormVariance,
}

/// Regression corpus - test cases that have caught mutants.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegressionCorpus {
    pub cases: Vec<RegressionCase>,
}

/// A single regression test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionCase {
    /// Workload signature for reproducibility.
    pub signature: WorkloadSignature,
    /// Random seed used to generate test data.
    pub seed: u64,
    /// Description of what this case tests.
    pub description: String,
    /// Mutant description this case was created to catch.
    pub catches_mutant: Option<String>,
}

impl RegressionCorpus {
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

    /// Add a case if not already present.
    pub fn add_case(&mut self, case: RegressionCase) -> bool {
        // Check for duplicate
        let exists = self.cases.iter().any(|c| {
            c.signature == case.signature && c.seed == case.seed
        });
        if exists {
            return false;
        }
        self.cases.push(case);
        true
    }

    /// Create default corpus with standard test cases.
    pub fn with_defaults() -> Self {
        Self {
            cases: vec![
                RegressionCase {
                    signature: WorkloadSignature::microblock(32, 32, 64),
                    seed: 42,
                    description: "Small microblock".into(),
                    catches_mutant: None,
                },
                RegressionCase {
                    signature: WorkloadSignature::microblock(64, 128, 256),
                    seed: 12345,
                    description: "Medium microblock (typical attention)".into(),
                    catches_mutant: None,
                },
                RegressionCase {
                    signature: WorkloadSignature::microblock(128, 128, 512),
                    seed: 98765,
                    description: "Larger microblock (MLP-like)".into(),
                    catches_mutant: None,
                },
            ],
        }
    }
}

/// Result of mutation testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationTestResult {
    pub total_mutants: usize,
    pub killed_mutants: usize,
    pub escaped_mutants: Vec<String>,
    pub new_cases_added: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutator_generates_mutants() {
        let plan = OptimizationPlan::default_cpu();
        let mutator = Mutator::new(42);
        let mutants = mutator.generate_mutants(&plan);
        
        assert!(!mutants.is_empty());
        
        // All mutants should fail validation
        for mutant in &mutants {
            assert!(mutant.plan.validate().is_err(), 
                "Mutant '{}' should fail validation", mutant.description);
        }
    }

    #[test]
    fn test_regression_corpus() {
        let corpus = RegressionCorpus::with_defaults();
        assert!(!corpus.cases.is_empty());
    }
}
