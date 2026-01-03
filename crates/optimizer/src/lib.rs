//! LLM-Guided Kernel Optimizer with Verification
//!
//! This crate provides the "Assured LLM-Guided Kernel/Pass Optimizer" for KernelForgeML.
//! The core idea: LLM proposes optimization plans → verifier checks correctness → system
//! learns and records best plans.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
//! │  Optimizer      │────▶│  Verifier        │────▶│  Results Cache  │
//! │  (LLM/Heuristic)│     │  (CPU vs GPU)    │     │  (best_plans)   │
//! └─────────────────┘     └──────────────────┘     └─────────────────┘
//! ```
//!
//! # Key Components
//!
//! - [`knobs::OptimizationKnobs`]: Tunable parameters for optimization passes
//! - [`plan::OptimizationPlan`]: Complete optimization strategy (pass order + knobs)
//! - [`optimizer::Optimizer`]: Trait for proposing optimization plans
//! - [`verifier::Verifier`]: Ensures plans are correct via numeric comparison
//! - [`mutator::Mutator`]: Generates broken variants for mutation testing

pub mod knobs;
pub mod mutator;
pub mod optimizer;
pub mod plan;
pub mod report;
pub mod verifier;
pub mod workload;

pub use knobs::OptimizationKnobs;
pub use optimizer::{HeuristicOptimizer, LlmOptimizer, Optimizer};
pub use plan::OptimizationPlan;
pub use report::{AuditReport, VerificationResult};
pub use verifier::Verifier;
pub use workload::{Microblock, WorkloadSignature};
