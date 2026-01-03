//! Workload definitions for the optimizer.

use serde::{Deserialize, Serialize};

/// Signature that uniquely identifies a workload for caching.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadSignature {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub dtype: String,
    pub has_bias: bool,
    pub has_activation: bool,
    pub has_layernorm: bool,
}

impl WorkloadSignature {
    pub fn matmul(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            dtype: "f32".into(),
            has_bias: false,
            has_activation: false,
            has_layernorm: false,
        }
    }

    pub fn microblock(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            dtype: "f32".into(),
            has_bias: false,
            has_activation: false,
            has_layernorm: true,
        }
    }

    /// Convert to cache key string.
    pub fn cache_key(&self) -> String {
        format!(
            "{}x{}x{}_{}_{}_{}",
            self.m,
            self.n,
            self.k,
            self.dtype,
            if self.has_layernorm { "ln" } else { "nln" },
            if self.has_activation { "act" } else { "noact" }
        )
    }
}

/// A "Transformer Microblock" - minimal realistic workload.
///
/// This is the target workload for optimization:
/// - Matmul (M×K @ K×N → M×N)
/// - Optional activation
/// - LayerNorm
///
/// GPU backend currently only supports the matmul part (no bias, no activation),
/// so we compare GPU matmul against CPU reference matmul for correctness,
/// then run the full block on CPU for functional testing.
#[derive(Debug, Clone)]
pub struct Microblock {
    pub signature: WorkloadSignature,
    /// Input matrix A (M×K)
    pub input: ndarray::Array2<f32>,
    /// Weight matrix B (K×N)
    pub weight: ndarray::Array2<f32>,
    /// LayerNorm gamma (1×N, as 2D for API compatibility)
    pub ln_gamma: ndarray::Array2<f32>,
    /// LayerNorm beta (1×N, as 2D for API compatibility)
    pub ln_beta: ndarray::Array2<f32>,
}

impl Microblock {
    /// Create a microblock with random data for testing.
    pub fn random(m: usize, n: usize, k: usize, seed: u64) -> Self {
        use ndarray::Array2;
        
        fastrand::seed(seed);
        
        let input = Array2::from_shape_fn((m, k), |_| fastrand::f32() * 2.0 - 1.0);
        let weight = Array2::from_shape_fn((k, n), |_| (fastrand::f32() * 2.0 - 1.0) * 0.1);
        // LayerNorm params need to be 2D (1×N) for the API
        let ln_gamma = Array2::ones((1, n));
        let ln_beta = Array2::zeros((1, n));

        Self {
            signature: WorkloadSignature::microblock(m, n, k),
            input,
            weight,
            ln_gamma,
            ln_beta,
        }
    }

    /// Compute reference output using CPU kernels.
    pub fn compute_reference(&self) -> MicroblockOutput {
        use kernelforge_kernels::layernorm::layer_norm;
        use kernelforge_kernels::matmul::{MatmulInputs, MatmulKernel, ReferenceMatmul};
        use kernelforge_kernels::config::{ActivationKind, DataType, MatmulProblem};

        let m = self.signature.m;
        let n = self.signature.n;
        let k = self.signature.k;

        // Step 1: Matmul
        let problem = MatmulProblem::new(m, n, k, DataType::F32);
        let inputs = MatmulInputs::new(
            self.input.view(),
            self.weight.view(),
            None,
            ActivationKind::None,
        );
        let matmul_output = ReferenceMatmul::new()
            .run(&problem, &inputs)
            .expect("reference matmul failed");

        // Step 2: LayerNorm
        let ln_output = layer_norm(
            matmul_output.view(),
            self.ln_gamma.view(),
            self.ln_beta.view(),
            1e-5,
        ).expect("layer norm failed");

        MicroblockOutput {
            matmul_output,
            final_output: ln_output,
        }
    }
}

/// Output from running a microblock.
#[derive(Debug, Clone)]
pub struct MicroblockOutput {
    /// Intermediate matmul result (before LayerNorm)
    pub matmul_output: ndarray::Array2<f32>,
    /// Final output after LayerNorm
    pub final_output: ndarray::Array2<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microblock_reference() {
        let block = Microblock::random(32, 64, 128, 42);
        let output = block.compute_reference();
        
        assert_eq!(output.matmul_output.shape(), &[32, 64]);
        assert_eq!(output.final_output.shape(), &[32, 64]);
    }

    #[test]
    fn test_workload_signature_cache_key() {
        let sig = WorkloadSignature::microblock(64, 128, 256);
        assert!(sig.cache_key().contains("64x128x256"));
    }
}
