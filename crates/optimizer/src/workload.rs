//! Workload definitions for the optimizer.

use serde::{Deserialize, Serialize};

/// Signature that uniquely identifies a workload for caching.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadSignature {
    /// Sequence length (number of tokens/rows)
    pub seq_len: usize,
    /// Model dimension (hidden size)
    pub d_model: usize,
    /// Head dimension (d_model / num_heads, but we use single head for simplicity)
    pub d_k: usize,
    pub dtype: String,
    pub has_attention: bool,
    pub has_layernorm: bool,
}

impl WorkloadSignature {
    pub fn matmul(m: usize, n: usize, k: usize) -> Self {
        Self {
            seq_len: m,
            d_model: n,
            d_k: k,
            dtype: "f32".into(),
            has_attention: false,
            has_layernorm: false,
        }
    }

    /// Microblock with just matmul + layernorm (legacy)
    pub fn microblock(m: usize, n: usize, k: usize) -> Self {
        Self {
            seq_len: m,
            d_model: n,
            d_k: k,
            dtype: "f32".into(),
            has_attention: false,
            has_layernorm: true,
        }
    }

    /// Transformer block: Matmul → Attention → LayerNorm
    pub fn transformer_block(seq_len: usize, d_model: usize) -> Self {
        Self {
            seq_len,
            d_model,
            d_k: d_model, // Single head, d_k = d_model
            dtype: "f32".into(),
            has_attention: true,
            has_layernorm: true,
        }
    }

    /// Convert to cache key string.
    pub fn cache_key(&self) -> String {
        format!(
            "seq{}_d{}_dk{}_{}_{}",
            self.seq_len,
            self.d_model,
            self.d_k,
            if self.has_attention { "attn" } else { "noattn" },
            if self.has_layernorm { "ln" } else { "noln" }
        )
    }
    
    /// Legacy compatibility: return m, n, k dimensions
    pub fn mnk(&self) -> (usize, usize, usize) {
        (self.seq_len, self.d_model, self.d_k)
    }
}

/// A "Transformer Microblock" - single transformer block workload.
///
/// Full workload:
/// - Q/K/V projections: Input (seq×d_model) @ W_q/k/v (d_model×d_k) → Q/K/V (seq×d_k)
/// - Attention: Q @ K^T → scale → softmax → @ V → (seq×d_k)
/// - LayerNorm on output
///
/// This is the minimal "transformer block" math without building a full LLM.
#[derive(Debug, Clone)]
pub struct TransformerMicroblock {
    pub signature: WorkloadSignature,
    /// Input embeddings (seq_len × d_model)
    pub input: ndarray::Array2<f32>,
    /// Q projection weight (d_model × d_k)
    pub w_q: ndarray::Array2<f32>,
    /// K projection weight (d_model × d_k)
    pub w_k: ndarray::Array2<f32>,
    /// V projection weight (d_model × d_k)
    pub w_v: ndarray::Array2<f32>,
    /// LayerNorm gamma (1 × d_k)
    pub ln_gamma: ndarray::Array2<f32>,
    /// LayerNorm beta (1 × d_k)
    pub ln_beta: ndarray::Array2<f32>,
}

impl TransformerMicroblock {
    /// Create a transformer microblock with random data for testing.
    pub fn random(seq_len: usize, d_model: usize, seed: u64) -> Self {
        use ndarray::Array2;
        
        fastrand::seed(seed);
        
        // Xavier-like initialization for stability
        let scale = (2.0 / (d_model as f32)).sqrt();
        
        let input = Array2::from_shape_fn((seq_len, d_model), |_| 
            (fastrand::f32() * 2.0 - 1.0) * scale);
        let w_q = Array2::from_shape_fn((d_model, d_model), |_| 
            (fastrand::f32() * 2.0 - 1.0) * scale);
        let w_k = Array2::from_shape_fn((d_model, d_model), |_| 
            (fastrand::f32() * 2.0 - 1.0) * scale);
        let w_v = Array2::from_shape_fn((d_model, d_model), |_| 
            (fastrand::f32() * 2.0 - 1.0) * scale);
        
        let ln_gamma = Array2::ones((1, d_model));
        let ln_beta = Array2::zeros((1, d_model));

        Self {
            signature: WorkloadSignature::transformer_block(seq_len, d_model),
            input,
            w_q,
            w_k,
            w_v,
            ln_gamma,
            ln_beta,
        }
    }

    /// Compute reference output using CPU kernels.
    /// 
    /// Steps:
    /// 1. Q = input @ W_q
    /// 2. K = input @ W_k
    /// 3. V = input @ W_v
    /// 4. scores = Q @ K^T * (1/sqrt(d_k))
    /// 5. attn_weights = softmax(scores)
    /// 6. attn_output = attn_weights @ V
    /// 7. output = LayerNorm(attn_output)
    pub fn compute_reference(&self) -> TransformerOutput {
        use kernelforge_kernels::attention::scaled_dot_product_attention;
        use kernelforge_kernels::layernorm::layer_norm;
        use kernelforge_kernels::matmul::{MatmulInputs, MatmulKernel, ReferenceMatmul};
        use kernelforge_kernels::config::{ActivationKind, DataType, MatmulProblem};

        let seq_len = self.signature.seq_len;
        let d_model = self.signature.d_model;

        // Step 1-3: Q/K/V projections
        let matmul = ReferenceMatmul::new();
        let proj_problem = MatmulProblem::new(seq_len, d_model, d_model, DataType::F32);
        
        let q = matmul.run(&proj_problem, &MatmulInputs::new(
            self.input.view(), self.w_q.view(), None, ActivationKind::None
        )).expect("Q projection failed");
        
        let k = matmul.run(&proj_problem, &MatmulInputs::new(
            self.input.view(), self.w_k.view(), None, ActivationKind::None
        )).expect("K projection failed");
        
        let v = matmul.run(&proj_problem, &MatmulInputs::new(
            self.input.view(), self.w_v.view(), None, ActivationKind::None
        )).expect("V projection failed");

        // Step 4-6: Attention
        let scale = 1.0 / (d_model as f32).sqrt();
        let attn_output = scaled_dot_product_attention(
            q.view(), k.view(), v.view(), None, scale
        ).expect("attention failed");

        // Step 7: LayerNorm
        let final_output = layer_norm(
            attn_output.view(),
            self.ln_gamma.view(),
            self.ln_beta.view(),
            1e-5,
        ).expect("layer norm failed");

        TransformerOutput {
            q,
            k,
            v,
            attn_output,
            final_output,
        }
    }
    
    /// Compute with a "broken" softmax (no max subtraction) for mutation testing.
    /// Actually, for more visible effect, we'll skip the scaling AND use raw dot product
    /// without proper normalization, which creates peaky distributions.
    pub fn compute_broken_softmax(&self) -> TransformerOutput {
        use kernelforge_kernels::layernorm::layer_norm;
        use kernelforge_kernels::matmul::{MatmulInputs, MatmulKernel, ReferenceMatmul};
        use kernelforge_kernels::config::{ActivationKind, DataType, MatmulProblem};

        let seq_len = self.signature.seq_len;
        let d_model = self.signature.d_model;

        let matmul = ReferenceMatmul::new();
        let proj_problem = MatmulProblem::new(seq_len, d_model, d_model, DataType::F32);
        
        let q = matmul.run(&proj_problem, &MatmulInputs::new(
            self.input.view(), self.w_q.view(), None, ActivationKind::None
        )).expect("Q projection failed");
        
        let k = matmul.run(&proj_problem, &MatmulInputs::new(
            self.input.view(), self.w_k.view(), None, ActivationKind::None
        )).expect("K projection failed");
        
        let v = matmul.run(&proj_problem, &MatmulInputs::new(
            self.input.view(), self.w_v.view(), None, ActivationKind::None
        )).expect("V projection failed");

        // BROKEN: Use uniform attention weights instead of proper softmax
        // This produces completely wrong results
        let mut scores = q.dot(&k.t());
        let n = scores.ncols() as f32;
        scores.fill(1.0 / n); // Uniform attention (wrong!)
        
        let attn_output = scores.dot(&v);

        let final_output = layer_norm(
            attn_output.view(),
            self.ln_gamma.view(),
            self.ln_beta.view(),
            1e-5,
        ).expect("layer norm failed");

        TransformerOutput { q, k, v, attn_output, final_output }
    }
    
    /// Compute with missing attention scale (1/sqrt(d_k)).
    /// This changes the magnitude of attention weights.
    pub fn compute_no_scale(&self) -> TransformerOutput {
        use kernelforge_kernels::layernorm::layer_norm;
        use kernelforge_kernels::matmul::{MatmulInputs, MatmulKernel, ReferenceMatmul};
        use kernelforge_kernels::config::{ActivationKind, DataType, MatmulProblem};
        use kernelforge_kernels::utils::softmax_inplace;

        let seq_len = self.signature.seq_len;
        let d_model = self.signature.d_model;

        let matmul = ReferenceMatmul::new();
        let proj_problem = MatmulProblem::new(seq_len, d_model, d_model, DataType::F32);
        
        let q = matmul.run(&proj_problem, &MatmulInputs::new(
            self.input.view(), self.w_q.view(), None, ActivationKind::None
        )).expect("Q projection failed");
        
        let k = matmul.run(&proj_problem, &MatmulInputs::new(
            self.input.view(), self.w_k.view(), None, ActivationKind::None
        )).expect("K projection failed");
        
        let v = matmul.run(&proj_problem, &MatmulInputs::new(
            self.input.view(), self.w_v.view(), None, ActivationKind::None
        )).expect("V projection failed");

        // Missing scale factor!
        let mut scores = q.dot(&k.t());
        // NO: scores *= 1.0 / (d_model as f32).sqrt();
        
        softmax_inplace(scores.view_mut());
        let attn_output = scores.dot(&v);

        let final_output = layer_norm(
            attn_output.view(),
            self.ln_gamma.view(),
            self.ln_beta.view(),
            1e-5,
        ).expect("layer norm failed");

        TransformerOutput { q, k, v, attn_output, final_output }
    }

    /// Compute with incorrect layer norm variance (missing mean subtraction).
    pub fn compute_bad_layernorm(&self) -> TransformerOutput {
        use kernelforge_kernels::matmul::{MatmulInputs, MatmulKernel, ReferenceMatmul};
        use kernelforge_kernels::config::{ActivationKind, DataType, MatmulProblem};
        use kernelforge_kernels::utils::normalize_rows_inplace;

        let seq_len = self.signature.seq_len;
        let d_model = self.signature.d_model;

        let matmul = ReferenceMatmul::new();
        let proj_problem = MatmulProblem::new(seq_len, d_model, d_model, DataType::F32);

        let q = matmul.run(&proj_problem, &MatmulInputs::new(
            self.input.view(), self.w_q.view(), None, ActivationKind::None
        )).expect("Q projection failed");

        let k = matmul.run(&proj_problem, &MatmulInputs::new(
            self.input.view(), self.w_k.view(), None, ActivationKind::None
        )).expect("K projection failed");

        let v = matmul.run(&proj_problem, &MatmulInputs::new(
            self.input.view(), self.w_v.view(), None, ActivationKind::None
        )).expect("V projection failed");

        // Correct attention
        let scale = 1.0 / (d_model as f32).sqrt();
        let attn_output = kernelforge_kernels::attention::scaled_dot_product_attention(
            q.view(), k.view(), v.view(), None, scale
        ).expect("attention failed");

        // BROKEN layer norm: normalize rows without applying gamma/beta and with biased variance
        let mut final_output = attn_output.to_owned();
        normalize_rows_inplace(final_output.view_mut(), 1e-5);
        final_output.mapv_inplace(|x| x * 0.95); // drift a bit to make error visible

        TransformerOutput { q, k, v, attn_output, final_output }
    }
}

/// Output from running a transformer microblock.
#[derive(Debug, Clone)]
pub struct TransformerOutput {
    /// Q projection output
    pub q: ndarray::Array2<f32>,
    /// K projection output
    pub k: ndarray::Array2<f32>,
    /// V projection output
    pub v: ndarray::Array2<f32>,
    /// Attention output (before LayerNorm)
    pub attn_output: ndarray::Array2<f32>,
    /// Final output after LayerNorm
    pub final_output: ndarray::Array2<f32>,
}

// ============================================================================
// Legacy Microblock for backward compatibility
// ============================================================================

/// Legacy "Microblock" - just Matmul + LayerNorm (no attention).
#[derive(Debug, Clone)]
pub struct Microblock {
    pub signature: WorkloadSignature,
    /// Input matrix A (M×K)
    pub input: ndarray::Array2<f32>,
    /// Weight matrix B (K×N)
    pub weight: ndarray::Array2<f32>,
    /// LayerNorm gamma (1×N)
    pub ln_gamma: ndarray::Array2<f32>,
    /// LayerNorm beta (1×N)
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

        let (m, n, k) = self.signature.mnk();

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

/// Output from running a legacy microblock.
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
        assert!(sig.cache_key().contains("seq64"));
    }
    
    #[test]
    fn test_transformer_microblock_reference() {
        let block = TransformerMicroblock::random(32, 64, 42);
        let output = block.compute_reference();
        
        assert_eq!(output.q.shape(), &[32, 64]);
        assert_eq!(output.k.shape(), &[32, 64]);
        assert_eq!(output.v.shape(), &[32, 64]);
        assert_eq!(output.attn_output.shape(), &[32, 64]);
        assert_eq!(output.final_output.shape(), &[32, 64]);
    }
    
    #[test]
    fn test_broken_softmax_differs() {
        let block = TransformerMicroblock::random(16, 32, 42);
        let correct = block.compute_reference();
        let broken = block.compute_broken_softmax();
        
        // The broken softmax should produce different results
        let diff = (&correct.final_output - &broken.final_output)
            .mapv(|x| x.abs())
            .sum();
        
        println!("Broken softmax total diff: {}", diff);
    }
    
    #[test]
    fn test_no_scale_differs() {
        let block = TransformerMicroblock::random(16, 32, 42);
        let correct = block.compute_reference();
        let no_scale = block.compute_no_scale();
        
        // Missing scale should produce different results
        let max_diff = (&correct.final_output - &no_scale.final_output)
            .mapv(|x| x.abs())
            .fold(0.0f32, |a, &b| a.max(b));
        
        println!("No-scale max diff: {}", max_diff);
        // Should be noticeably different
        assert!(max_diff > 1e-6, "no-scale mutation should cause difference");
    }
}
