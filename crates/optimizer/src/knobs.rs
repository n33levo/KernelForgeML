//! Optimization knobs - tunable parameters for passes.
//!
//! These are the "dials" that the LLM (or heuristic) can turn to control
//! how optimization passes behave.

use serde::{Deserialize, Serialize};

/// Tunable parameters for optimization passes.
///
/// These knobs control tiling, vectorization, and fusion behavior.
/// An LLM or heuristic optimizer proposes values for these knobs,
/// and the verifier checks if the resulting execution is correct.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptimizationKnobs {
    // Matmul tiling parameters
    /// Tile size for M dimension (rows of output)
    pub tile_m: usize,
    /// Tile size for N dimension (columns of output)
    pub tile_n: usize,
    /// Tile size for K dimension (reduction dimension)
    pub tile_k: usize,

    // Vectorization parameters
    /// Vector width for SIMD operations (e.g., 4, 8, 16)
    pub vector_width: usize,

    // Fusion toggles
    /// Enable matmul + activation fusion
    pub enable_fuse_matmul_activation: bool,
    /// Enable MLP block fusion (Linear → Activation → Linear)
    pub enable_fuse_mlp: bool,
    /// Enable constant folding
    pub enable_fold_constants: bool,

    // LayerNorm parameters
    /// Epsilon for numerical stability in layer norm
    pub layernorm_epsilon: f32,
}

impl Default for OptimizationKnobs {
    fn default() -> Self {
        Self {
            tile_m: 64,
            tile_n: 64,
            tile_k: 32,
            vector_width: 8,
            enable_fuse_matmul_activation: true,
            enable_fuse_mlp: true,
            enable_fold_constants: true,
            layernorm_epsilon: 1e-5,
        }
    }
}

impl OptimizationKnobs {
    /// Create knobs optimized for small matrices.
    pub fn for_small() -> Self {
        Self {
            tile_m: 32,
            tile_n: 32,
            tile_k: 16,
            vector_width: 4,
            ..Default::default()
        }
    }

    /// Create knobs optimized for large matrices.
    pub fn for_large() -> Self {
        Self {
            tile_m: 128,
            tile_n: 128,
            tile_k: 64,
            vector_width: 8,
            ..Default::default()
        }
    }

    /// Create knobs for GPU execution (larger tiles, less fusion).
    pub fn for_gpu() -> Self {
        Self {
            tile_m: 16,  // GPU workgroup size
            tile_n: 16,
            tile_k: 16,
            vector_width: 4,
            enable_fuse_matmul_activation: false, // GPU shader doesn't support yet
            enable_fuse_mlp: false,
            enable_fold_constants: true,
            layernorm_epsilon: 1e-5,
        }
    }

    /// Validate that knobs are within reasonable bounds.
    pub fn validate(&self) -> Result<(), String> {
        if self.tile_m == 0 || self.tile_n == 0 || self.tile_k == 0 {
            return Err("Tile dimensions must be > 0".into());
        }
        if self.tile_m > 512 || self.tile_n > 512 || self.tile_k > 512 {
            return Err("Tile dimensions too large (max 512)".into());
        }
        if !self.vector_width.is_power_of_two() || self.vector_width > 16 {
            return Err("Vector width must be power of 2 and <= 16".into());
        }
        if self.layernorm_epsilon <= 0.0 || self.layernorm_epsilon > 1e-3 {
            return Err("LayerNorm epsilon must be in (0, 1e-3]".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_knobs_valid() {
        let knobs = OptimizationKnobs::default();
        assert!(knobs.validate().is_ok());
    }

    #[test]
    fn test_knobs_serialization() {
        let knobs = OptimizationKnobs::default();
        let json = serde_json::to_string(&knobs).unwrap();
        let parsed: OptimizationKnobs = serde_json::from_str(&json).unwrap();
        assert_eq!(knobs, parsed);
    }

    #[test]
    fn test_invalid_knobs() {
        let mut knobs = OptimizationKnobs::default();
        knobs.tile_m = 0;
        assert!(knobs.validate().is_err());

        knobs = OptimizationKnobs::default();
        knobs.vector_width = 3; // Not power of 2
        assert!(knobs.validate().is_err());
    }
}
