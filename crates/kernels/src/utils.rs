//! Shared helpers for kernel implementations.

use crate::config::ActivationKind;
use anyhow::{bail, Result};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis};
use rayon::prelude::*;

pub fn validate_matmul_inputs(lhs: &ArrayView2<'_, f32>, rhs: &ArrayView2<'_, f32>) -> Result<()> {
    if lhs.ncols() != rhs.nrows() {
        bail!(
            "matmul dimension mismatch: lhs {}x{} vs rhs {}x{}",
            lhs.nrows(),
            lhs.ncols(),
            rhs.nrows(),
            rhs.ncols()
        );
    }
    Ok(())
}

pub fn apply_bias_activation(
    mut output: Array2<f32>,
    bias: Option<&ArrayView2<'_, f32>>,
    activation: ActivationKind,
) -> Result<Array2<f32>> {
    if let Some(bias) = bias {
        if bias.shape() != output.shape() {
            bail!(
                "bias shape {:?} incompatible with output {:?}",
                bias.shape(),
                output.shape()
            );
        }
        output += bias;
    }

    match activation {
        ActivationKind::None => Ok(output),
        ActivationKind::Relu => {
            output.mapv_inplace(|x| x.max(0.0));
            Ok(output)
        }
        ActivationKind::Gelu => {
            // Approximate GELU (tanh formulation).
            output.mapv_inplace(|x| {
                let c = (2.0 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
            });
            Ok(output)
        }
    }
}

pub fn softmax_inplace(mut scores: ArrayViewMut2<'_, f32>) {
    scores
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            row -= max;
            row.mapv_inplace(|x| x.exp());
            let sum = row.sum();
            row /= sum.max(f32::EPSILON);
        });
}

pub fn normalize_rows_inplace(mut data: ArrayViewMut2<'_, f32>, eps: f32) {
    data.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let mean = row.mean().unwrap_or(0.0);
            let var = row.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
            let inv_std = 1.0 / (var + eps).sqrt();
            row.mapv_inplace(|x| (x - mean) * inv_std);
        });
}
