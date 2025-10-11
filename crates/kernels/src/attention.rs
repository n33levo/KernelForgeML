//! Attention mechanism kernels.

use crate::utils::softmax_inplace;
use anyhow::{ensure, Result};
use ndarray::{Array2, ArrayView2};

pub fn scaled_dot_product_attention(
    query: ArrayView2<'_, f32>,
    key: ArrayView2<'_, f32>,
    value: ArrayView2<'_, f32>,
    mask: Option<ArrayView2<'_, f32>>,
    scale: f32,
) -> Result<Array2<f32>> {
    ensure!(
        query.ncols() == key.ncols(),
        "query dim {} must match key dim {}",
        query.ncols(),
        key.ncols()
    );
    let mut scores = query.dot(&key.t());
    scores *= scale;

    if let Some(mask) = mask {
        ensure!(
            mask.dim() == scores.dim(),
            "mask shape {:?} incompatible with attention scores {:?}",
            mask.dim(),
            scores.dim()
        );
        scores += &mask;
    }

    softmax_inplace(scores.view_mut());
    ensure!(
        scores.ncols() == value.nrows(),
        "scores column count {} differs from value rows {}",
        scores.ncols(),
        value.nrows()
    );
    let output = scores.dot(&value);
    Ok(output)
}
