//! Layer normalization kernels.

use anyhow::ensure;
use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis};
use rayon::prelude::*;

pub fn layer_norm_inplace(
    mut data: ArrayViewMut2<'_, f32>,
    gamma: Option<ArrayView1<'_, f32>>,
    beta: Option<ArrayView1<'_, f32>>,
    epsilon: f32,
) -> Result<()> {
    let feature_dim = data.len_of(Axis(1));

    let gamma_vec = if let Some(gamma) = gamma {
        ensure!(
            gamma.len() == feature_dim,
            "gamma length {} must equal feature dimension {}",
            gamma.len(),
            feature_dim
        );
        Some(gamma.to_owned())
    } else {
        None
    };

    let beta_vec = if let Some(beta) = beta {
        ensure!(
            beta.len() == feature_dim,
            "beta length {} must equal feature dimension {}",
            beta.len(),
            feature_dim
        );
        Some(beta.to_owned())
    } else {
        None
    };

    let gamma = gamma_vec.as_ref().and_then(Array1::as_slice);
    let beta = beta_vec.as_ref().and_then(Array1::as_slice);

    data.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let len = row.len() as f32;
            let mean = row.iter().sum::<f32>() / len;
            let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / len;
            let inv_std = 1.0 / (var + epsilon).sqrt();

            for (idx, value) in row.iter_mut().enumerate() {
                let mut normalized = (*value - mean) * inv_std;
                if let Some(gamma) = gamma {
                    normalized *= gamma[idx];
                }
                if let Some(beta) = beta {
                    normalized += beta[idx];
                }
                *value = normalized;
            }
        });

    Ok(())
}

/// Non-mutating layer norm that returns a new array.
pub fn layer_norm(
    data: ArrayView2<'_, f32>,
    gamma: ArrayView2<'_, f32>,
    beta: ArrayView2<'_, f32>,
    epsilon: f32,
) -> Result<Array2<f32>> {
    let mut output = data.to_owned();
    let gamma_1d = gamma.row(0);
    let beta_1d = beta.row(0);
    layer_norm_inplace(output.view_mut(), Some(gamma_1d), Some(beta_1d), epsilon)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let mut data = Array2::from_shape_fn((4, 8), |(i, j)| ((i + 1) * (j + 1)) as f32 * 0.125);
        let gamma = Array1::from_elem(8, 1.0f32);
        let beta = Array1::from_elem(8, 0.0f32);

        layer_norm_inplace(data.view_mut(), Some(gamma.view()), Some(beta.view()), 1e-5)
            .expect("layer norm");

        for row in data.axis_iter(Axis(0)) {
            let len = row.len() as f32;
            let mean = row.iter().sum::<f32>() / len;
            let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / len;
            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-4);
            assert_abs_diff_eq!(var, 1.0, epsilon = 5e-4);
        }
    }
}
