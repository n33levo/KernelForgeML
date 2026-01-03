//! Matrix multiplication kernels.

use crate::config::{ActivationKind, MatmulProblem, MatmulTilingConfig};
use crate::utils::{apply_bias_activation, validate_matmul_inputs};
use anyhow::{ensure, Result};
use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3, Axis};
use rayon::prelude::*;
use std::sync::Arc;

pub struct MatmulInputs<'a> {
    pub lhs: ArrayView2<'a, f32>,
    pub rhs: ArrayView2<'a, f32>,
    pub bias: Option<ArrayView2<'a, f32>>,
    pub activation: ActivationKind,
}

impl<'a> MatmulInputs<'a> {
    pub fn new(
        lhs: ArrayView2<'a, f32>,
        rhs: ArrayView2<'a, f32>,
        bias: Option<ArrayView2<'a, f32>>,
        activation: ActivationKind,
    ) -> Self {
        Self {
            lhs,
            rhs,
            bias,
            activation,
        }
    }
}

pub trait MatmulKernel: Send + Sync {
    fn name(&self) -> &'static str;
    fn config(&self) -> MatmulTilingConfig;
    fn supports(&self, problem: &MatmulProblem) -> bool;
    fn run(&self, problem: &MatmulProblem, inputs: &MatmulInputs<'_>) -> Result<Array2<f32>>;
}

pub type DynMatmulKernel = Arc<dyn MatmulKernel>;

#[derive(Default)]
pub struct ReferenceMatmul;

impl ReferenceMatmul {
    pub fn new() -> Self {
        Self
    }
}

impl MatmulKernel for ReferenceMatmul {
    fn name(&self) -> &'static str {
        "reference"
    }

    fn config(&self) -> MatmulTilingConfig {
        MatmulTilingConfig::default()
    }

    fn supports(&self, _problem: &MatmulProblem) -> bool {
        true
    }

    fn run(&self, problem: &MatmulProblem, inputs: &MatmulInputs<'_>) -> Result<Array2<f32>> {
        ensure!(
            problem.batch <= 1,
            "reference kernel does not support batching yet"
        );
        validate_matmul_inputs(&inputs.lhs, &inputs.rhs)?;

        let result = inputs.lhs.dot(&inputs.rhs);
        let result = apply_bias_activation(result, inputs.bias.as_ref(), inputs.activation)?;
        Ok(result)
    }
}

pub struct BlockedMatmul {
    config: MatmulTilingConfig,
}

impl BlockedMatmul {
    pub fn new() -> Self {
        Self {
            config: MatmulTilingConfig {
                tile_m: 128,
                tile_n: 128,
                tile_k: 64,
                unroll: 8,
            },
        }
    }
}

impl Default for BlockedMatmul {
    fn default() -> Self {
        Self::new()
    }
}

impl MatmulKernel for BlockedMatmul {
    fn name(&self) -> &'static str {
        "blocked"
    }

    fn config(&self) -> MatmulTilingConfig {
        self.config
    }

    fn supports(&self, problem: &MatmulProblem) -> bool {
        problem.m >= self.config.tile_m
            && problem.n >= self.config.tile_n
            && problem.k >= self.config.tile_k
    }

    fn run(&self, problem: &MatmulProblem, inputs: &MatmulInputs<'_>) -> Result<Array2<f32>> {
        ensure!(
            problem.batch <= 1,
            "blocked kernel does not support batching yet"
        );
        validate_matmul_inputs(&inputs.lhs, &inputs.rhs)?;

        let mut output = Array2::<f32>::zeros((problem.m, problem.n));
        let tm = self.config.tile_m.min(problem.m.max(1));
        let tn = self.config.tile_n.min(problem.n.max(1));
        let tk = self.config.tile_k.min(problem.k.max(1));

        let lhs = inputs.lhs;
        let rhs = inputs.rhs;

        let m = problem.m;
        let n = problem.n;
        let k = problem.k;

        for i0 in (0..m).step_by(tm) {
            let i_max = (i0 + tm).min(m);
            for j0 in (0..n).step_by(tn) {
                let j_max = (j0 + tn).min(n);
                for p0 in (0..k).step_by(tk) {
                    let p_max = (p0 + tk).min(k);
                    let a_block = lhs.slice(s![i0..i_max, p0..p_max]);
                    let b_block = rhs.slice(s![p0..p_max, j0..j_max]);
                    let mut c_block = output.slice_mut(s![i0..i_max, j0..j_max]);

                    for (row_idx, a_row) in a_block.outer_iter().enumerate() {
                        for (col_idx, b_col) in b_block.axis_iter(ndarray::Axis(1)).enumerate() {
                            let acc = a_row.dot(&b_col);
                            let entry = c_block.get_mut((row_idx, col_idx)).unwrap();
                            *entry += acc;
                        }
                    }
                }
            }
        }

        let output = apply_bias_activation(output, inputs.bias.as_ref(), inputs.activation)?;
        Ok(output)
    }
}

pub struct ParallelMatmul;

impl ParallelMatmul {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ParallelMatmul {
    fn default() -> Self {
        Self::new()
    }
}

impl MatmulKernel for ParallelMatmul {
    fn name(&self) -> &'static str {
        "parallel"
    }

    fn config(&self) -> MatmulTilingConfig {
        MatmulTilingConfig::default()
    }

    fn supports(&self, _problem: &MatmulProblem) -> bool {
        true
    }

    fn run(&self, problem: &MatmulProblem, inputs: &MatmulInputs<'_>) -> Result<Array2<f32>> {
        ensure!(
            problem.batch <= 1,
            "parallel kernel does not support batching yet"
        );
        validate_matmul_inputs(&inputs.lhs, &inputs.rhs)?;

        let lhs = inputs.lhs;
        let rhs = inputs.rhs;
        let mut output = Array2::<f32>::zeros((problem.m, problem.n));

        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(row_idx, mut row)| {
                let lhs_row = lhs.row(row_idx);
                for (col_idx, value) in row.iter_mut().enumerate() {
                    let rhs_col = rhs.column(col_idx);
                    *value = lhs_row.dot(&rhs_col);
                }
            });

        let output = apply_bias_activation(output, inputs.bias.as_ref(), inputs.activation)?;
        Ok(output)
    }
}

pub fn batched_reference_matmul(
    lhs: ArrayView3<'_, f32>,
    rhs: ArrayView3<'_, f32>,
    bias: Option<ArrayView3<'_, f32>>,
    activation: ActivationKind,
) -> Result<Array3<f32>> {
    ensure!(
        lhs.len_of(Axis(0)) == rhs.len_of(Axis(0)),
        "batch dimension mismatch"
    );
    ensure!(
        lhs.len_of(Axis(2)) == rhs.len_of(Axis(1)),
        "inner dimension mismatch"
    );

    let batch = lhs.len_of(Axis(0));
    let m = lhs.len_of(Axis(1));
    let n = rhs.len_of(Axis(2));

    if let Some(bias) = &bias {
        ensure!(
            bias.dim() == (batch, m, n),
            "bias shape {:?} incompatible with output {:?}",
            bias.dim(),
            (batch, m, n)
        );
    }

    let mut output = Array3::<f32>::zeros((batch, m, n));

    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(batch_idx, mut batch_view)| {
            let lhs_batch = lhs.index_axis(Axis(0), batch_idx);
            let rhs_batch = rhs.index_axis(Axis(0), batch_idx);

            for (row_idx, mut row) in batch_view.outer_iter_mut().enumerate() {
                let lhs_row = lhs_batch.row(row_idx);
                for (col_idx, value) in row.iter_mut().enumerate() {
                    let rhs_col = rhs_batch.column(col_idx);
                    *value = lhs_row.dot(&rhs_col);
                }
            }
        });

    if let Some(bias) = bias {
        output += &bias;
    }

    match activation {
        ActivationKind::None => {}
        ActivationKind::Relu => {
            output.par_iter_mut().for_each(|x| {
                if *x < 0.0 {
                    *x = 0.0;
                }
            });
        }
        ActivationKind::Gelu => {
            output.par_iter_mut().for_each(|x| {
                let c = (2.0 / std::f32::consts::PI).sqrt();
                *x = 0.5 * *x * (1.0 + (c * (*x + 0.044715 * x.powi(3))).tanh());
            });
        }
    }

    Ok(output)
}

/// Plan-driven blocked matmul that honors explicit tiling and vector-width hints.
pub struct PlannedMatmul {
    config: MatmulTilingConfig,
}

impl PlannedMatmul {
    pub fn new(config: MatmulTilingConfig) -> Self {
        Self { config }
    }
}

impl MatmulKernel for PlannedMatmul {
    fn name(&self) -> &'static str {
        "plan-tiled"
    }

    fn config(&self) -> MatmulTilingConfig {
        self.config
    }

    fn supports(&self, _problem: &MatmulProblem) -> bool {
        true
    }

    fn run(&self, problem: &MatmulProblem, inputs: &MatmulInputs<'_>) -> Result<Array2<f32>> {
        ensure!(
            problem.batch <= 1,
            "plan-tiled kernel does not support batching yet"
        );
        validate_matmul_inputs(&inputs.lhs, &inputs.rhs)?;

        let mut output = Array2::<f32>::zeros((problem.m, problem.n));
        let tm = self.config.tile_m.max(1);
        let tn = self.config.tile_n.max(1);
        let tk = self.config.tile_k.max(1);
        let vw = self.config.unroll.max(1);

        let lhs = inputs.lhs;
        let rhs = inputs.rhs;

        let m = problem.m;
        let n = problem.n;
        let k = problem.k;

        for i0 in (0..m).step_by(tm) {
            let i_max = (i0 + tm).min(m);
            for j0 in (0..n).step_by(tn) {
                let j_max = (j0 + tn).min(n);
                for p0 in (0..k).step_by(tk) {
                    let p_max = (p0 + tk).min(k);
                    let a_block = lhs.slice(s![i0..i_max, p0..p_max]);
                    let b_block = rhs.slice(s![p0..p_max, j0..j_max]);
                    let mut c_block = output.slice_mut(s![i0..i_max, j0..j_max]);

                    for (row_idx, a_row) in a_block.outer_iter().enumerate() {
                        for (col_idx, b_col) in b_block.axis_iter(ndarray::Axis(1)).enumerate() {
                            let mut acc = 0.0f32;
                            for chunk in (0..a_row.len()).step_by(vw) {
                                let end = (chunk + vw).min(a_row.len());
                                for idx in chunk..end {
                                    acc += a_row[idx] * b_col[idx];
                                }
                            }
                            let entry = c_block.get_mut((row_idx, col_idx)).unwrap();
                            *entry += acc;
                        }
                    }
                }
            }
        }

        let output = apply_bias_activation(output, inputs.bias.as_ref(), inputs.activation)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, Array3, ArrayView2};

    #[test]
    fn batched_reference_matches_individual() {
        let batch = 4;
        let m = 8;
        let k = 16;
        let n = 6;

        let mut lhs = Array3::<f32>::zeros((batch, m, k));
        let mut rhs = Array3::<f32>::zeros((batch, k, n));

        for b in 0..batch {
            for i in 0..m {
                for j in 0..k {
                    lhs[(b, i, j)] = (b + i + j) as f32 * 0.13;
                }
            }
            for i in 0..k {
                for j in 0..n {
                    rhs[(b, i, j)] = (b + i * j + 1) as f32 * 0.07;
                }
            }
        }

        let output = batched_reference_matmul(lhs.view(), rhs.view(), None, ActivationKind::None)
            .expect("batched matmul");

        for b in 0..batch {
            let lhs_slice: ArrayView2<'_, f32> = lhs.index_axis(Axis(0), b);
            let rhs_slice: ArrayView2<'_, f32> = rhs.index_axis(Axis(0), b);
            let reference = lhs_slice.dot(&rhs_slice);
            let batch_slice: ArrayView2<'_, f32> = output.index_axis(Axis(0), b);

            for i in 0..m {
                for j in 0..n {
                    assert_abs_diff_eq!(batch_slice[(i, j)], reference[(i, j)], epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn parallel_matmul_matches_reference() {
        let problem = MatmulProblem::new(32, 24, 16, crate::config::DataType::F32);
        let lhs = Array2::from_shape_fn((problem.m, problem.k), |(i, j)| (i + j) as f32 * 0.1);
        let rhs = Array2::from_shape_fn((problem.k, problem.n), |(i, j)| (i * j + 1) as f32 * 0.05);
        let inputs = MatmulInputs::new(lhs.view(), rhs.view(), None, ActivationKind::None);

        let reference = ReferenceMatmul::new()
            .run(&problem, &inputs)
            .expect("reference matmul");
        let parallel = ParallelMatmul::new()
            .run(&problem, &inputs)
            .expect("parallel matmul");

        for i in 0..problem.m {
            for j in 0..problem.n {
                assert_abs_diff_eq!(reference[(i, j)], parallel[(i, j)], epsilon = 1e-4);
            }
        }
    }
}
