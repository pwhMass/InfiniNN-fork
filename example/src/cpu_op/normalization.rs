use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::{CpuOp, Tensor, unique};
use crate::types;
use gemm::{Parallelism::Rayon, gemm};
use nn::Arg;
use nn::macros::*;
use std::ptr::null;
use std::{iter::zip, mem::size_of, ops::Add};
pub struct RmsNorm;

impl CpuOp for RmsNorm {
    fn launch(&self, inputs: &[Tensor], outputs: &[Tensor], args: Option<&Arg>) {
        let Some(Arg::Float(epsilon)) = args else {
            panic!("Expected f32 arg for epsilon");
        };

        destruct!([y] = outputs);
        destruct!([x, scale] = inputs);

        dims!([n, d] = x);
        dims!([d2] = scale);
        dims!([n2, d3] = y);

        assert_eq!(d, d2);
        assert_eq!(n, n2);
        assert_eq!(d, d3);

        let x_ptr = *x.get() as usize;
        let scale_ptr = *scale.get() as usize;
        let y_ptr = *y.get() as usize;

        strides!([nsx, dsx] = x);
        strides!([s_scale] = scale);
        strides!([nsy, dsy] = y);

        (0..*n).into_par_iter().for_each(|i| {
            let x_ptr = x_ptr as *const f32;
            let scale_ptr = scale_ptr as *const f32;
            let y_ptr = y_ptr as *mut f32;
            // 计算均方根
            let mut sum_sq = 0.0f32;
            for j in 0..*d {
                unsafe {
                    let x_val = *x_ptr.add((i as isize * nsx + j as isize * dsx) as usize);
                    sum_sq += x_val * x_val;
                }
            }
            let rms = (sum_sq / (*d as f32) + *epsilon as f32).sqrt();

            // 应用归一化和缩放
            for j in 0..*d {
                unsafe {
                    let x_val = *x_ptr.add((i as isize * nsx + j as isize * dsx) as usize);
                    let scale_val = *scale_ptr.add((j as isize * s_scale) as usize);
                    *y_ptr.add((i as isize * nsy + j as isize * dsy) as usize) =
                        (x_val / rms) * scale_val;
                }
            }
        });
    }
}

pub struct LayerNorm;

impl CpuOp for LayerNorm {
    fn launch(&self, inputs: &[Tensor], outputs: &[Tensor], args: Option<&Arg>) {
        let Some(Arg::Float(epsilon)) = args else {
            panic!("Expected f32 arg for epsilon");
        };

        destruct!([y] = outputs);
        destruct!([x, scale, bias] = inputs);

        dims!([n, d] = x);
        dims!([d2] = scale);
        dims!([d3] = bias);
        dims!([n2, d4] = y);

        assert_eq!(n, n2);
        let d = unique(&[d, d2, d3, d4]).unwrap();

        strides!([nsx, dsx] = x);
        strides!([s_scale] = scale);
        strides!([b_scale] = bias);
        strides!([nsy, dsy] = y);

        let x_ptr = *x.get() as usize;
        let scale_ptr = *scale.get() as usize;
        let bias_ptr = *bias.get() as usize;
        let y_ptr = *y.get() as usize;

        (0..*n).into_par_iter().for_each(|i| {
            let x_ptr = x_ptr as *const f32;
            let scale_ptr = scale_ptr as *const f32;
            let bias_ptr = bias_ptr as *const f32;
            let y_ptr = y_ptr as *mut f32;

            // 计算均值
            let mut sum = 0.0f32;
            for j in 0..*d {
                unsafe {
                    sum += *x_ptr.add((i as isize * nsx + j as isize * dsx) as usize);
                }
            }
            let mean = sum / (*d as f32);

            // 计算方差
            let mut sum_sq = 0.0f32;
            for j in 0..*d {
                unsafe {
                    let x_val = *x_ptr.add((i as isize * nsx + j as isize * dsx) as usize);
                    let diff = x_val - mean;
                    sum_sq += diff * diff;
                }
            }
            let var = sum_sq / (*d as f32);
            let rstd = 1.0 / (var + (*epsilon as f32)).sqrt();

            // 应用归一化、缩放和偏置
            for j in 0..*d {
                unsafe {
                    let x_val = *x_ptr.add((i as isize * nsx + j as isize * dsx) as usize);
                    let scale_val = *scale_ptr.add((j as isize * s_scale) as usize);
                    let bias_val = *bias_ptr.add((j as isize * b_scale) as usize);
                    let normalized = (x_val - mean) * rstd;
                    *y_ptr.add((i as isize * nsy + j as isize * dsy) as usize) =
                        normalized * scale_val + bias_val;
                }
            }
        });
    }
}
