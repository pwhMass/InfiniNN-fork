use super::{CpuOp, Tensor, unique};
use crate::types;
use gemm::{Parallelism::Rayon, gemm};
use nn::Arg;
use nn::macros::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::ptr::null;
use std::{iter::zip, mem::size_of, ops::Add};

pub struct Linear;

impl CpuOp for Linear {
    fn launch(&self, inputs: &[Tensor], outputs: &[Tensor], args: Option<&Arg>) {
        let Some(Arg::Bool(residual)) = args else {
            panic!("Expected bool arg for residual");
        };

        destruct!([y] = outputs);
        let mut inputs = inputs.iter();

        let x = inputs.next().unwrap();
        let second = inputs.next().unwrap();
        let w = if *residual {
            inputs.next().unwrap()
        } else {
            second
        };
        let b = if *residual {
            inputs.next()
        } else {
            inputs.next()
        };
        let residual = if *residual { Some(second) } else { None };

        dims!([m, k] = x);
        dims!([n, k2] = w);
        dims!([m2, n2] = y);

        assert_eq!(k, k2);
        assert_eq!(m, m2);
        assert_eq!(n, n2);

        assert!(x.is_big_endian_contiguous());
        assert!(w.is_big_endian_contiguous());
        assert!(y.is_big_endian_contiguous());

        // Handle residual if present
        if let Some(residual) = residual {
            dims!([m3, n3] = residual);
            assert_eq!(m, m3);
            assert_eq!(n, n3);
            assert!(residual.is_big_endian_contiguous());

            unsafe {
                std::ptr::copy_nonoverlapping(
                    *residual.get(),
                    (*y.get()).cast_mut(),
                    m.checked_mul(*n).unwrap() * size_of::<f32>(),
                );
            }
        }

        // Handle bias if present
        if let Some(b) = b {
            dims!([n3] = b);
            assert_eq!(n, n3);
            assert!(b.is_contiguous());

            let y_ptr = *y.get() as usize;
            let b_ptr = *b.get() as usize;

            (0..*m).into_par_iter().for_each(|i| {
                let y_ptr = y_ptr as *mut f32;
                let b_ptr = b_ptr as *const f32;
                for j in 0..*n {
                    unsafe {
                        *y_ptr.add(i * *n + j) = *b_ptr.add(j);
                    }
                }
            });
        }

        unsafe {
            gemm::<f32>(
                *m,
                *n,
                *k,
                (*y.get()).cast_mut().cast::<f32>(),
                1,
                *n as isize,
                residual.is_some() || b.is_some(),
                (*x.get()).cast(),
                1,
                *k as isize,
                (*w.get()).cast(),
                *k as isize,
                1,
                1.,
                1.,
                false,
                false,
                false,
                Rayon(0),
            )
        }
    }
}
