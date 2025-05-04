use super::{CpuOp, Tensor};
use nn::Arg;
use nn::macros::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{f32::consts::PI, sync::LazyLock};

const GELU_MAGIC: f32 = 0.044715;
static GELU_FACTOR: LazyLock<f32> = LazyLock::new(|| (2. / PI).sqrt());

// pub struct SwiGLU;

// impl CpuOp for SwiGLU {
//     fn launch(&self, inputs: &[Tensor], outputs: &[Tensor], args: Option<&Arg>) {
//         if args.is_some() {
//             panic!("SwiGLU does not accept arguments");
//         }

//         destruct!([gate, up] = inputs);
//         destruct!([y] = outputs);

//         dims!([m, n] = gate);
//         dims!([m2, n2] = up);
//         dims!([m3, n3] = y);

//         assert_eq!(m, m2);
//         assert_eq!(n, n2);
//         assert_eq!(m, m3);
//         assert_eq!(n, n3);

//         assert!(gate.is_big_endian_contiguous());
//         assert!(up.is_big_endian_contiguous());
//         assert!(y.is_big_endian_contiguous());

//         let gate_ptr = *gate.get() as usize;
//         let up_ptr = *up.get() as usize;
//         let y_ptr = *y.get() as usize;

//         (0..*m).into_par_iter().for_each(|i| {
//             let gate_ptr = gate_ptr as *const f32;
//             let up_ptr = up_ptr as *const f32;
//             let y_ptr = y_ptr as *mut f32;

//             for j in 0..*n {
//                 unsafe {
//                     let gate_val = *gate_ptr.add(i * *n + j);
//                     let up_val = *up_ptr.add(i * *n + j);

//                     // SwiGLU: gate * sigmoid(gate) * up
//                     let sigmoid_gate = 1.0 / (1.0 + (-gate_val).exp());
//                     *y_ptr.add(i * *n + j) = gate_val * sigmoid_gate * up_val;
//                 }
//             }
//         });
//     }
// }

pub struct GeLU;

impl CpuOp for GeLU {
    fn launch(&self, inputs: &[Tensor], outputs: &[Tensor], args: Option<&Arg>) {
        if args.is_some() {
            panic!("GeLU does not accept arguments");
        }

        destruct!([x] = inputs);
        destruct!([y] = outputs);

        dims!([n, d] = x);
        dims!([n2, d2] = y);

        assert_eq!(n, n2);
        assert_eq!(d, d2);

        assert!(x.is_big_endian_contiguous());
        assert!(y.is_big_endian_contiguous());

        let x_ptr = *x.get() as usize;
        let y_ptr = *y.get() as usize;

        (0..*n).into_par_iter().for_each(|i| {
            let x_ptr = x_ptr as *const f32;
            let y_ptr = y_ptr as *mut f32;

            for j in 0..*d {
                unsafe {
                    let x_val = *x_ptr.add(i * *d + j);

                    // GeLU algorithm from gelu.rs
                    let x3 = GELU_MAGIC * x_val.powi(3);
                    let tanh = *GELU_FACTOR * (x_val + x3);
                    let result = 0.5 * x_val * (1. + tanh.tanh());

                    *y_ptr.add(i * *d + j) = result;
                }
            }
        });
    }
}
