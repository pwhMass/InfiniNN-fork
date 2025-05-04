// use super::{CpuOp, Tensor, unique};
// use super::{OpError, Operator};
// use crate::{Arg, Dim, TensorMeta};
// use nn::macros::*;
// use operators::TensorLayout;
// use operators::TensorLayout;
// use rayon::iter::{IntoParallelIterator, ParallelIterator};
// use std::{iter::zip, mem::size_of, ops::Add};

// pub struct Concat;

// impl Operator for Concat {
//     fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
//         let Some(Arg::Int(axis)) = args else {
//             return Err(OpError::ArgError);
//         };
//         let axis = *axis as usize;

//         // TODO 判定其他维度相等

//         let dt = inputs[0].dt;
//         let mut origin_shape = inputs[0].shape.to_vec();

//         let axis_sum = inputs
//             .iter()
//             .map(|t| t.shape[axis].clone())
//             .fold(Dim::Constant(0), |acc, d| acc + d);

//         origin_shape[axis] = axis_sum;

//         Ok(vec![TensorMeta::new(dt, origin_shape)])
//     }
// }

// impl CpuOp for Concat {
//     fn launch(&self, inputs: &[Tensor], outputs: &[Tensor], args: Option<&Arg>) {
//         let Some(Arg::Int(axis)) = args else {
//             panic!("Expected int arg for axis");
//         };
//         let axis = *axis as usize;

//         destruct!([y] = outputs);
//         let inputs: Vec<&Tensor> = inputs.iter().collect();

//         // 验证所有输入张量的维度数量相同
//         let ndim = inputs[0].shape().len();
//         assert!(inputs.iter().all(|x| x.shape().len() == ndim));

//         // 验证除了拼接轴外的其他维度相同
//         for d in 0..ndim {
//             if d != axis {
//                 let dim = inputs[0].shape()[d];
//                 assert!(inputs.iter().all(|x| x.shape()[d] == dim));
//             }
//         }

//         // 计算在axis维度上的总长度和每个输入的长度
//         let mut offset = 0;
//         for input in inputs {
//             let input_size = input.shape()[axis];

//             // 计算需要复制的连续内存块大小
//             let block_size = (axis + 1..ndim)
//                 .map(|d| input.shape()[d])
//                 .product::<usize>()
//                 * size_of::<f32>();

//             // 计算每个维度的步长
//             let outer_dim = (0..axis).map(|d| input.shape()[d]).product::<usize>();

//             // 并行复制数据
//             (0..outer_dim).into_par_iter().for_each(|i| unsafe {
//                 let src = (*input.get() as *const u8).add(i * input_size * block_size);
//                 let dst = (*y.get() as *mut u8)
//                     .add(i * y.shape()[axis] * block_size)
//                     .add(offset * block_size);
//                 std::ptr::copy_nonoverlapping(src, dst, input_size * block_size);
//             });

//             offset += input_size;
//         }
//     }
// }
