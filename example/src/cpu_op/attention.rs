use super::{CpuOp, Tensor, unique};
use crate::types;
use nn::Arg;
use nn::macros::*;
use std::{
    mem::size_of,
    slice::{from_raw_parts, from_raw_parts_mut},
};

macro_rules! assert_mod_0_div {
    ($a:expr, $b:expr) => {{
        assert_eq!($a % $b, 0);
        $a / $b
    }};
}

pub struct Attention;

impl CpuOp for Attention {
    fn launch(&self, inputs: &[Tensor], outputs: &[Tensor], args: Option<&Arg>) {
        let Some(Arg::Int(dh)) = args else {
            panic!("Expected Dim arg for attention");
        };

        destruct!([y] = outputs);
        let mut inputs = inputs.iter();

        let q = inputs.next().unwrap();
        let k = inputs.next().unwrap();
        let v = inputs.next().unwrap();

        let dt = unique(&[q.dt(), k.dt(), v.dt(), y.dt()]).unwrap();
        assert_eq!(dt, types::F32);

        dims!([n, dq] = q);
        dims!([n, dk] = k);
        dims!([n, dv] = v);
        dims!([n, dy] = y);

        assert_eq!(dy, dq);

        let q_head = assert_mod_0_div!(*dq, *dh as usize);
        let k_head = assert_mod_0_div!(*dk, *dh as usize);
        let v_head = assert_mod_0_div!(*dv, *dh as usize);
        assert_eq!(*dy, *dq);

        assert_eq!(k_head, v_head);

        // 计算注意力
        let dh_val = *dh as usize;
        let head_count = *dq / dh_val; // 头的数量
        let scale = (dh_val as f32).powf(-0.5);

        // 每个头的维度需要符合要求
        assert_eq!(*dq % head_count, 0);

        // 临时张量存储中间结果
        let mut preatt = vec![0.0f32; *n * head_count * *n];
        let mut att = vec![0.0f32; *n * head_count * *n];

        // 计算注意力权重
        unsafe {
            let q_ptr = *q.get() as *const f32;
            let k_ptr = *k.get() as *const f32;
            let v_ptr = *v.get() as *const f32;
            let y_ptr = *y.get() as *mut f32;

            for b in 0..*n {
                for h in 0..head_count {
                    // 计算当前序列位置的q [b]
                    let q_base_offset = b * *dq;

                    // 计算对应的输出y位置 [b]
                    let y_base_offset = b * *dy;

                    // 计算此头的偏移
                    let head_offset = h * dh_val;

                    // 计算当前q头的位置 [b, h*dh:(h+1)*dh]
                    let q_head_ptr = q_ptr.add(q_base_offset + head_offset);
                    let q_slice = from_raw_parts(q_head_ptr, dh_val);

                    // 计算输出位置 [b, h*dh:(h+1)*dh]
                    let y_head_ptr = y_ptr.add(y_base_offset + head_offset);
                    let y_slice = from_raw_parts_mut(y_head_ptr, dh_val);

                    // 初始化输出为0
                    for i in 0..dh_val {
                        y_slice[i] = 0.0;
                    }

                    // 计算每个位置的注意力
                    let att_base_idx = (b * head_count + h) * *n;

                    // 计算当前query与所有key的点积
                    let mut max_val = f32::NEG_INFINITY;

                    // 对所有序列位置计算注意力
                    for t in 0..*n {
                        // 当前key的位置 [t]
                        let k_base_offset = t * *dk;

                        // 计算key头的位置 [t, h*dh:(h+1)*dh]
                        let k_head_ptr = k_ptr.add(k_base_offset + head_offset);
                        let k_slice = from_raw_parts(k_head_ptr, dh_val);

                        // 计算点积
                        let mut dot_product = 0.0f32;
                        for i in 0..dh_val {
                            dot_product += q_slice[i] * k_slice[i];
                        }
                        dot_product *= scale;

                        preatt[att_base_idx + t] = dot_product;
                        if dot_product > max_val {
                            max_val = dot_product;
                        }
                    }

                    // 计算softmax
                    let mut exp_sum = 0.0f32;
                    for t in 0..*n {
                        let exp_val = (preatt[att_base_idx + t] - max_val).exp();
                        att[att_base_idx + t] = exp_val;
                        exp_sum += exp_val;
                    }

                    // 归一化
                    let exp_sum_inv = 1.0 / exp_sum;
                    for t in 0..*n {
                        att[att_base_idx + t] *= exp_sum_inv;
                    }

                    // 计算加权和
                    for t in 0..*n {
                        let att_val = att[att_base_idx + t];

                        // 获取value向量的位置 [t]
                        let v_base_offset = t * *dv;

                        // 计算value头的位置 [t, h*dh:(h+1)*dh]
                        let v_head_ptr = v_ptr.add(v_base_offset + head_offset);
                        let v_slice = from_raw_parts(v_head_ptr, dh_val);

                        // 累加到输出
                        for i in 0..dh_val {
                            y_slice[i] += att_val * v_slice[i];
                        }
                    }
                }
            }
        }
    }
}
