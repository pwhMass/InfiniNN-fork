use super::{CpuOp, Tensor, unique};
use crate::types;
use nn::Arg;
use nn::macros::*;
use std::ptr::null;
use std::{iter::zip, mem::size_of, ops::Add};

pub trait Index {
    fn as_usize(&self) -> usize;
}

impl Index for u16 {
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

pub struct Embedding;

impl CpuOp for Embedding {
    fn launch(&self, inputs: &[Tensor], outputs: &[Tensor], args: Option<&Arg>) {
        assert!(args.is_none());
        destruct!([y] = outputs);
        let mut inputs = inputs.iter();

        let wte = inputs.next().unwrap();
        let tokens = inputs.next().unwrap();

        dims!([_, d] = wte);
        dims!([n] = tokens);
        dims!([n2, d2] = y);

        assert_eq!(n, n2);
        assert_eq!(d, d2);

        // 验证步长

        strides!([nsy, dsy] = y);
        strides!([ns1] = tokens);

        assert_eq!(dsy, y.dt().nbytes() as isize);
        assert_eq!(ns1, tokens.dt().nbytes() as isize);
        assert!(wte.is_contiguous());
        assert!(tokens.is_contiguous());

        let mut scheme = Scheme {
            n: *n,
            d: *d,
            nsy: nsy,
            y: *y.get(),
            tokens: *tokens.get(),
            pos: null(),
            wte: *wte.get(),
            wpe: null(),
            has_wpe: false,
        };
        let mut pos_dt = None;
        if let Some(wpe) = inputs.next() {
            let pos = inputs.next().unwrap();
            pos_dt = Some(pos.dt());

            dims!([_, d2] = wpe);
            dims!([n2] = pos);

            assert_eq!(d, d2);
            assert_eq!(n, n2);

            //验证步长
            strides!([ns2] = pos);
            assert_eq!(ns2, pos.dt().nbytes() as isize);
            assert!(wpe.is_big_endian_contiguous());
            assert!(pos.is_big_endian_contiguous());

            scheme.wpe = *wpe.get();
            scheme.pos = *pos.get();
            scheme.has_wpe = true;
        }
        match (y.dt(), tokens.dt(), pos_dt.unwrap_or(tokens.dt())) {
            (types::F32, types::U16, types::U16) => scheme.compute::<f32, u16, u16>(),
            (_, _, _) => todo!(),
        }
    }
}

struct Scheme {
    n: usize,
    d: usize,
    nsy: isize,
    y: *const u8,
    tokens: *const u8,
    pos: *const u8,
    wte: *const u8,
    wpe: *const u8,
    has_wpe: bool,
}

impl Scheme {
    fn compute<T: Add<Output = T>, I1: Index, I2: Index>(&self) {
        let &Self {
            n,
            d,
            nsy,
            y,
            tokens: i1,
            pos: i2,
            wte: table1,
            wpe: table2,
            has_wpe,
        } = self;
        if has_wpe {
            let i1 = unsafe { std::slice::from_raw_parts(i1.cast::<I1>(), n) };
            let i2 = unsafe { std::slice::from_raw_parts(i2.cast::<I2>(), n) };
            for (i, (i1, i2)) in zip(i1, i2).enumerate() {
                let y = unsafe { y.cast_mut().byte_offset(nsy * i as isize) }.cast::<T>();
                let x1 = unsafe { table1.byte_add(i1.as_usize() * d * size_of::<T>()) }.cast::<T>();
                let x2 = unsafe { table2.byte_add(i2.as_usize() * d * size_of::<T>()) }.cast::<T>();
                for i in 0..d {
                    unsafe { y.add(i).write(x1.add(i).read() + x2.add(i).read()) }
                }
            }
        } else {
            let i1 = unsafe { std::slice::from_raw_parts(i1.cast::<I1>(), n) };
            for (i, i1) in i1.into_iter().enumerate() {
                let y = unsafe { y.cast_mut().byte_offset(nsy * i as isize) }.cast::<T>();
                let x1 = unsafe { table1.byte_add(i1.as_usize() * d * size_of::<T>()) }.cast::<T>();
                for i in 0..d {
                    unsafe { y.add(i).write(x1.add(i).read()) }
                }
            }
        }
    }
}
