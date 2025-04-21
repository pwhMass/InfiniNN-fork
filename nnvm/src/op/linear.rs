use std::any::Any;

use super::Operator;
use crate::{Domain, NNTensor, VirtualMachine, kernel::MutMulArgs, macros::dims};

pub struct Forward;
pub struct Backward;

impl<VM: VirtualMachine> Operator<VM> for Forward {
    type Args = ();

    fn call(
        (): &Self::Args,
        inputs: impl IntoIterator<Item = NNTensor<VM>>,
        domain: &VM::Domain,
    ) -> Vec<NNTensor<VM>> {
        let mut inputs = inputs.into_iter();
        let x = inputs.next().unwrap();
        let weight = inputs.next().unwrap();

        let maybe_bias = inputs.next();
        dims!([m, k] = x);
        dims!([n, k_] = weight);

        assert_eq!(k, k_);

        let mut y = NNTensor::<VM>::from(domain.tensor(x.dt(), &[m, n]));

        let weight = weight.transform(|layout| layout.transpose(&[1, 0]));
        match maybe_bias {
            Some(bias) => {
                dims!([n_] = bias);
                assert_eq!(n, n_);
                let bias = bias.transform(|layout| layout.tile_be(0, &[1, n]).broadcast(0, m));

                domain.launch("rearrange", [y.kernel_mut(), bias.kernel_ref()], []);

                const ATTR: &MutMulArgs = &MutMulArgs {
                    beta: 1.0,
                    alpha: 1.0,
                    read_dst: true,
                };
                domain.launch(
                    "mut-mul",
                    [y.kernel_mut(), x.kernel_ref(), weight.kernel_ref()],
                    [ATTR as &dyn Any],
                );
            }
            None => {
                const ATTR: &MutMulArgs = &MutMulArgs {
                    beta: 1.0,
                    alpha: 1.0,
                    read_dst: false,
                };
                domain.launch(
                    "mut-mul",
                    [y.kernel_mut(), x.kernel_ref(), weight.kernel_ref()],
                    [ATTR as &dyn Any],
                );
            }
        }

        vec![y]
    }
}

impl<VM: VirtualMachine> Operator<VM> for Backward {
    // has_bias: true 表示有偏置，false 表示没有偏置
    type Args = bool;

    fn call(
        has_bias: &Self::Args,
        inputs: impl IntoIterator<Item = NNTensor<VM>>,
        domain: &VM::Domain,
    ) -> Vec<NNTensor<VM>> {
        let mut inputs = inputs.into_iter();

        let dy = inputs.next().unwrap();
        let x = inputs.next().unwrap();
        let weight = inputs.next().unwrap();

        dims!([m, k] = x);
        dims!([n, k_] = weight);
        dims!([m_, n_] = dy);
        assert_eq!(k, k_);
        assert_eq!(m, m_);
        assert_eq!(n, n_);

        let mut dx = NNTensor::<VM>::from(domain.tensor(x.dt(), &[m, k]));
        let mut dw = NNTensor::<VM>::from(domain.tensor(x.dt(), &[n, k]));

        const ATTR: &MutMulArgs = &MutMulArgs {
            beta: 1.0,
            alpha: 1.0,
            read_dst: false,
        };

        // 计算 dx
        domain.launch(
            "mut-mul",
            [dx.kernel_mut(), dy.kernel_ref(), weight.kernel_ref()],
            [ATTR as &dyn Any],
        );

        // 计算 dw
        let dy_transpose = dy.clone().transform(|layout| layout.transpose(&[1, 0]));
        domain.launch(
            "mut-mul",
            [dw.kernel_mut(), dy_transpose.kernel_ref(), x.kernel_ref()],
            [ATTR as &dyn Any],
        );

        // 计算 db
        // TODO 是造个单位元之后broadcast 还是弄一个 sum 的 kernel
        if *has_bias {
            let mut db = NNTensor::<VM>::from(domain.tensor(x.dt(), &[n]));
            domain.launch(
                "sum",
                [db.kernel_mut(), dy.kernel_ref()],
                [ATTR as &dyn Any],
            );
            vec![dx, dw, db]
        } else {
            vec![dx, dw]
        }
    }
}
