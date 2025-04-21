use super::NeuralNetwork;
use crate::op::linear::{Backward, Forward};
use crate::{Context, NNTensor, VirtualMachine};
pub struct Linear;

impl<VM: VirtualMachine> NeuralNetwork<VM> for Linear {
    type Init = (NNTensor<VM>, Option<NNTensor<VM>>);
    type Args = ();

    fn init(init: Self::Init, ctx: &mut Context<VM>) {
        let (w, b) = init;
        ctx.save_weight("weight", w);
        if let Some(b) = b {
            ctx.save_weight("bias", b)
        }
    }

    fn forward(
        (): &Self::Args,
        inputs: impl IntoIterator<Item = NNTensor<VM>>,
        ctx: &mut Context<VM>,
    ) -> Vec<NNTensor<VM>> {
        let mut inputs = inputs.into_iter();
        let x = inputs.next().unwrap();
        let weight = ctx.load_weight("weight").unwrap();
        let bias = ctx.load_weight("bias");

        if let Some(mut backward) = ctx.take_backward_builder() {
            if let Some(bias) = bias {
                let x_save = x.save();
                let weight_save = weight.save();
                let d_bias = bias.grad();
                let outputs = ctx.call::<Forward>((), [x, weight, bias]);
                let dy = outputs[0].grad();

                backward.call::<Backward>(true, [dy, x_save, weight_save, d_bias], []);
                ctx.put_backward_builder(backward);
            } else {
                let x_save = x.save();
                let weight_save = weight.save();
                let outputs = ctx.call::<Forward>((), [x, weight]);
                let dy = outputs[0].grad();

                backward.call::<Backward>(false, [dy, x_save, weight_save], []);
                ctx.put_backward_builder(backward);
            }
        } else {
            if let Some(bias) = bias {
                ctx.call::<Forward>((), [x, weight, bias]);
            } else {
                ctx.call::<Forward>((), [x, weight]);
            }
        }

        vec![]
    }
}
