pub mod activation;
pub mod attention;
pub mod embedding;
pub mod linear;
pub mod normalization;

use nn::Arg;
type Tensor = tensor::Tensor<*const u8, 2>;

trait CpuOp {
    fn launch(&self, inputs: &[Tensor], outputs: &[Tensor], args: Option<&Arg>);
}

fn unique<T: Copy + Eq>(vals: &[T]) -> Option<T> {
    let [val, tail @ ..] = vals else {
        return None;
    };
    for v in tail {
        if v != val {
            return None;
        }
    }
    Some(*val)
}
