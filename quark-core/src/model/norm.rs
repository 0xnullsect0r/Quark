#![allow(dead_code, unused_imports, unused_variables)]

use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor},
};

/// Root-Mean-Square Layer Normalization.
///
/// Formula: `Y = X / sqrt(mean(X^2) + eps) * weight`
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    eps: f64,
    hidden_size: usize,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(hidden_size: usize, eps: f64, device: &B::Device) -> Self {
        let weight = Param::from_tensor(Tensor::<B, 1>::ones([hidden_size], device));
        Self { weight, eps, hidden_size }
    }

    /// Forward pass. Input/output shape: `[batch, seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, hidden] = x.dims();
        // rms shape: [batch, seq, 1] (mean_dim keeps the dim)
        let rms = (x.clone().powf_scalar(2.0).mean_dim(2) + self.eps).sqrt();
        // weight: [hidden] -> [1, 1, hidden] for broadcasting
        let w = self.weight.val().reshape([1, 1, hidden]);
        (x / rms) * w
    }
}
