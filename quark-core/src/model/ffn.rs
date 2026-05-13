#![allow(dead_code, unused_imports, unused_variables)]

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::silu, backend::Backend, Tensor},
};

use super::config::QuarkConfig;

/// SwiGLU Feed-Forward Network.
///
/// Formula: `out = down_proj(silu(gate_proj(x)) * up_proj(x))`
#[derive(Module, Debug)]
pub struct SwiGluFfn<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> SwiGluFfn<B> {
    pub fn new(hidden_size: usize, intermediate_size: usize, device: &B::Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(intermediate_size, hidden_size)
                .with_bias(false)
                .init(device),
        }
    }

    /// Forward pass. Input/output shape: `[batch, seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}
