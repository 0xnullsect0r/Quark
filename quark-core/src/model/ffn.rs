#![allow(dead_code, unused_imports, unused_variables)]

use burn::{
    module::Module,
    tensor::{activation::silu, backend::Backend, Tensor},
};

use super::{config::QuarkConfig, proj::Proj};

/// SwiGLU Feed-Forward Network.
///
/// Formula: `out = down_proj(silu(gate_proj(x)) * up_proj(x))`
#[derive(Module, Debug)]
pub struct SwiGluFfn<B: Backend> {
    gate_proj: Proj<B>,
    up_proj: Proj<B>,
    down_proj: Proj<B>,
}

impl<B: Backend> SwiGluFfn<B> {
    pub fn new(hidden_size: usize, intermediate_size: usize, device: &B::Device) -> Self {
        Self {
            gate_proj: Proj::new(hidden_size, intermediate_size, device),
            up_proj: Proj::new(hidden_size, intermediate_size, device),
            down_proj: Proj::new(intermediate_size, hidden_size, device),
        }
    }

    /// Forward pass. Input/output shape: `[batch, seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }

    /// The projections, by path relative to this module.
    pub fn projs_mut(&mut self, prefix: &str) -> Vec<(String, &mut Proj<B>)> {
        vec![
            (format!("{prefix}gate_proj"), &mut self.gate_proj),
            (format!("{prefix}up_proj"), &mut self.up_proj),
            (format!("{prefix}down_proj"), &mut self.down_proj),
        ]
    }
}
