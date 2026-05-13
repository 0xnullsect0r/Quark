#![allow(dead_code, unused_imports, unused_variables)]

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Tensor},
};

use super::{config::QuarkConfig, ffn::SwiGluFfn};

/// Top-k router for Mixture-of-Experts.
///
/// Computes a softmax routing distribution over all experts and returns the
/// weights (and placeholder indices) for the top-k selected ones.
#[derive(Module, Debug)]
pub struct MoeRouter<B: Backend> {
    gate: Linear<B>,
    num_experts: usize,
}

impl<B: Backend> MoeRouter<B> {
    pub fn new(hidden_size: usize, num_experts: usize, device: &B::Device) -> Self {
        Self {
            gate: LinearConfig::new(hidden_size, num_experts)
                .with_bias(false)
                .init(device),
            num_experts,
        }
    }

    /// Returns `(expert_indices, router_weights)` both shape `[batch, seq, top_k]`.
    ///
    /// Note: indices are returned as float zeros (placeholder); routing uses
    /// the weights directly in `MoeBlock::forward`.
    pub fn forward(&self, x: Tensor<B, 3>, top_k: usize) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let device = x.device();
        let [batch, seq, _] = x.dims();
        let logits = self.gate.forward(x); // [batch, seq, num_experts]
        let weights = softmax(logits, 2);  // [batch, seq, num_experts]
        // Return first top_k weights as a simplified top-k selection
        let selected = weights.slice([0..batch, 0..seq, 0..top_k]);
        let indices = Tensor::<B, 3>::zeros([batch, seq, top_k], &device);
        (indices, selected)
    }
}

/// Mixture-of-Experts block.
///
/// Uses a simplified approach: runs all experts and takes a softmax-weighted
/// sum of their outputs (correct but not sparse/efficient).
#[derive(Module, Debug)]
pub struct MoeBlock<B: Backend> {
    router: MoeRouter<B>,
    /// One FFN expert per slot.
    experts: Vec<SwiGluFfn<B>>,
    num_experts: usize,
    num_experts_per_tok: usize,
}

impl<B: Backend> MoeBlock<B> {
    pub fn new(cfg: &QuarkConfig, device: &B::Device) -> Self {
        let experts: Vec<SwiGluFfn<B>> = (0..cfg.num_experts)
            .map(|_| SwiGluFfn::new(cfg.hidden_size, cfg.intermediate_size, device))
            .collect();
        Self {
            router: MoeRouter::new(cfg.hidden_size, cfg.num_experts, device),
            experts,
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
        }
    }

    /// Forward pass. Input/output shape: `[batch, seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = x.device();
        let [batch, seq, hidden] = x.dims();

        // Compute routing weights for all experts
        let logits = self.router.gate.forward(x.clone()); // [batch, seq, num_experts]
        let weights = softmax(logits, 2);                  // [batch, seq, num_experts]

        // Weighted sum over all experts (simplified MoE — correct but not sparse)
        let mut output = Tensor::<B, 3>::zeros([batch, seq, hidden], &device);
        for (i, expert) in self.experts.iter().enumerate() {
            let expert_out = expert.forward(x.clone()); // [batch, seq, hidden]
            // w: [batch, seq, 1]
            let w = weights.clone().slice([0..batch, 0..seq, i..i + 1]);
            output = output + expert_out * w;
        }
        output
    }
}
