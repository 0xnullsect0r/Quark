#![allow(dead_code, unused_imports, unused_variables)]

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Tensor},
};

use super::{config::QuarkConfig, ffn::SwiGluFfn};

/// Learned router for Mixture-of-Experts.
///
/// Produces a softmax distribution over experts.  The top-k experts for each
/// token are selected in `MoeBlock::forward`; others are zeroed out and the
/// surviving weights are renormalised.
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

    /// Returns router logits `[batch, seq, num_experts]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.gate.forward(x)
    }
}

/// Mixture-of-Experts block.
///
/// Uses top-k sparse routing: for each token only the top-k experts run
/// (the rest receive a routing weight of 0).  Their weights are renormalised
/// so they sum to 1.
#[derive(Module, Debug)]
pub struct MoeBlock<B: Backend> {
    router: MoeRouter<B>,
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
    ///
    /// Sparse top-k routing: for each token the top-k expert weights are
    /// kept and renormalised; others are zeroed out so only top-k experts
    /// contribute to the output.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = x.device();
        let [batch, seq, hidden] = x.dims();
        let top_k = self.num_experts_per_tok.min(self.num_experts);

        // Router logits and softmax weights: [batch, seq, num_experts]
        let logits = self.router.forward(x.clone());
        let weights = softmax(logits, 2);

        let final_weights = top_k_weights(weights, top_k); // [batch, seq, num_experts]

        // Weighted sum over expert outputs (non-top-k experts have weight ≈ 0).
        let mut output = Tensor::<B, 3>::zeros([batch, seq, hidden], &device);
        for (i, expert) in self.experts.iter().enumerate() {
            let expert_out = expert.forward(x.clone()); // [batch, seq, hidden]
            let w = final_weights.clone().narrow(2, i, 1); // [batch, seq, 1]
            output = output + expert_out * w;
        }

        output
    }
}

/// Keep the `top_k` largest routing weights per token (renormalised to sum
/// to 1) and zero out the rest. Input/output shape: `[batch, seq, num_experts]`.
pub fn top_k_weights<B: Backend>(weights: Tensor<B, 3>, top_k: usize) -> Tensor<B, 3> {
    // k-th largest weight per token; everything >= it is in the top-k set.
    let threshold = weights.clone().topk(top_k, 2).narrow(2, top_k - 1, 1); // [batch, seq, 1]
    let mask = weights.clone().greater_equal(threshold).float();
    let kept = weights * mask;
    let sum = kept.clone().sum_dim(2) + 1e-9_f32; // [batch, seq, 1]
    kept / sum
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    use super::*;

    type B = NdArray<f32>;

    fn weights() -> Tensor<B, 3> {
        let data = TensorData::new(vec![0.1f32, 0.6, 0.3, 0.5, 0.2, 0.3], [1, 2, 3]);
        Tensor::from_data(data, &Default::default())
    }

    #[test]
    fn top1_is_one_hot() {
        let out: Vec<f32> = top_k_weights(weights(), 1).into_data().into_vec().unwrap();
        let expected = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        for (a, b) in out.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "{out:?}");
        }
    }

    #[test]
    fn top2_keeps_two_and_renormalises() {
        let out: Vec<f32> = top_k_weights(weights(), 2).into_data().into_vec().unwrap();
        let expected = [0.0, 0.6 / 0.9, 0.3 / 0.9, 0.5 / 0.8, 0.0, 0.3 / 0.8];
        for (a, b) in out.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "{out:?}");
        }
    }
}
