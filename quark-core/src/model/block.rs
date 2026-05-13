#![allow(dead_code, unused_imports, unused_variables)]

use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

use super::{
    attention::GroupedQueryAttention,
    config::QuarkConfig,
    ffn::SwiGluFfn,
    moe::MoeBlock,
    norm::RmsNorm,
};

/// A single transformer decoder block (dense or MoE depending on `is_moe`).
#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    input_norm: RmsNorm<B>,
    attn: GroupedQueryAttention<B>,
    post_attn_norm: RmsNorm<B>,
    /// Dense FFN (used when `is_moe` is false).
    ffn: SwiGluFfn<B>,
    /// MoE block (used when `is_moe` is true).
    moe: MoeBlock<B>,
    /// Selects between dense FFN and MoE; set at construction time.
    is_moe: bool,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn new(cfg: &QuarkConfig, is_moe_layer: bool, device: &B::Device) -> Self {
        Self {
            input_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, device),
            attn: GroupedQueryAttention::new(cfg, device),
            post_attn_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, device),
            ffn: SwiGluFfn::new(cfg.hidden_size, cfg.intermediate_size, device),
            moe: MoeBlock::new(cfg, device),
            is_moe: is_moe_layer,
        }
    }

    /// Forward pass.
    ///
    /// - `x` shape: `[batch, seq, hidden]`
    /// - `mask`: optional additive causal mask `[1, 1, seq, seq]`
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        // Attention sub-layer with pre-norm and residual
        let residual = x.clone();
        let x = self.input_norm.forward(x);
        let x = self.attn.forward(x, mask);
        let x = x + residual;

        // FFN sub-layer with pre-norm and residual
        let residual = x.clone();
        let x = self.post_attn_norm.forward(x);
        let x = if self.is_moe {
            self.moe.forward(x)
        } else {
            self.ffn.forward(x)
        };
        x + residual
    }
}
