#![allow(dead_code, unused_imports, unused_variables)]

use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

use super::{
    attention::GroupedQueryAttention, config::QuarkConfig, ffn::SwiGluFfn, moe::MoeBlock,
    norm::RmsNorm,
};
use crate::inference::cache::KvCache;

/// A single transformer decoder block (dense or MoE depending on `is_moe`).
#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    input_norm: RmsNorm<B>,
    attn: GroupedQueryAttention<B>,
    post_attn_norm: RmsNorm<B>,
    /// Dense FFN (present on dense layers only).
    ffn: Option<SwiGluFfn<B>>,
    /// MoE block (present on MoE layers only).
    moe: Option<MoeBlock<B>>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn new(cfg: &QuarkConfig, is_moe_layer: bool, device: &B::Device) -> Self {
        Self {
            input_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, device),
            attn: GroupedQueryAttention::new(cfg, device),
            post_attn_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, device),
            ffn: (!is_moe_layer)
                .then(|| SwiGluFfn::new(cfg.hidden_size, cfg.intermediate_size, device)),
            moe: is_moe_layer.then(|| MoeBlock::new(cfg, device)),
        }
    }

    /// Forward pass.
    ///
    /// - `x` shape: `[batch, seq, hidden]`
    /// - `causal`: causal self-attention (always true for a decoder)
    pub fn forward(&self, x: Tensor<B, 3>, causal: bool) -> Tensor<B, 3> {
        self.forward_with_aux(x, causal).0
    }

    /// Incremental forward pass for generation; see
    /// [`GroupedQueryAttention::forward_cached`].
    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        cache: &mut KvCache<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.input_norm.forward(x);
        let x = self.attn.forward_cached(x, cache, start_pos) + residual;
        self.feed_forward(x).0
    }

    pub fn new_cache(&self) -> KvCache<B> {
        self.attn.new_cache()
    }

    /// Forward pass that also returns the MoE load-balancing loss (`None` on
    /// dense layers).
    pub fn forward_with_aux(
        &self,
        x: Tensor<B, 3>,
        causal: bool,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 1>>) {
        // Attention sub-layer with pre-norm and residual
        let residual = x.clone();
        let x = self.input_norm.forward(x);
        let x = self.attn.forward(x, causal);
        let x = x + residual;
        self.feed_forward(x)
    }

    /// FFN (dense or MoE) sub-layer with pre-norm and residual.
    fn feed_forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Option<Tensor<B, 1>>) {
        let residual = x.clone();
        let x = self.post_attn_norm.forward(x);
        let (x, aux) = match (&self.moe, &self.ffn) {
            (Some(moe), _) => {
                let (x, aux) = moe.forward_with_aux(x);
                (x, Some(aux))
            }
            (None, Some(ffn)) => (ffn.forward(x), None),
            (None, None) => unreachable!("decoder block has neither FFN nor MoE"),
        };
        (x + residual, aux)
    }
}
