//! The model split into stages that can be loaded, run and dropped one at a
//! time: the token embedding, each [`DecoderBlock`], and the head (final norm
//! plus LM head). Parameter paths inside each stage match those inside
//! [`QuarkModel`](super::QuarkModel), so stages and whole models convert freely.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Int, Tensor, TensorData},
};

use super::{config::QuarkConfig, norm::RmsNorm};

/// Stage name of the token embedding.
pub const EMBED_STAGE: &str = "embed";
/// Stage name of the final norm + LM head.
pub const HEAD_STAGE: &str = "head";

/// Stage name of decoder layer `i`.
pub fn layer_stage(i: usize) -> String {
    format!("layer.{i:04}")
}

/// Every stage name of a model, in forward order.
pub fn stage_names(cfg: &QuarkConfig) -> Vec<String> {
    let mut names = vec![EMBED_STAGE.to_owned()];
    names.extend((0..cfg.num_hidden_layers).map(layer_stage));
    names.push(HEAD_STAGE.to_owned());
    names
}

#[derive(Module, Debug)]
pub struct EmbedStage<B: Backend> {
    pub embed_tokens: Embedding<B>,
}

impl<B: Backend> EmbedStage<B> {
    pub fn new(cfg: &QuarkConfig, device: &B::Device) -> Self {
        Self { embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device) }
    }

    /// `[batch, seq]` ids → `[batch, seq, hidden]`.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed_tokens.forward(input_ids)
    }
}

#[derive(Module, Debug)]
pub struct HeadStage<B: Backend> {
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
}

impl<B: Backend> HeadStage<B> {
    pub fn new(cfg: &QuarkConfig, device: &B::Device) -> Self {
        Self {
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, device),
            lm_head: LinearConfig::new(cfg.hidden_size, cfg.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }

    /// `[batch, seq, hidden]` → logits `[batch, seq, vocab]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.lm_head.forward(self.norm.forward(x))
    }
}

/// Additive causal mask `[1, 1, seq, seq]`: 0 where a position may attend,
/// -inf for future positions.
pub fn causal_mask<B: Backend>(seq: usize, device: &B::Device) -> Tensor<B, 4> {
    let data: Vec<f32> = (0..seq)
        .flat_map(|i| (0..seq).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY }))
        .collect();
    Tensor::<B, 1>::from_data(TensorData::new(data, vec![seq * seq]), device)
        .reshape([1, 1, seq, seq])
}
