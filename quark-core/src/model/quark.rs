#![allow(dead_code, unused_imports, unused_variables)]

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Int, Tensor, TensorData},
};

use super::{
    block::DecoderBlock,
    config::QuarkConfig,
    norm::RmsNorm,
};

/// The full Quark transformer model.
///
/// Weight tying between `embed_tokens` and `lm_head` is a future optimisation;
/// for now a separate `lm_head` linear layer is always allocated.
#[derive(Module, Debug)]
pub struct QuarkModel<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<DecoderBlock<B>>,
    norm: RmsNorm<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> QuarkModel<B> {
    pub fn new(cfg: &QuarkConfig, device: &B::Device) -> Self {
        // Layer i is a MoE layer iff `i % moe_layer_freq == 0`
        let layers: Vec<DecoderBlock<B>> = (0..cfg.num_hidden_layers)
            .map(|i| {
                let is_moe = cfg.moe_layer_freq > 0 && i % cfg.moe_layer_freq == 0;
                DecoderBlock::new(cfg, is_moe, device)
            })
            .collect();

        Self {
            embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, device),
            lm_head: LinearConfig::new(cfg.hidden_size, cfg.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }

    /// Forward pass. Returns logits of shape `[batch, seq, vocab]`.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = input_ids.device();
        let [batch, seq] = input_ids.dims();

        // Token embeddings: [batch, seq, hidden]
        let mut x = self.embed_tokens.forward(input_ids);

        // Build an additive causal mask [1, 1, seq, seq]:
        //   0.0   for positions that can attend (lower triangle + diagonal)
        //   -inf  for future positions (upper triangle)
        let mask_flat: Vec<f32> = (0..seq)
            .flat_map(|i| {
                (0..seq).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY })
            })
            .collect();
        let mask: Tensor<B, 4> =
            Tensor::<B, 1>::from_data(TensorData::new(mask_flat, vec![seq * seq]), &device)
                .reshape([1_usize, 1, seq, seq]);

        // Forward through all decoder layers
        for layer in &self.layers {
            x = layer.forward(x, Some(mask.clone()));
        }

        // Final layer norm and language-model head
        x = self.norm.forward(x);
        self.lm_head.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    /// A tiny configuration suitable for unit-testing shapes without OOM.
    fn test_cfg() -> QuarkConfig {
        QuarkConfig {
            vocab_size: 256,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            num_experts: 2,
            num_experts_per_tok: 1,
            num_moe_layers: 1,
            moe_layer_freq: 2,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn test_forward_shapes() {
        type B = NdArray<f32>;
        let device = Default::default();
        let cfg = test_cfg();
        let model = QuarkModel::<B>::new(&cfg, &device);

        let batch = 2usize;
        let seq = 8usize;
        let ids = Tensor::<B, 2, Int>::zeros([batch, seq], &device);
        let logits = model.forward(ids);
        let [b, s, v] = logits.dims();
        assert_eq!(b, batch);
        assert_eq!(s, seq);
        assert_eq!(v, cfg.vocab_size);
    }
}
