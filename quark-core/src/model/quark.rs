#![allow(dead_code, unused_imports, unused_variables)]

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Int, Tensor, TensorData},
};

use crate::inference::cache::KvCache;

use super::{block::DecoderBlock, config::QuarkConfig, norm::RmsNorm};

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
        self.forward_with_aux(input_ids).0
    }

    /// One empty KV cache per decoder layer, for [`Self::forward_cached`].
    pub fn new_kv_caches(&self) -> Vec<KvCache<B>> {
        self.layers.iter().map(|layer| layer.new_cache()).collect()
    }

    /// Incremental forward pass for generation.
    ///
    /// `input_ids` holds only the new tokens, at positions
    /// `start_pos..start_pos + seq`; earlier positions are read from `caches`
    /// (one per layer, see [`Self::new_kv_caches`]), which are then extended.
    /// Returns logits for the new tokens: `[batch, seq, vocab]`.
    pub fn forward_cached(
        &self,
        input_ids: Tensor<B, 2, Int>,
        caches: &mut [KvCache<B>],
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let mut x = self.embed_tokens.forward(input_ids);
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            x = layer.forward_cached(x, cache, start_pos);
        }
        self.lm_head.forward(self.norm.forward(x))
    }

    /// Forward pass that also returns the MoE load-balancing loss averaged
    /// over MoE layers (`None` if the model has no MoE layers).
    pub fn forward_with_aux(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 1>>) {
        let device = input_ids.device();
        let [batch, seq] = input_ids.dims();

        // Token embeddings: [batch, seq, hidden]
        let mut x = self.embed_tokens.forward(input_ids);

        // Build an additive causal mask [1, 1, seq, seq]:
        //   0.0   for positions that can attend (lower triangle + diagonal)
        //   -inf  for future positions (upper triangle)
        let mask_flat: Vec<f32> = (0..seq)
            .flat_map(|i| (0..seq).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY }))
            .collect();
        let mask: Tensor<B, 4> =
            Tensor::<B, 1>::from_data(TensorData::new(mask_flat, vec![seq * seq]), &device)
                .reshape([1_usize, 1, seq, seq]);

        // Forward through all decoder layers
        let mut aux_sum: Option<Tensor<B, 1>> = None;
        let mut moe_layers = 0usize;
        for layer in &self.layers {
            let (out, aux) = layer.forward_with_aux(x, Some(mask.clone()));
            x = out;
            if let Some(aux) = aux {
                moe_layers += 1;
                aux_sum = Some(match aux_sum {
                    Some(sum) => sum + aux,
                    None => aux,
                });
            }
        }
        let aux = aux_sum.map(|sum| sum / moe_layers as f32);

        // Final layer norm and language-model head
        x = self.norm.forward(x);
        (self.lm_head.forward(x), aux)
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

    fn ids(v: &[i32]) -> Tensor<NdArray<f32>, 2, Int> {
        Tensor::from_data(
            TensorData::new(v.to_vec(), [1, v.len()]),
            &Default::default(),
        )
    }

    fn values(t: Tensor<NdArray<f32>, 3>) -> Vec<f32> {
        t.into_data().into_vec().unwrap()
    }

    fn assert_close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn future_tokens_do_not_affect_past_logits() {
        let cfg = test_cfg();
        let model = QuarkModel::<NdArray<f32>>::new(&cfg, &Default::default());
        let a = values(model.forward(ids(&[5, 9, 17, 3, 42, 8])).narrow(1, 0, 3));
        let b = values(model.forward(ids(&[5, 9, 17, 200, 1, 77])).narrow(1, 0, 3));
        assert_close(&a, &b);
    }

    #[test]
    fn kv_cache_matches_full_forward() {
        let cfg = test_cfg();
        let model = QuarkModel::<NdArray<f32>>::new(&cfg, &Default::default());
        let tokens = [5, 9, 17, 3, 42, 8, 11];
        let full = values(model.forward(ids(&tokens)));

        // Prefill 4 tokens, then feed the remaining 3 one at a time.
        let mut caches = model.new_kv_caches();
        let mut cached = values(model.forward_cached(ids(&tokens[..4]), &mut caches, 0));
        for (pos, &tok) in tokens.iter().enumerate().skip(4) {
            cached.extend(values(model.forward_cached(ids(&[tok]), &mut caches, pos)));
        }
        assert_close(&full, &cached);
    }
}
