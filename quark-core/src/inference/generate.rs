#![allow(dead_code, unused_imports, unused_variables)]

use std::sync::mpsc as std_mpsc;

use anyhow::Result;
use burn::tensor::{backend::Backend, Int, Tensor};
use rand::SeedableRng;

use super::sampling::SamplingParams;
use crate::tokenizer::bpe::EOS_ID;

/// Configuration for a single generation call.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub prompt_ids: Vec<u32>,
    pub sampling: SamplingParams,
    /// RNG seed for reproducible sampling.
    pub seed: u64,
    /// Only the last `max_context` tokens are fed to the model (the model's
    /// `max_position_embeddings`). `0` means unlimited.
    pub max_context: usize,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            prompt_ids: vec![],
            sampling: SamplingParams::default(),
            seed: 42,
            max_context: 0,
        }
    }
}

// ── Shared helper ─────────────────────────────────────────────────────────────

fn id_tensor<B: Backend>(ids: &[u32], device: &B::Device) -> Tensor<B, 2, Int> {
    let int_ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    Tensor::from_data(
        burn::tensor::TensorData::new(int_ids, [1, ids.len()]),
        device,
    )
}

/// Logit vector for the last position of `[1, seq, vocab]` logits.
fn last_position<B: Backend>(logits: Tensor<B, 3>) -> Option<Vec<f32>> {
    let seq_len = logits.dims()[1];
    let last: Tensor<B, 2> = logits.narrow(1, seq_len - 1, 1).squeeze_dim::<2>(1);
    last.into_data().into_vec::<f32>().ok()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Generate tokens autoregressively, calling `on_token` with each new token.
///
/// Generation stops at a stop token, after `max_new_tokens`, or as soon as
/// `on_token` returns `false`. Returns the full token sequence
/// (prompt + newly generated tokens).
pub fn generate_with<B: Backend>(
    model: &crate::model::QuarkModel<B>,
    config: GenerateConfig,
    device: &B::Device,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<Vec<u32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let mut generated = config.prompt_ids.clone();
    if generated.is_empty() {
        return Ok(generated);
    }

    // Tokens `window_start..processed` of `generated` are in the KV caches,
    // at positions `0..processed - window_start`.
    let mut caches = model.new_kv_caches();
    let mut window_start = 0usize;
    let mut processed = 0usize;

    for _ in 0..config.sampling.max_new_tokens {
        // When the context is full, re-prefill a shorter window so the
        // caches aren't rebuilt on every subsequent token.
        if config.max_context > 0 && generated.len() - window_start > config.max_context {
            let keep = if processed == window_start {
                config.max_context
            } else {
                (config.max_context / 2).max(1)
            };
            window_start = generated.len() - keep;
            processed = window_start;
            caches = model.new_kv_caches();
        }

        let new_ids = id_tensor::<B>(&generated[processed..], device);
        let logits = model.forward_cached(new_ids, &mut caches, processed - window_start);
        processed = generated.len();
        let mut logits = match last_position(logits) {
            Some(l) => l,
            None => break,
        };

        let next_token = config.sampling.sample(&mut logits, &mut rng);
        generated.push(next_token);

        let keep_going = on_token(next_token);
        if !keep_going || config.sampling.stop_tokens.contains(&next_token) || next_token == EOS_ID
        {
            break;
        }
    }

    Ok(generated)
}

/// Generate tokens autoregressively.
///
/// Returns the full token sequence (prompt + newly generated tokens).
pub fn generate<B: Backend>(
    model: &crate::model::QuarkModel<B>,
    config: GenerateConfig,
    device: &B::Device,
) -> Result<Vec<u32>> {
    generate_with(model, config, device, |_| true)
}

/// Streaming variant: sends each newly generated token through `token_sender`
/// before checking the stop condition. Stops early if the receiver is dropped.
///
/// Returns the full token sequence (prompt + generated).
pub fn generate_streaming<B: Backend>(
    model: &crate::model::QuarkModel<B>,
    config: GenerateConfig,
    device: &B::Device,
    token_sender: std_mpsc::Sender<u32>,
) -> Result<Vec<u32>> {
    generate_with(model, config, device, |tok| token_sender.send(tok).is_ok())
}

#[cfg(test)]
mod tests {
    use crate::backend::InferBackend as TestBackend;

    use super::*;
    use crate::model::{config::QuarkConfig, QuarkModel};

    #[test]
    fn generates_past_the_context_window() {
        let cfg = QuarkConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 64,
            max_position_embeddings: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            num_experts: 2,
            num_experts_per_tok: 1,
            num_moe_layers: 1,
            moe_layer_freq: 2,
            tie_word_embeddings: false,
        };
        let device = Default::default();
        let model = QuarkModel::<TestBackend>::new(&cfg, &device);
        let config = GenerateConfig {
            prompt_ids: (10..22).collect(), // longer than the window
            sampling: SamplingParams {
                max_new_tokens: 20,
                stop_tokens: vec![],
                ..SamplingParams::default()
            },
            seed: 1,
            max_context: cfg.max_position_embeddings,
        };
        let mut streamed = 0;
        let out = generate_with(&model, config, &device, |_| {
            streamed += 1;
            true
        })
        .unwrap();
        assert_eq!(out.len() - 12, streamed);
        assert!(out.iter().all(|&t| (t as usize) < cfg.vocab_size));
    }
}
