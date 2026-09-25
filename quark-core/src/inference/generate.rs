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

/// Run one forward pass and return the last token's logit vector.
///
/// `ids` is the current sequence of token ids.  Returns `None` if the
/// sequence is empty or if tensor data extraction fails.
fn last_logits<B: Backend>(
    model: &crate::model::QuarkModel<B>,
    ids: &[u32],
    device: &B::Device,
) -> Option<Vec<f32>> {
    if ids.is_empty() {
        return None;
    }
    let int_ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    let seq = ids.len();

    let input = Tensor::<B, 2, Int>::from_data(
        burn::tensor::TensorData::new(int_ids, [1, seq]),
        device,
    );

    // Forward returns [batch=1, seq, vocab].
    let logits = model.forward(input);
    let seq_len = logits.dims()[1];

    // Extract [1, vocab] for the last position, then flatten to Vec<f32>.
    let last: Tensor<B, 2> = logits.narrow(1, seq_len - 1, 1).squeeze::<2>(1);
    let data = last.into_data().into_vec::<f32>().ok()?;
    Some(data)
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

    for _ in 0..config.sampling.max_new_tokens {
        let window_start = match config.max_context {
            0 => 0,
            n => generated.len().saturating_sub(n),
        };
        let mut logits = match last_logits(model, &generated[window_start..], device) {
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
