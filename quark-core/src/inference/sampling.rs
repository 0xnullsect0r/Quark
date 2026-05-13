#![allow(dead_code, unused_imports, unused_variables)]

use rand::Rng;

/// Parameters that control token sampling during inference.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_new_tokens: usize,
    pub stop_tokens: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            max_new_tokens: 256,
            stop_tokens: vec![],
        }
    }
}

// ── Sampling utilities ────────────────────────────────────────────────────────

/// Apply temperature scaling to logits in-place.
/// A temperature of 0.0 or 1.0 is a no-op (greedy / unchanged).
pub fn apply_temperature(logits: &mut Vec<f32>, temperature: f32) {
    if temperature <= 0.0 || (temperature - 1.0).abs() < 1e-6 {
        return;
    }
    for l in logits.iter_mut() {
        *l /= temperature;
    }
}

/// Greedy decoding: return the argmax token id.
pub fn sample_greedy(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Top-K sampling: keep only the top-k logits, then sample proportionally.
pub fn sample_top_k(logits: &[f32], k: usize, rng: &mut impl Rng) -> u32 {
    if k == 0 || k >= logits.len() {
        return sample_from_probs(&softmax_vec(logits), rng);
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut masked = vec![f32::NEG_INFINITY; logits.len()];
    for (idx, val) in indexed.iter().take(k) {
        masked[*idx] = *val;
    }
    sample_from_probs(&softmax_vec(&masked), rng)
}

/// Top-P (nucleus) sampling.
pub fn sample_top_p(logits: &[f32], p: f32, rng: &mut impl Rng) -> u32 {
    // Sort indices by descending probability.
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Compute softmax over sorted logits.
    let sorted_logits: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
    let sorted_probs = softmax_vec(&sorted_logits);

    // Build the nucleus (cumulative probability >= p).
    let mut cumsum = 0.0f32;
    let mut nucleus: Vec<(usize, f32)> = Vec::new();
    for ((idx, _), prob) in indexed.iter().zip(sorted_probs.iter()) {
        cumsum += prob;
        nucleus.push((*idx, *prob));
        if cumsum >= p {
            break;
        }
    }

    // Sample uniformly from the nucleus (re-normalised by their sum).
    let total: f32 = nucleus.iter().map(|(_, prob)| prob).sum();
    let target = rng.gen::<f32>() * total;
    let mut acc = 0.0f32;
    for (idx, prob) in &nucleus {
        acc += prob;
        if acc >= target {
            return *idx as u32;
        }
    }
    nucleus.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

// ── Internal helpers ─────────────────────────────────────────────────────────

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum::<f32>().max(1e-10);
    exp.iter().map(|x| x / sum).collect()
}

fn sample_from_probs(probs: &[f32], rng: &mut impl Rng) -> u32 {
    let target: f32 = rng.gen();
    let mut acc = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if acc >= target {
            return i as u32;
        }
    }
    (probs.len().saturating_sub(1)) as u32
}

// ── SamplingParams::sample ────────────────────────────────────────────────────

impl SamplingParams {
    /// Apply temperature then dispatch to the configured sampling strategy.
    pub fn sample(&self, logits: &mut Vec<f32>, rng: &mut impl Rng) -> u32 {
        apply_temperature(logits, self.temperature);

        if self.temperature == 0.0 || self.top_k == 1 {
            sample_greedy(logits)
        } else if self.top_p < 1.0 {
            sample_top_p(logits, self.top_p, rng)
        } else {
            sample_top_k(logits, self.top_k, rng)
        }
    }
}
