#![allow(dead_code, unused_imports, unused_variables)]

use burn::{
    module::Module,
    tensor::{
        activation::softmax,
        backend::Backend,
        module::attention,
        ops::AttentionModuleOptions,
        Tensor, TensorData,
    },
};

use super::{config::QuarkConfig, proj::Proj};
use crate::inference::cache::KvCache;

/// Precompute RoPE cos/sin frequency tables.
///
/// Returns `(cos, sin)` each of shape `[max_seq_len, head_dim/2]`.
pub fn precompute_rope_freqs<B: Backend>(
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    rope_freqs_at::<B>(head_dim, 0, max_seq_len, theta, device)
}

/// RoPE cos/sin tables for positions `start_pos..start_pos + len`.
///
/// Returns `(cos, sin)` each of shape `[len, head_dim/2]`.
pub fn rope_freqs_at<B: Backend>(
    head_dim: usize,
    start_pos: usize,
    len: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let max_seq_len = len;
    let half_dim = head_dim / 2;

    // inv_freq[i] = 1 / theta^(2i / head_dim)
    let inv_freq_data: Vec<f32> = (0..half_dim)
        .map(|i| 1.0f32 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq =
        Tensor::<B, 1>::from_data(TensorData::new(inv_freq_data, vec![half_dim]), device)
            .reshape([1, half_dim]); // [1, half_dim]

    let positions_data: Vec<f32> = (start_pos..start_pos + max_seq_len)
        .map(|i| i as f32)
        .collect();
    let positions =
        Tensor::<B, 1>::from_data(TensorData::new(positions_data, vec![max_seq_len]), device)
            .reshape([max_seq_len, 1]); // [max_seq_len, 1]

    // Outer product: [max_seq_len, half_dim]
    let freqs = positions.matmul(inv_freq);

    let cos = freqs.clone().cos();
    let sin = freqs.sin();
    (cos, sin)
}

/// Apply RoPE rotation to query and key tensors.
///
/// - `q`, `k` shape: `[batch, heads, seq, head_dim]`
/// - `cos`, `sin` shape: `[seq, head_dim/2]`
pub fn apply_rope<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, heads, seq, head_dim] = q.dims();
    let [kbatch, kheads, kseq, _] = k.dims();
    let half = head_dim / 2;

    // Split along last dim into two halves
    let q1 = q.clone().slice([0..batch, 0..heads, 0..seq, 0..half]);
    let q2 = q.slice([0..batch, 0..heads, 0..seq, half..head_dim]);
    let k1 = k.clone().slice([0..kbatch, 0..kheads, 0..kseq, 0..half]);
    let k2 = k.slice([0..kbatch, 0..kheads, 0..kseq, half..head_dim]);

    // cos/sin: [seq, half] -> [1, 1, seq, half] for broadcasting
    let cos_r = cos.reshape([1, 1, seq, half]);
    let sin_r = sin.reshape([1, 1, seq, half]);

    // Rotation: (x1*cos - x2*sin, x1*sin + x2*cos)
    let q_out = Tensor::cat(
        vec![
            q1.clone() * cos_r.clone() - q2.clone() * sin_r.clone(),
            q1 * sin_r.clone() + q2 * cos_r.clone(),
        ],
        3,
    );
    let k_out = Tensor::cat(
        vec![
            k1.clone() * cos_r.clone() - k2.clone() * sin_r.clone(),
            k1 * sin_r + k2 * cos_r,
        ],
        3,
    );
    (q_out, k_out)
}

/// Expand KV heads from `kv_heads` to `kv_heads * groups` by repeating each head.
fn expand_kv<B: Backend>(
    t: Tensor<B, 4>,
    kv_heads: usize,
    groups: usize,
    batch: usize,
    seq: usize,
    head_dim: usize,
) -> Tensor<B, 4> {
    let mut heads = Vec::with_capacity(kv_heads * groups);
    for h in 0..kv_heads {
        let head = t.clone().slice([0..batch, h..h + 1, 0..seq, 0..head_dim]);
        for _ in 0..groups {
            heads.push(head.clone());
        }
    }
    Tensor::cat(heads, 1)
}

/// How attention scores are masked.
enum Scores<B: Backend> {
    /// The backend's attention op, optionally causal (query `i` sees keys `..=i`).
    Fused { causal: bool },
    /// Explicit additive mask `[1, 1, seq, kv_seq]` (0 / -inf).
    Masked(Tensor<B, 4>),
}

/// Grouped-Query Attention with Rotary Position Embedding.
#[derive(Module, Debug)]
pub struct GroupedQueryAttention<B: Backend> {
    q_proj: Proj<B>,
    k_proj: Proj<B>,
    v_proj: Proj<B>,
    o_proj: Proj<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

impl<B: Backend> GroupedQueryAttention<B> {
    pub fn new(cfg: &QuarkConfig, device: &B::Device) -> Self {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Self {
            q_proj: Proj::new(cfg.hidden_size, cfg.num_attention_heads * head_dim, device),
            k_proj: Proj::new(cfg.hidden_size, cfg.num_key_value_heads * head_dim, device),
            v_proj: Proj::new(cfg.hidden_size, cfg.num_key_value_heads * head_dim, device),
            o_proj: Proj::new(cfg.num_attention_heads * head_dim, cfg.hidden_size, device),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            rope_theta: cfg.rope_theta,
        }
    }

    /// Forward pass.
    ///
    /// - `x` shape: `[batch, seq, hidden]`
    /// - `causal`: each position attends only to itself and earlier ones
    /// - output shape: `[batch, seq, hidden]`
    pub fn forward(&self, x: Tensor<B, 3>, causal: bool) -> Tensor<B, 3> {
        let (q, k, v) = self.project(x, 0);
        self.attend(q, k, v, Scores::Fused { causal })
    }

    /// Incremental forward pass for generation.
    ///
    /// `x` holds only the new tokens, at positions `start_pos..start_pos + seq`;
    /// keys/values for earlier positions come from (and are appended to) `cache`.
    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        cache: &mut KvCache<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let device = x.device();
        let seq = x.dims()[1];
        let (q, k, v) = self.project(x, start_pos);
        let (k, v) = cache.update(k, v);
        let total = k.dims()[2];

        // A single new token may attend to everything, and a first chunk is
        // plain causal attention; a later multi-token chunk needs a causal
        // mask offset by the cached length.
        let past = total - seq;
        let scores = if seq == 1 || past == 0 {
            Scores::Fused { causal: seq > 1 }
        } else {
            let data: Vec<f32> = (0..seq)
                .flat_map(|i| {
                    (0..total).map(move |j| if j <= past + i { 0.0 } else { f32::NEG_INFINITY })
                })
                .collect();
            let mask = Tensor::<B, 1>::from_data(TensorData::new(data, vec![seq * total]), &device)
                .reshape([1, 1, seq, total]);
            Scores::Masked(mask)
        };
        self.attend(q, k, v, scores)
    }

    /// The projections, by path relative to this module.
    pub fn projs_mut(&mut self, prefix: &str) -> Vec<(String, &mut Proj<B>)> {
        vec![
            (format!("{prefix}q_proj"), &mut self.q_proj),
            (format!("{prefix}k_proj"), &mut self.k_proj),
            (format!("{prefix}v_proj"), &mut self.v_proj),
            (format!("{prefix}o_proj"), &mut self.o_proj),
        ]
    }

    /// An empty KV cache sized for this layer.
    pub fn new_cache(&self) -> KvCache<B> {
        KvCache::new(usize::MAX, self.num_kv_heads, self.head_dim)
    }

    /// Project `x` to RoPE-rotated `q` `[batch, heads, seq, head_dim]` and
    /// `k`, `v` `[batch, kv_heads, seq, head_dim]`.
    fn project(
        &self,
        x: Tensor<B, 3>,
        start_pos: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let device = x.device();
        let [batch, seq, _hidden] = x.dims();

        // Linear projections
        let q = self.q_proj.forward(x.clone()); // [batch, seq, num_heads * head_dim]
        let k = self.k_proj.forward(x.clone()); // [batch, seq, num_kv_heads * head_dim]
        let v = self.v_proj.forward(x); // [batch, seq, num_kv_heads * head_dim]

        // Reshape -> [batch, seq, heads, head_dim], permute -> [batch, heads, seq, head_dim]
        let q = q
            .reshape([batch, seq, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let k = k
            .reshape([batch, seq, self.num_kv_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let v = v
            .reshape([batch, seq, self.num_kv_heads, self.head_dim])
            .permute([0, 2, 1, 3]);

        // Apply RoPE
        let (cos, sin) =
            rope_freqs_at::<B>(self.head_dim, start_pos, seq, self.rope_theta, &device);
        let (q, k) = apply_rope(q, k, cos, sin);
        (q, k, v)
    }

    /// Scaled dot-product attention of `q` over `k`/`v` (which may be longer
    /// than `q` when cached), followed by the output projection.
    fn attend(&self, q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>, scores: Scores<B>) -> Tensor<B, 3> {
        let [batch, _, seq, _] = q.dims();
        let kv_seq = k.dims()[2];

        // Expand KV heads for GQA (repeat each KV head num_groups times)
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let groups = self.num_heads / self.num_kv_heads;
            let k = expand_kv(k, self.num_kv_heads, groups, batch, kv_seq, self.head_dim);
            let v = expand_kv(v, self.num_kv_heads, groups, batch, kv_seq, self.head_dim);
            (k, v)
        } else {
            (k, v)
        };

        let ctx = match scores {
            // Backend attention kernel (flash attention on GPU backends).
            Scores::Fused { causal } => {
                let options = AttentionModuleOptions { scale: None, softcap: None, is_causal: causal };
                attention(q, k, v, None, None, options)
            }
            Scores::Masked(mask) => {
                let scale = 1.0f32 / (self.head_dim as f32).sqrt();
                let scores = q.matmul(k.permute([0, 1, 3, 2])).mul_scalar(scale) + mask;
                softmax(scores, 3).matmul(v) // [batch, heads, seq, head_dim]
            }
        };

        // Permute back and reshape to [batch, seq, hidden]
        let ctx = ctx
            .permute([0, 2, 1, 3])
            .reshape([batch, seq, self.num_heads * self.head_dim]);

        self.o_proj.forward(ctx)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::InferBackend as TestBackend;

    use super::*;

    type B = TestBackend;

    #[test]
    fn expand_kv_repeats_each_head() {
        let device = Default::default();
        // [batch=1, kv_heads=2, seq=1, head_dim=2]: head 0 = [1, 2], head 1 = [3, 4]
        let t = Tensor::<B, 4>::from_data(
            TensorData::new(vec![1.0f32, 2., 3., 4.], [1, 2, 1, 2]),
            &device,
        );
        let out = expand_kv(t, 2, 3, 1, 1, 2);
        assert_eq!(out.dims(), [1, 6, 1, 2]);
        let v: Vec<f32> = out.into_data().into_vec().unwrap();
        assert_eq!(v, vec![1., 2., 1., 2., 1., 2., 3., 4., 3., 4., 3., 4.]);
    }

    #[test]
    fn rope_offset_matches_full_table() {
        let device = Default::default();
        let (cos_full, sin_full) = precompute_rope_freqs::<B>(8, 10, 10000.0, &device);
        let (cos, sin) = rope_freqs_at::<B>(8, 6, 4, 10000.0, &device);
        let a: Vec<f32> = cos_full.narrow(0, 6, 4).into_data().into_vec().unwrap();
        let b: Vec<f32> = cos.into_data().into_vec().unwrap();
        assert_eq!(a, b);
        let a: Vec<f32> = sin_full.narrow(0, 6, 4).into_data().into_vec().unwrap();
        let b: Vec<f32> = sin.into_data().into_vec().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn fused_causal_matches_masked() {
        let device = Default::default();
        let cfg = crate::model::config::QuarkConfig {
            hidden_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            ..crate::model::config::QuarkConfig::quark_tiny()
        };
        let attn = GroupedQueryAttention::<B>::new(&cfg, &device);
        let x = Tensor::<B, 3>::random([2, 7, 32], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let fused: Vec<f32> = attn.forward(x.clone(), true).into_data().to_vec().unwrap();
        let (q, k, v) = attn.project(x, 0);
        let mask = crate::model::stages::causal_mask::<B>(7, &device);
        let manual: Vec<f32> = attn.attend(q, k, v, Scores::Masked(mask)).into_data().to_vec().unwrap();
        for (a, b) in fused.iter().zip(&manual) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
