#![allow(dead_code, unused_imports, unused_variables)]

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, TensorData, Tensor},
};

use super::config::QuarkConfig;

/// Precompute RoPE cos/sin frequency tables.
///
/// Returns `(cos, sin)` each of shape `[max_seq_len, head_dim/2]`.
pub fn precompute_rope_freqs<B: Backend>(
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let half_dim = head_dim / 2;

    // inv_freq[i] = 1 / theta^(2i / head_dim)
    let inv_freq_data: Vec<f32> = (0..half_dim)
        .map(|i| 1.0f32 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq =
        Tensor::<B, 1>::from_data(TensorData::new(inv_freq_data, vec![half_dim]), device)
            .reshape([1, half_dim]); // [1, half_dim]

    let positions_data: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
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

/// Grouped-Query Attention with Rotary Position Embedding.
#[derive(Module, Debug)]
pub struct GroupedQueryAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

impl<B: Backend> GroupedQueryAttention<B> {
    pub fn new(cfg: &QuarkConfig, device: &B::Device) -> Self {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Self {
            q_proj: LinearConfig::new(cfg.hidden_size, cfg.num_attention_heads * head_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(cfg.hidden_size, cfg.num_key_value_heads * head_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(cfg.hidden_size, cfg.num_key_value_heads * head_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(cfg.num_attention_heads * head_dim, cfg.hidden_size)
                .with_bias(false)
                .init(device),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            rope_theta: cfg.rope_theta,
        }
    }

    /// Forward pass.
    ///
    /// - `x` shape: `[batch, seq, hidden]`
    /// - `mask` shape (optional): `[1, 1, seq, seq]` — additive causal mask (-inf / 0)
    /// - output shape: `[batch, seq, hidden]`
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        let device = x.device();
        let [batch, seq, _hidden] = x.dims();

        // Linear projections
        let q = self.q_proj.forward(x.clone()); // [batch, seq, num_heads * head_dim]
        let k = self.k_proj.forward(x.clone()); // [batch, seq, num_kv_heads * head_dim]
        let v = self.v_proj.forward(x);         // [batch, seq, num_kv_heads * head_dim]

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
        let (cos, sin) = precompute_rope_freqs::<B>(self.head_dim, seq, self.rope_theta, &device);
        let (q, k) = apply_rope(q, k, cos, sin);

        // Expand KV heads for GQA (repeat each KV head num_groups times)
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let groups = self.num_heads / self.num_kv_heads;
            let k = expand_kv(k, self.num_kv_heads, groups, batch, seq, self.head_dim);
            let v = expand_kv(v, self.num_kv_heads, groups, batch, seq, self.head_dim);
            (k, v)
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        // k^T: [batch, heads, head_dim, seq]
        let scores = q.matmul(k.permute([0, 1, 3, 2])).mul_scalar(scale);

        // Apply additive causal mask
        let scores = match mask {
            Some(m) => scores + m,
            None => scores,
        };

        let weights = softmax(scores, 3); // [batch, heads, seq, seq]
        let ctx = weights.matmul(v);      // [batch, heads, seq, head_dim]

        // Permute back and reshape to [batch, seq, hidden]
        let ctx = ctx
            .permute([0, 2, 1, 3])
            .reshape([batch, seq, self.num_heads * self.head_dim]);

        self.o_proj.forward(ctx)
    }
}
