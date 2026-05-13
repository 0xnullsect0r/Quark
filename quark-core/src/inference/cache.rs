#![allow(dead_code, unused_imports, unused_variables)]

use burn::tensor::{backend::Backend, Tensor};

/// Key-Value cache for a single attention layer.
///
/// Grows dynamically as tokens are generated.  Once the accumulated sequence
/// length exceeds `max_seq_len` the oldest tokens are silently dropped
/// (sliding-window behaviour).
pub struct KvCache<B: Backend> {
    /// Cached keys: `[batch, num_kv_heads, seq_so_far, head_dim]`
    pub key: Option<Tensor<B, 4>>,
    /// Cached values: `[batch, num_kv_heads, seq_so_far, head_dim]`
    pub value: Option<Tensor<B, 4>>,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> KvCache<B> {
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            key: None,
            value: None,
            max_seq_len,
            num_kv_heads,
            head_dim,
        }
    }

    /// Append `new_k` / `new_v` to the cache and return the full accumulated
    /// `(K, V)` pair.
    ///
    /// `new_k` and `new_v` must have shape `[batch, num_kv_heads, new_seq, head_dim]`.
    pub fn update(
        &mut self,
        new_k: Tensor<B, 4>,
        new_v: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (k, v) = match (self.key.take(), self.value.take()) {
            (Some(cached_k), Some(cached_v)) => {
                let k = Tensor::cat(vec![cached_k, new_k], 2);
                let v = Tensor::cat(vec![cached_v, new_v], 2);
                (k, v)
            }
            _ => (new_k, new_v),
        };

        // Sliding-window truncation: keep only the most recent tokens.
        let seq_len = k.dims()[2];
        let (k, v) = if seq_len > self.max_seq_len {
            let start = seq_len - self.max_seq_len;
            (
                k.narrow(2, start, self.max_seq_len),
                v.narrow(2, start, self.max_seq_len),
            )
        } else {
            (k, v)
        };

        self.key = Some(k.clone());
        self.value = Some(v.clone());
        (k, v)
    }

    /// Reset the cache (e.g. between independent generation requests).
    pub fn clear(&mut self) {
        self.key = None;
        self.value = None;
    }

    /// Number of tokens currently cached.
    pub fn seq_len(&self) -> usize {
        self.key.as_ref().map(|k| k.dims()[2]).unwrap_or(0)
    }
}
