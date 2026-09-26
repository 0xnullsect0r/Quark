use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarkConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_moe_layers: usize,
    pub moe_layer_freq: usize,
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelPreset {
    QuarkTiny,
    QuarkSmall,
    /// ~10B total / ~2B active per token: the largest preset meant for one
    /// machine (trained with offloading, run 4-bit quantized).
    Quark10BA2B,
    Quark1B,
    Quark3B,
    Quark7B,
    Quark20B,
    Quark30B,
    Quark48B,
    Quark74B,
    Quark120B,
    Quark249B,
    Quark300B,
    Quark400B,
    Custom,
}

impl QuarkConfig {
    /// Read the `config.json` the trainer writes next to its checkpoints.
    /// (A sharded checkpoint directory holds its own `config.json`.)
    pub fn for_checkpoint(checkpoint: &std::path::Path) -> Option<Self> {
        let dir = if checkpoint.is_dir() { checkpoint } else { checkpoint.parent()? };
        let txt = std::fs::read_to_string(dir.join("config.json")).ok()?;
        serde_json::from_str(&txt).ok()
    }

    /// ~12M parameters: trains on a laptop CPU in minutes to hours.
    pub fn quark_tiny() -> Self {
        Self {
            vocab_size: 8000,
            hidden_size: 256,
            num_hidden_layers: 6,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 704,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            num_experts: 4,
            num_experts_per_tok: 2,
            num_moe_layers: 2,
            moe_layer_freq: 3,
            tie_word_embeddings: false,
        }
    }

    /// ~220M parameters: realistic on a single consumer GPU.
    pub fn quark_small() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_key_value_heads: 4,
            intermediate_size: 2048,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            num_experts: 8,
            num_experts_per_tok: 2,
            num_moe_layers: 3,
            moe_layer_freq: 4,
            tie_word_embeddings: false,
        }
    }

    /// ~10.0B parameters, ~2.1B active per token (16 experts, top-2, MoE in
    /// every layer). Trains with offloading (layer by layer through RAM and
    /// disk) on one machine; runs at 4 bits in ≈ 6 GB.
    pub fn quark_10b_a2b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 3072,
            num_hidden_layers: 30,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            intermediate_size: 2048,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            num_experts: 16,
            num_experts_per_tok: 2,
            num_moe_layers: 30,
            moe_layer_freq: 1,
            tie_word_embeddings: false,
        }
    }

    /// Parameters used per token (MoE layers count only the routed experts).
    pub fn active_param_count(&self) -> u64 {
        let h = self.hidden_size as u64;
        let ffn = 3 * h * self.intermediate_size as u64;
        let moe_layers = (0..self.num_hidden_layers).filter(|&i| self.is_moe_layer(i)).count() as u64;
        let unused_experts = self.num_experts.saturating_sub(self.num_experts_per_tok) as u64;
        self.param_count() - moe_layers * unused_experts * ffn
    }

    pub fn from_preset(preset: ModelPreset) -> Option<Self> {
        Some(match preset {
            ModelPreset::QuarkTiny => Self::quark_tiny(),
            ModelPreset::QuarkSmall => Self::quark_small(),
            ModelPreset::Quark10BA2B => Self::quark_10b_a2b(),
            ModelPreset::Quark1B => Self::quark_1b(),
            ModelPreset::Quark3B => Self::quark_3b(),
            ModelPreset::Quark7B => Self::quark_7b(),
            ModelPreset::Quark20B => Self::quark_20b(),
            ModelPreset::Quark30B => Self::quark_30b(),
            ModelPreset::Quark48B => Self::quark_48b(),
            ModelPreset::Quark74B => Self::quark_74b(),
            ModelPreset::Quark120B => Self::quark_120b(),
            ModelPreset::Quark249B => Self::quark_249b(),
            ModelPreset::Quark300B => Self::quark_300b(),
            ModelPreset::Quark400B => Self::quark_400b(),
            ModelPreset::Custom => return None,
        })
    }

    /// Whether decoder layer `i` is a MoE layer (mirrors `QuarkModel::new`).
    pub fn is_moe_layer(&self, i: usize) -> bool {
        self.moe_layer_freq > 0 && i.is_multiple_of(self.moe_layer_freq)
    }

    /// Exact number of trainable parameters `QuarkModel::new` allocates.
    pub fn param_count(&self) -> u64 {
        let h = self.hidden_size as u64;
        let head_dim = (self.hidden_size / self.num_attention_heads.max(1)) as u64;
        let q = self.num_attention_heads as u64 * head_dim;
        let kv = self.num_key_value_heads as u64 * head_dim;
        let attn = h * q + 2 * h * kv + q * h;
        let ffn = 3 * h * self.intermediate_size as u64;
        let moe = h * self.num_experts as u64 + self.num_experts as u64 * ffn;
        let norms = 2 * h;

        let layers: u64 = (0..self.num_hidden_layers)
            .map(|i| attn + norms + if self.is_moe_layer(i) { moe } else { ffn })
            .sum();
        let vocab = self.vocab_size as u64;
        // token embedding + lm_head (always untied) + final norm
        vocab * h + layers + h * vocab + h
    }

    /// Rough peak memory for training, in bytes: weights, gradients (plus an
    /// accumulation copy) and AdamW moments, plus activations kept for the
    /// backward pass. `bytes_per_elem` is 4 for f32, 2 for bf16. Balanced
    /// gradient checkpointing roughly halves the activation term.
    pub fn training_memory_bytes(
        &self,
        batch: usize,
        seq: usize,
        bytes_per_elem: u64,
        checkpointing: bool,
    ) -> u64 {
        let state = self.param_count() * 5 * bytes_per_elem;

        let tokens = (batch * seq) as u64;
        let h = self.hidden_size as u64;
        let per_layer: u64 = (0..self.num_hidden_layers)
            .map(|i| {
                let experts = if self.is_moe_layer(i) { self.num_experts_per_tok } else { 1 };
                let linear = tokens * (16 * h + 3 * self.intermediate_size as u64 * experts as u64);
                let attn = (batch * self.num_attention_heads * seq * seq) as u64 * 3;
                linear + attn
            })
            .sum();
        let logits = tokens * self.vocab_size as u64 * 3;
        let activations = (per_layer + logits) * bytes_per_elem;
        state + if checkpointing { activations / 2 } else { activations }
    }

    pub fn quark_1b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            num_hidden_layers: 16,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            intermediate_size: 5632,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            num_experts: 8,
            num_experts_per_tok: 2,
            num_moe_layers: 4,
            moe_layer_freq: 4,
            tie_word_embeddings: true,
        }
    }

    pub fn quark_3b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            intermediate_size: 8192,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            num_experts: 8,
            num_experts_per_tok: 2,
            num_moe_layers: 6,
            moe_layer_freq: 4,
            tie_word_embeddings: true,
        }
    }
    pub fn quark_7b() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 14336,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            num_experts: 8,
            num_experts_per_tok: 2,
            num_moe_layers: 16,
            moe_layer_freq: 2,
            tie_word_embeddings: false,
        }
    }

    pub fn quark_20b() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 5120,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: 8,
            intermediate_size: 16384,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            num_experts: 8,
            num_experts_per_tok: 2,
            num_moe_layers: 20,
            moe_layer_freq: 2,
            tie_word_embeddings: false,
        }
    }

    pub fn quark_30b() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 6144,
            num_hidden_layers: 48,
            num_attention_heads: 48,
            num_key_value_heads: 8,
            intermediate_size: 16384,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            num_experts: 8,
            num_experts_per_tok: 2,
            num_moe_layers: 24,
            moe_layer_freq: 2,
            tie_word_embeddings: false,
        }
    }

    pub fn quark_48b() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 7168,
            num_hidden_layers: 56,
            num_attention_heads: 56,
            num_key_value_heads: 8,
            intermediate_size: 20480,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            num_experts: 16,
            num_experts_per_tok: 4,
            num_moe_layers: 28,
            moe_layer_freq: 2,
            tie_word_embeddings: false,
        }
    }

    pub fn quark_74b() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 8192,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            intermediate_size: 28672,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            num_experts: 16,
            num_experts_per_tok: 4,
            num_moe_layers: 40,
            moe_layer_freq: 2,
            tie_word_embeddings: false,
        }
    }

    pub fn quark_120b() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 10240,
            num_hidden_layers: 96,
            num_attention_heads: 80,
            num_key_value_heads: 8,
            intermediate_size: 36864,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            num_experts: 16,
            num_experts_per_tok: 4,
            num_moe_layers: 48,
            moe_layer_freq: 2,
            tie_word_embeddings: false,
        }
    }

    pub fn quark_249b() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 14336,
            num_hidden_layers: 96,
            num_attention_heads: 112,
            num_key_value_heads: 8,
            intermediate_size: 49152,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            num_experts: 32,
            num_experts_per_tok: 8,
            num_moe_layers: 48,
            moe_layer_freq: 2,
            tie_word_embeddings: false,
        }
    }

    pub fn quark_300b() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 14336,
            num_hidden_layers: 112,
            num_attention_heads: 112,
            num_key_value_heads: 8,
            intermediate_size: 57344,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            num_experts: 32,
            num_experts_per_tok: 8,
            num_moe_layers: 56,
            moe_layer_freq: 2,
            tie_word_embeddings: false,
        }
    }

    pub fn quark_400b() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 16384,
            num_hidden_layers: 128,
            num_attention_heads: 128,
            num_key_value_heads: 16,
            intermediate_size: 65536,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            num_experts: 64,
            num_experts_per_tok: 8,
            num_moe_layers: 64,
            moe_layer_freq: 2,
            tie_word_embeddings: false,
        }
    }
}

impl Default for QuarkConfig {
    fn default() -> Self {
        Self::quark_tiny()
    }
}
