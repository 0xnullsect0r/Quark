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
        Self::quark_1b()
    }
}
