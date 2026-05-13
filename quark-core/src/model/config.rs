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
}

impl Default for QuarkConfig {
    fn default() -> Self {
        Self::quark_1b()
    }
}
