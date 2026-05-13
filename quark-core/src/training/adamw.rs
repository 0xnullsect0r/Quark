use serde::{Deserialize, Serialize};

/// AdamW optimiser configuration.
///
/// Holds all hyperparameters needed to configure the AdamW optimizer.
/// Call [`AdamWConfig::to_burn_config`] to obtain a burn-native config
/// that can be used to initialize an optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWConfig {
    /// Base learning rate (overridden per-step by the LR schedule).
    pub lr: f64,
    /// First moment decay (β₁).
    pub beta1: f64,
    /// Second moment decay (β₂).
    pub beta2: f64,
    /// Numerical stability epsilon.
    pub eps: f64,
    /// Decoupled weight-decay coefficient.
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
        }
    }
}

impl AdamWConfig {
    /// Convert to burn's [`burn::optim::AdamWConfig`].
    ///
    /// The resulting config can be used with `AdamWConfig::init()` to
    /// create an optimizer that operates on any autodiff-capable backend.
    pub fn to_burn_config(&self) -> burn::optim::AdamWConfig {
        burn::optim::AdamWConfig::new()
            .with_beta_1(self.beta1 as f32)
            .with_beta_2(self.beta2 as f32)
            .with_epsilon(self.eps as f32)
            .with_weight_decay(self.weight_decay as f32)
    }
}
