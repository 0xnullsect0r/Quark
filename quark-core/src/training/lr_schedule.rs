use serde::{Deserialize, Serialize};

/// Cosine annealing with linear warm-up.
///
/// During `warmup_steps` the learning rate rises linearly from 0 to `max_lr`.
/// Afterwards it follows a cosine decay curve down to `min_lr` by `max_steps`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosineSchedule {
    /// Number of steps over which to linearly warm up the learning rate.
    pub warmup_steps: u64,
    /// Total number of training steps (learning rate reaches `min_lr` at this point).
    pub max_steps: u64,
    /// Peak learning rate (reached at end of warmup).
    pub max_lr: f64,
    /// Floor learning rate (maintained after `max_steps`).
    pub min_lr: f64,
}

impl Default for CosineSchedule {
    fn default() -> Self {
        Self {
            warmup_steps: 100,
            max_steps: 10_000,
            max_lr: 3e-4,
            min_lr: 3e-5,
        }
    }
}

impl CosineSchedule {
    /// Returns the learning rate for the given training `step`.
    pub fn get_lr(&self, step: u64) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup from 0 → max_lr
            self.max_lr * (step as f64 / self.warmup_steps.max(1) as f64)
        } else if step >= self.max_steps {
            self.min_lr
        } else {
            // Cosine annealing from max_lr → min_lr
            let progress = (step - self.warmup_steps) as f64
                / (self.max_steps - self.warmup_steps).max(1) as f64;
            let cosine = (std::f64::consts::PI * progress).cos();
            self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + cosine)
        }
    }
}
