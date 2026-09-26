//! What a training run will need: in memory or offloaded, and how much device
//! memory and disk. Shown in the GUI before starting and logged by the trainer.

use std::path::{Path, PathBuf};

use crate::memory::budget::HardwareBudget;
use crate::model::config::QuarkConfig;
use crate::training::streamed::offload_dir;
use crate::training::trainer::{estimate_memory, MemoryEstimate, OffloadMode, TrainerConfig};

/// Sharded checkpoints a streamed run keeps (see `streamed::run_streamed`).
const KEPT_CHECKPOINTS: u64 = 2;

#[derive(Debug, Clone)]
pub struct OffloadPlan {
    /// Whether the run will be streamed layer by layer.
    pub streamed: bool,
    /// In-memory estimate (what `Auto` decides on).
    pub in_memory: MemoryEstimate,
    /// Peak compute-device memory while streaming (one stage at a time).
    pub device_bytes: u64,
    pub weights_bytes: u64,
    pub optimizer_bytes: u64,
    pub activation_bytes: u64,
    pub checkpoint_bytes: u64,
    /// Total offload + checkpoint disk space needed when streaming.
    pub disk_needed: u64,
    /// Free space on the drive holding the offload directory, if known.
    pub disk_free: Option<u64>,
    pub offload_dir: PathBuf,
}

impl OffloadPlan {
    pub fn disk_fits(&self) -> bool {
        self.disk_free.is_none_or(|free| self.disk_needed <= free)
    }
}

/// Plan a run of `model` with `config` on this machine.
pub fn offload_plan(model: &QuarkConfig, config: &TrainerConfig, budget: &HardwareBudget) -> OffloadPlan {
    let in_memory = estimate_memory(model, config, budget);
    let streamed = match config.offload {
        OffloadMode::On => true,
        OffloadMode::Off => false,
        OffloadMode::Auto => !in_memory.fits(),
    };

    let total = model.param_count();
    let h = model.hidden_size as u64;
    let vocab = model.vocab_size as u64;
    let layer_params = (total - 2 * vocab * h - h) / model.num_hidden_layers.max(1) as u64;
    let largest_stage = layer_params.max(vocab * h);
    // Weights + accumulated grads + per-micro-batch grads (f32), plus one layer's
    // activations for a micro-batch.
    let (batch, seq) = (config.batch_size.max(1) as u64, model.max_position_embeddings as u64);
    let k = if model.num_experts > 0 { model.num_experts_per_tok as u64 } else { 1 };
    let layer_acts = batch * seq * (16 * h + 3 * model.intermediate_size as u64 * k)
        + batch * model.num_attention_heads as u64 * seq * seq * 3;
    let head_chunk = 64u64 << 20; // logits per head chunk (see streamed.rs)
    let device_bytes = 12 * largest_stage + 4 * layer_acts.max(head_chunk * 2);

    let weights_bytes = 4 * total;
    let optimizer_bytes = (config.optimizer.state_bytes_per_param() * total as f64) as u64;
    let micro = config.grad_accum_steps.max(1) as u64;
    let activation_bytes = (model.num_hidden_layers as u64 + 1) * micro * batch * seq * h * 4;
    let checkpoint_bytes = KEPT_CHECKPOINTS * (weights_bytes + optimizer_bytes);
    let disk_needed = weights_bytes + optimizer_bytes + activation_bytes + checkpoint_bytes;

    let dir = offload_dir(config);
    OffloadPlan {
        streamed,
        in_memory,
        device_bytes,
        weights_bytes,
        optimizer_bytes,
        activation_bytes,
        checkpoint_bytes,
        disk_needed,
        disk_free: free_space(&dir),
        offload_dir: dir,
    }
}

/// Free bytes on the disk that holds `path` (or its nearest existing parent).
pub fn free_space(path: &Path) -> Option<u64> {
    let mut existing = path.to_path_buf();
    while !existing.exists() {
        existing = existing.parent()?.to_path_buf();
    }
    let existing = existing.canonicalize().ok()?;
    let disks = sysinfo::Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .filter(|d| existing.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::optim::OptimizerKind;

    #[test]
    fn ten_b_plan_fits_a_good_pc() {
        let model = QuarkConfig::quark_10b_a2b();
        let config = TrainerConfig {
            batch_size: 1,
            grad_accum_steps: 8,
            optimizer: OptimizerKind::AdamWCompact,
            ..TrainerConfig::default()
        };
        let budget = HardwareBudget::detect();
        let plan = offload_plan(&model, &config, &budget);
        let gb = |b: u64| b as f64 / 1e9;
        // Too big for memory on any consumer machine, so Auto streams it…
        assert!(plan.in_memory.needed_bytes > 100_000_000_000, "{}", gb(plan.in_memory.needed_bytes));
        assert!(plan.streamed);
        // …with one layer at a time fitting an 8 GB GPU.
        assert!(plan.device_bytes < 8_000_000_000, "device {:.1} GB", gb(plan.device_bytes));
        // f32 master weights + ~3 B/param optimizer state.
        assert!((39.0..42.0).contains(&gb(plan.weights_bytes)), "{}", gb(plan.weights_bytes));
        assert!(gb(plan.optimizer_bytes) < 32.0);
        assert!(gb(plan.disk_needed) < 260.0, "disk {:.0} GB", gb(plan.disk_needed));
    }
}
