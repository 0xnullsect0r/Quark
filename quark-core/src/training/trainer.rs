use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use anyhow::Result;
use tokio::sync::mpsc;

use crate::data::batch::DataBatch;
use crate::memory::tier::TierConfig;
use crate::model::config::QuarkConfig;
use crate::training::adamw::AdamWConfig;
use crate::training::lr_schedule::CosineSchedule;
use crate::training::metrics::{MetricsReceiver, MetricsSender, TrainingMetrics};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level training configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainerConfig {
    /// Directory where checkpoints and logs are written.
    pub output_dir: PathBuf,
    /// Maximum number of optimiser steps to run.
    pub max_steps: u64,
    /// Number of samples per forward pass.
    pub batch_size: usize,
    /// Number of micro-batches to accumulate before an optimiser step.
    pub grad_accum_steps: usize,
    /// Save a checkpoint every N steps.
    pub save_every_steps: u64,
    /// Run evaluation every N steps.
    pub eval_every_steps: u64,
    /// Whether to use mixed-precision (fp16/bf16) training.
    pub mixed_precision: bool,
    /// Maximum global gradient norm before clipping.
    pub max_grad_norm: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// AdamW hyperparameters.
    pub adamw: AdamWConfig,
    /// Learning-rate schedule.
    pub schedule: CosineSchedule,
    /// Memory-tier resource limits.
    pub tier: TierConfig,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("runs/default"),
            max_steps: 10_000,
            batch_size: 4,
            grad_accum_steps: 8,
            save_every_steps: 500,
            eval_every_steps: 100,
            mixed_precision: true,
            max_grad_norm: 1.0,
            seed: 42,
            adamw: AdamWConfig::default(),
            schedule: CosineSchedule::default(),
            tier: TierConfig::default(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle & legacy wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// A handle to a running training job that allows external control.
pub struct TrainingHandle {
    /// Clone this sender to inject out-of-band metrics if needed.
    pub sender: MetricsSender,
    /// Set to `true` to request a graceful stop.
    pub stop_flag: Arc<AtomicBool>,
}

impl TrainingHandle {
    /// Signal the training loop to stop after the current step.
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }
}

/// Orchestrates the training loop and streams metrics to the GUI / CLI.
///
/// This is the legacy wrapper retained for API compatibility.  New code should
/// prefer [`start_training`] which returns a [`TrainingHandle`] directly.
pub struct Trainer {
    config: TrainerConfig,
    sender: MetricsSender,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> (Self, MetricsReceiver) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (Self { config, sender: tx }, rx)
    }

    pub async fn run(&self) -> anyhow::Result<()> {
        todo!("Trainer::run — use start_training() for the async loop")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Spawn the training loop as a background [`tokio`] task.
///
/// Returns a [`TrainingHandle`] (for stopping the job and monitoring it) and
/// a [`MetricsReceiver`] that yields a [`TrainingMetrics`] snapshot after every
/// optimiser step.
///
/// # Example
/// ```no_run
/// # use quark_core::training::trainer::{TrainerConfig, start_training};
/// # use quark_core::model::config::QuarkConfig;
/// let (handle, mut rx) = start_training(
///     QuarkConfig::quark_1b(),
///     TrainerConfig::default(),
///     vec![],
/// );
/// // Later…
/// handle.stop();
/// ```
pub fn start_training(
    model_config: QuarkConfig,
    trainer_config: TrainerConfig,
    batches: Vec<DataBatch>,
) -> (TrainingHandle, MetricsReceiver) {
    let (tx, rx) = mpsc::unbounded_channel::<TrainingMetrics>();
    let stop_flag = Arc::new(AtomicBool::new(false));

    let stop_clone = Arc::clone(&stop_flag);
    let tx_clone = tx.clone();

    tokio::spawn(async move {
        if let Err(e) =
            run_training_loop(model_config, trainer_config, batches, tx_clone, stop_clone).await
        {
            tracing::error!("Training error: {e}");
        }
    });

    (TrainingHandle { sender: tx, stop_flag }, rx)
}

// ─────────────────────────────────────────────────────────────────────────────
// Training loop
// ─────────────────────────────────────────────────────────────────────────────

/// Core training loop executed inside the background task.
///
/// # ⚠ Simulation notice
///
/// **This function currently *simulates* a training loop** — it does not
/// perform real forward or backward passes through the model.  Actual gradient
/// computation requires:
///
/// 1. The `QuarkModel` architecture to be finalised and compiled.
/// 2. An autodiff backend (e.g. `burn-autodiff` wrapping `burn-ndarray` or
///    `burn-wgpu`) to be selected at runtime and threaded through the model.
/// 3. Burn's `AutodiffModule::backward` + `GradientsParams` machinery to
///    compute per-parameter gradients.
/// 4. The AdamW optimiser obtained from `AdamWConfig::to_burn_config().init()`
///    to apply those gradients.
/// 5. Per-step learning-rate injection via `CosineSchedule::get_lr`.
///
/// The placeholder loss curve (`1 - step/max_steps + noise`) and the fixed
/// `grad_norm: 1.0` are intentional stand-ins.  They allow the GUI metrics
/// panel to be exercised end-to-end while model development is ongoing.
///
/// Once `QuarkModel::forward` is available this function should be replaced
/// with a real loop structured roughly as:
/// ```text
/// let logits  = model.forward(batch.input_ids);
/// let loss    = cross_entropy(logits, batch.labels);
/// let grads   = loss.backward();
/// let grads   = GradientsParams::from_grads(grads, &model);
/// let model   = optimizer.step(lr, model, grads);
/// ```
async fn run_training_loop(
    model_config: QuarkConfig,
    config: TrainerConfig,
    batches: Vec<DataBatch>,
    tx: MetricsSender,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    use std::time::Instant;

    tracing::info!("Starting training for {} steps", config.max_steps);
    std::fs::create_dir_all(&config.output_dir)?;

    let total_batches = batches.len();
    if total_batches == 0 {
        tracing::warn!("No training batches provided — running dry loop");
    }

    let start_time = Instant::now();
    let mut step = 0u64;
    let mut epoch = 0u32;
    let mut batch_idx = 0usize;

    while step < config.max_steps && !stop.load(Ordering::SeqCst) {
        if total_batches > 0 {
            if batch_idx >= total_batches {
                batch_idx = 0;
                epoch += 1;
            }
            let _batch = &batches[batch_idx];
            batch_idx += 1;
        }

        let lr = config.schedule.get_lr(step) as f32;

        // Placeholder loss that decreases monotonically with some noise.
        let loss = (1.0_f32 - step as f32 / config.max_steps as f32).max(0.1)
            + 0.05 * rand_f32_seed(step);

        let tokens_per_sec = {
            let elapsed = start_time.elapsed().as_secs_f32();
            let avg_step_time = elapsed / step.max(1) as f32;
            model_config.max_position_embeddings as f32 * config.batch_size as f32
                / avg_step_time.max(1e-6)
        };

        let elapsed_secs = start_time.elapsed().as_secs();
        let eta_secs = if step > 0 {
            elapsed_secs.checked_mul(config.max_steps - step)
                .and_then(|n| n.checked_div(step))
                .unwrap_or(0)
        } else {
            0
        };

        let _ = tx.send(TrainingMetrics {
            step,
            loss,
            learning_rate: lr,
            tokens_per_sec,
            grad_norm: 1.0,
            vram_used_bytes: 0,
            ram_used_bytes: 0,
            disk_used_bytes: 0,
            epoch,
            eta_secs,
        });

        if step > 0 && step.is_multiple_of(config.save_every_steps) {
            let ckpt = config.output_dir.join(format!("checkpoint-{step}"));
            std::fs::create_dir_all(&ckpt)?;
            tracing::info!("Saved checkpoint at step {step} → {}", ckpt.display());
        }

        step += 1;
        // Yield so the async runtime can deliver metrics to the GUI.
        tokio::task::yield_now().await;
    }

    tracing::info!("Training complete at step {step}");
    Ok(())
}

/// Deterministic low-quality pseudo-random noise for the placeholder loss curve.
fn rand_f32_seed(seed: u64) -> f32 {
    let x = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((x >> 33) as f32) / (u32::MAX as f32)
}
