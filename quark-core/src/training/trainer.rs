use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::checkpoint::{save_checkpoint, TensorData};
use crate::data::batch::DataBatch;
use crate::memory::tier::TierConfig;
use crate::model::config::QuarkConfig;
use crate::training::adamw::AdamWConfig;
use crate::training::lr_schedule::CosineSchedule;
use crate::training::metrics::{MetricsReceiver, MetricsSender, TrainingEvent, TrainingMetrics};

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
            output_dir: crate::paths::checkpoints_dir(),
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
    pub sender: MetricsSender,
    /// Set to `true` to request a graceful stop after the current step.
    pub stop_flag: Arc<AtomicBool>,
}

impl TrainingHandle {
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }
}

/// Orchestrates the training loop and streams events to the GUI / CLI.
///
/// This is the legacy wrapper retained for API compatibility.  New code should
/// prefer [`start_training`] which returns a [`TrainingHandle`] directly.
pub struct Trainer {
    config: TrainerConfig,
    sender: MetricsSender,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> (Self, MetricsReceiver) {
        let (tx, rx) = std::sync::mpsc::channel();
        (Self { config, sender: tx }, rx)
    }

    pub fn run(&self) -> anyhow::Result<()> {
        todo!("Trainer::run — use start_training() for the background loop")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Spawn the training loop as a background thread.
///
/// Returns a [`TrainingHandle`] (for stopping the job) and a [`MetricsReceiver`]
/// that yields [`TrainingEvent`]s.  The channel closes when the thread finishes.
pub fn start_training(
    model_config: QuarkConfig,
    trainer_config: TrainerConfig,
    batches: Vec<DataBatch>,
) -> (TrainingHandle, MetricsReceiver) {
    let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();
    let stop_flag = Arc::new(AtomicBool::new(false));

    let stop_clone = Arc::clone(&stop_flag);
    let tx_clone = tx.clone();

    std::thread::spawn(move || {
        run_training_loop(model_config, trainer_config, batches, tx_clone, stop_clone);
    });

    (TrainingHandle { sender: tx, stop_flag }, rx)
}

// ─────────────────────────────────────────────────────────────────────────────
// Training loop
// ─────────────────────────────────────────────────────────────────────────────

/// Core training loop executed in a background thread.
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
/// The placeholder loss curve and fixed `grad_norm: 1.0` are intentional
/// stand-ins that allow the GUI to be exercised end-to-end while model
/// development is ongoing.
fn run_training_loop(
    model_config: QuarkConfig,
    config: TrainerConfig,
    batches: Vec<DataBatch>,
    tx: MetricsSender,
    stop: Arc<AtomicBool>,
) {
    macro_rules! log {
        ($($t:tt)*) => {{ let _ = tx.send(TrainingEvent::Log(format!($($t)*))); }};
    }
    macro_rules! phase {
        ($($t:tt)*) => {{ let _ = tx.send(TrainingEvent::Phase(format!($($t)*))); }};
    }

    tracing::info!("Starting training for {} steps", config.max_steps);

    if let Err(e) = std::fs::create_dir_all(&config.output_dir) {
        let _ = tx.send(TrainingEvent::Error(format!("Cannot create output dir: {e}")));
        return;
    }

    let total_batches = batches.len();
    if total_batches == 0 {
        log!("⚠  No training batches provided — running dry simulation loop");
    }

    log!("▶  Training started");
    log!(
        "   Model:     {} layers, {} heads, hidden={}",
        model_config.num_hidden_layers,
        model_config.num_attention_heads,
        model_config.hidden_size
    );
    log!(
        "   Config:    max_steps={}, batch_size={}, grad_accum={}",
        config.max_steps,
        config.batch_size,
        config.grad_accum_steps
    );
    log!(
        "   Output:    {}",
        config.output_dir.display()
    );
    phase!("Initialising…");

    let start_time = Instant::now();
    let mut step = 0u64;
    let mut epoch = 0u32;
    let mut batch_idx = 0usize;

    while step < config.max_steps && !stop.load(Ordering::SeqCst) {
        if total_batches > 0 {
            if batch_idx >= total_batches {
                batch_idx = 0;
                epoch += 1;
                log!("━━  Epoch {} started", epoch + 1);
            }
            let _batch = &batches[batch_idx];
            batch_idx += 1;
        }

        let lr = config.schedule.get_lr(step) as f32;
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
            elapsed_secs
                .checked_mul(config.max_steps - step)
                .and_then(|n| n.checked_div(step))
                .unwrap_or(0)
        } else {
            0
        };

        let _ = tx.send(TrainingEvent::Metrics(TrainingMetrics {
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
        }));

        // Log at milestones
        if step == 0 {
            phase!("Training…");
        } else if step.is_multiple_of(100) {
            log!(
                "   step={step:>6}  loss={loss:.4}  lr={lr:.2e}  {:.0} tok/s",
                tokens_per_sec
            );
        }

        if step > 0 && step.is_multiple_of(config.save_every_steps) {
            let ckpt_path = config.output_dir.join(format!("checkpoint-{step}.safetensors"));
            let stub = TensorData { name: "step".into(), data: vec![step as f32], shape: vec![1] };
            match save_checkpoint(&ckpt_path, &[stub]) {
                Ok(()) => log!("💾  Checkpoint saved → {}", ckpt_path.display()),
                Err(e) => log!("⚠  Checkpoint save failed: {e}"),
            }
        }

        step += 1;
        // Yield briefly so the channel consumer can drain messages without pegging a CPU core.
        std::thread::sleep(Duration::from_millis(1));
    }

    if stop.load(Ordering::SeqCst) {
        log!("⏹  Training stopped at step {step}");
        phase!("Stopped");
    } else {
        log!("✅  Training complete — {step} steps in {:.1}s", start_time.elapsed().as_secs_f32());
        phase!("Complete!");
    }

    // Save a final checkpoint regardless of stop/complete.
    let final_path = config.output_dir.join("checkpoint-final.safetensors");
    let stub = TensorData { name: "step".into(), data: vec![step as f32], shape: vec![1] };
    match save_checkpoint(&final_path, &[stub]) {
        Ok(()) => log!("💾  Final checkpoint → {}", final_path.display()),
        Err(e) => log!("⚠  Final checkpoint save failed: {e}"),
    }

    let _ = tx.send(TrainingEvent::Done);
}

/// Deterministic low-quality pseudo-random noise for the placeholder loss curve.
fn rand_f32_seed(seed: u64) -> f32 {
    let x = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((x >> 33) as f32) / (u32::MAX as f32)
}
