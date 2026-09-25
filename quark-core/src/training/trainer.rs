use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use burn::{
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::{GradientsParams, Optimizer},
    record::{CompactRecorder, Recorder},
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend},
};

use crate::checkpoint::{save_checkpoint, TensorData as CkptTensorData};
use crate::data::batch::{DataBatch, collate_batch};
use crate::data::loader::TextLoader;
use crate::data::packing::pack_sequences;
use crate::memory::tier::TierConfig;
use crate::model::QuarkModel;
use crate::model::config::QuarkConfig;
use crate::tokenizer::bpe::{PAD_ID, QuarkTokenizer};
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

pub struct TrainingHandle {
    pub sender: MetricsSender,
    pub stop_flag: Arc<AtomicBool>,
}

impl TrainingHandle {
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Spawn the training loop as a background thread.
///
/// `corpus_files` are `.txt` / `.jsonl` files to train on.
/// `tokenizer_path` is the trained BPE tokenizer; if `None`, the default
/// `~/.quark/datasets/tokenizer.json` is tried.
pub fn start_training(
    model_config: QuarkConfig,
    trainer_config: TrainerConfig,
    corpus_files: Vec<PathBuf>,
    tokenizer_path: Option<PathBuf>,
) -> (TrainingHandle, MetricsReceiver) {
    let (tx, rx) = std::sync::mpsc::channel::<TrainingEvent>();
    let stop_flag = Arc::new(AtomicBool::new(false));

    let stop_clone = Arc::clone(&stop_flag);
    let tx_clone = tx.clone();

    std::thread::spawn(move || {
        run_training_loop(model_config, trainer_config, corpus_files, tokenizer_path, tx_clone, stop_clone);
    });

    (TrainingHandle { sender: tx, stop_flag }, rx)
}

// ─────────────────────────────────────────────────────────────────────────────
// Training loop
// ─────────────────────────────────────────────────────────────────────────────

fn run_training_loop(
    mut model_config: QuarkConfig,
    config: TrainerConfig,
    corpus_files: Vec<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    tx: MetricsSender,
    stop: Arc<AtomicBool>,
) {
    macro_rules! log {
        ($($t:tt)*) => {{ let _ = tx.send(TrainingEvent::Log(format!($($t)*))); }};
    }
    macro_rules! phase {
        ($($t:tt)*) => {{ let _ = tx.send(TrainingEvent::Phase(format!($($t)*))); }};
    }
    macro_rules! bail {
        ($($t:tt)*) => {{
            let _ = tx.send(TrainingEvent::Error(format!($($t)*)));
            return;
        }};
    }

    tracing::info!("Starting training for {} steps", config.max_steps);

    if let Err(e) = std::fs::create_dir_all(&config.output_dir) {
        bail!("Cannot create output dir: {e}");
    }

    log!("▶  Training started");
    log!(
        "   Model:  {} layers, {} heads, hidden={}",
        model_config.num_hidden_layers,
        model_config.num_attention_heads,
        model_config.hidden_size
    );
    log!(
        "   Config: max_steps={}, batch_size={}, grad_accum={}",
        config.max_steps,
        config.batch_size,
        config.grad_accum_steps
    );
    log!("   Output: {}", config.output_dir.display());

    // ── Load tokenizer ────────────────────────────────────────────────────────
    phase!("Loading tokenizer…");
    let tok_path = tokenizer_path.unwrap_or_else(|| {
        crate::paths::datasets_dir().join("tokenizer.json")
    });

    let tokenizer = match QuarkTokenizer::load(&tok_path) {
        Ok(t) => {
            log!("   Tokenizer: {} vocab tokens", t.vocab_size());
            t
        }
        Err(e) => bail!("Tokenizer load failed: {e}\nTrain a tokenizer in the Dataset tab first."),
    };

    // The embedding and LM head must cover every id the tokenizer can emit.
    let vocab_size = tokenizer.vocab_size();
    if model_config.vocab_size != vocab_size {
        log!(
            "   vocab_size {} → {} (matching tokenizer)",
            model_config.vocab_size,
            vocab_size
        );
        model_config.vocab_size = vocab_size;
    }

    // Save the exact architecture + tokenizer next to the checkpoints so
    // inference and export can rebuild the same model.
    match serde_json::to_string_pretty(&model_config) {
        Ok(json) => {
            if let Err(e) = std::fs::write(config.output_dir.join("config.json"), json) {
                log!("⚠  Could not write config.json: {e}");
            }
        }
        Err(e) => log!("⚠  Could not serialise model config: {e}"),
    }
    let tok_dest = config.output_dir.join("tokenizer.json");
    if tok_dest != tok_path {
        if let Err(e) = std::fs::copy(&tok_path, &tok_dest) {
            log!("⚠  Could not copy tokenizer.json: {e}");
        }
    }

    // ── Load and tokenize corpus ──────────────────────────────────────────────
    let batches = if !corpus_files.is_empty() {
        phase!("Loading corpus…");
        log!("   Loading {} corpus file(s)…", corpus_files.len());

        let loader = TextLoader::new(corpus_files, model_config.max_position_embeddings);
        let texts = match loader.load_texts() {
            Ok(t) => t,
            Err(e) => bail!("Corpus load failed: {e}"),
        };
        log!("   Loaded {} documents", texts.len());

        phase!("Tokenizing…");
        let mut token_seqs: Vec<Vec<u32>> = Vec::new();
        for text in &texts {
            if let Ok(ids) = tokenizer.encode(text) {
                if !ids.is_empty() {
                    token_seqs.push(ids);
                }
            }
        }
        log!("   Tokenized {} sequences", token_seqs.len());

        let packed = pack_sequences(token_seqs, model_config.max_position_embeddings);
        log!("   Packed into {} chunks of {} tokens", packed.len(), model_config.max_position_embeddings);

        if packed.is_empty() {
            bail!("No training tokens after packing. Check your corpus files.");
        }

        packed
            .chunks(config.batch_size)
            .map(|chunk| collate_batch(chunk.to_vec(), PAD_ID))
            .collect::<Vec<_>>()
    } else {
        log!("⚠  No corpus files provided — running demo loop with random inputs");
        vec![]
    };

    log!("   {} training batches ready", batches.len());

    // ── Initialise model ──────────────────────────────────────────────────────
    phase!("Initialising model…");

    use crate::backend::TrainBackend;
    type AB = TrainBackend;

    let device = <AB as burn::tensor::backend::Backend>::Device::default();
    let mut model = QuarkModel::<AB>::new(&model_config, &device);
    log!("   Model initialised on {:?}", device);

    // ── Initialise optimiser ──────────────────────────────────────────────────
    let mut optim = config.adamw.to_burn_config().init();

    // ── Main loop ─────────────────────────────────────────────────────────────
    phase!("Training…");
    log!("▶  Starting training loop");

    let start_time = Instant::now();
    let mut step = 0u64;
    let mut epoch = 0u32;
    let mut batch_idx = 0usize;

    let total_batches = batches.len();
    let mut step_loss;

    while step < config.max_steps && !stop.load(Ordering::SeqCst) {
        // ── Get next batch ────────────────────────────────────────────────────
        let (input_ids, label_ids) = if total_batches > 0 {
            if batch_idx >= total_batches {
                batch_idx = 0;
                epoch += 1;
                log!("━━  Epoch {} started", epoch + 1);
            }
            let batch = &batches[batch_idx];
            batch_idx += 1;

            let b = batch.input_ids.len();
            let s = batch.input_ids.first().map(|r| r.len()).unwrap_or(1).max(1);

            let input_flat: Vec<i32> = batch
                .input_ids
                .iter()
                .flat_map(|row| row.iter().map(|&id| id as i32))
                .collect();
            let label_flat: Vec<i32> = batch
                .labels
                .iter()
                .flat_map(|row| row.iter().map(|&id| id as i32))
                .collect();

            (
                Tensor::<AB, 2, Int>::from_data(TensorData::new(input_flat, [b, s]), &device),
                Tensor::<AB, 2, Int>::from_data(TensorData::new(label_flat, [b, s]), &device),
            )
        } else {
            // Demo: random token ids in range [0, vocab_size)
            let b = config.batch_size;
            let s = model_config.max_position_embeddings.min(64);
            let ids: Vec<i32> = (0..b * s)
                .map(|i| ((step as usize * b * s + i) % vocab_size) as i32)
                .collect();
            let lbl: Vec<i32> = (0..b * s)
                .map(|i| ((step as usize * b * s + i + 1) % vocab_size) as i32)
                .collect();
            (
                Tensor::<AB, 2, Int>::from_data(TensorData::new(ids, [b, s]), &device),
                Tensor::<AB, 2, Int>::from_data(TensorData::new(lbl, [b, s]), &device),
            )
        };

        // ── Forward pass ──────────────────────────────────────────────────────
        let logits = model.forward(input_ids); // [batch, seq, vocab]
        let [b, s, v] = logits.dims();
        let logits_2d = logits.reshape([b * s, v]);
        let labels_1d = label_ids.reshape([b * s]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![PAD_ID as usize]))
            .init(&device)
            .forward(logits_2d, labels_1d);

        step_loss = loss.clone().into_scalar();
        // ── Backward + optimizer step ─────────────────────────────────────────
        // NOTE: Burn's optimizer API does not support explicit gradient
        // accumulation; grad_accum_steps is accepted in config but each
        // forward/backward updates the model immediately.
        let lr = config.schedule.get_lr(step);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(lr, model, grads);


        let elapsed = start_time.elapsed().as_secs_f32();
        let tokens_per_sec = if elapsed > 0.0 {
            (step + 1) as f32
                * config.batch_size as f32
                * model_config.max_position_embeddings as f32
                / elapsed
        } else {
            0.0
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
            loss: step_loss,
            learning_rate: lr as f32,
            tokens_per_sec,
            grad_norm: 1.0,
            vram_used_bytes: 0,
            ram_used_bytes: 0,
            disk_used_bytes: 0,
            epoch,
            eta_secs,
        }));

        if step == 0 {
            log!("   First optimizer step complete");
        } else if step.is_multiple_of(100) {
            log!(
                "   step={step:>6}  loss={step_loss:.4}  lr={lr:.2e}  {:.0} tok/s",
                tokens_per_sec
            );
        }

        // ── Save checkpoint ───────────────────────────────────────────────────
        if step > 0 && step.is_multiple_of(config.save_every_steps) {
            save_burn_checkpoint(&model, &config.output_dir, step, &tx);
        }

        step += 1;
    }

    // ── Final checkpoint ──────────────────────────────────────────────────────
    if stop.load(Ordering::SeqCst) {
        log!("⏹  Training stopped at step {step}");
        phase!("Stopped");
    } else {
        log!("✅  Training complete — {step} steps in {:.1}s", start_time.elapsed().as_secs_f32());
        phase!("Complete!");
    }

    save_burn_checkpoint(&model, &config.output_dir, step, &tx);
    let _ = tx.send(TrainingEvent::Done);
}

// ── Checkpoint helpers ────────────────────────────────────────────────────────

fn save_burn_checkpoint(
    model: &QuarkModel<crate::backend::TrainBackend>,
    output_dir: &std::path::Path,
    step: u64,
    tx: &MetricsSender,
) {
    macro_rules! log {
        ($($t:tt)*) => {{ let _ = tx.send(TrainingEvent::Log(format!($($t)*))); }};
    }

    // CompactRecorder appends ".bin" automatically.
    let stem = if step == u64::MAX {
        output_dir.join("checkpoint-final")
    } else {
        output_dir.join(format!("checkpoint-{step}"))
    };

    let record = model.clone().into_record();
    match CompactRecorder::new().record(record, stem.clone()) {
        Ok(_) => log!("💾  Checkpoint saved → {}.bin", stem.display()),
        Err(e) => log!("⚠  Checkpoint save failed: {e}"),
    }
}
