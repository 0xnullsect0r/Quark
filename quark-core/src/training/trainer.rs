use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use burn::{
    module::{AutodiffModule, Module},
    nn::loss::CrossEntropyLossConfig,
    optim::{GradientsAccumulator, GradientsParams, Optimizer},
    record::Recorder,
    tensor::{
        ElementConversion, Int, Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
};

use crate::checkpoint::CheckpointRecorder;
use crate::data::batch::{DataBatch, collate_batch};
use crate::data::loader::TextLoader;
use crate::data::packing::pack_sequences;
use crate::memory::tier::TierConfig;
use crate::model::QuarkModel;
use crate::model::config::QuarkConfig;
use crate::tokenizer::bpe::{PAD_ID, QuarkTokenizer};
use crate::training::adamw::AdamWConfig;
use crate::training::grad_clip::clip_grad_norm;
use crate::training::lr_schedule::CosineSchedule;
use crate::training::metrics::{MetricsReceiver, MetricsSender, TrainingEvent, TrainingMetrics};

/// Weight of the MoE load-balancing loss added to the LM loss.
const AUX_LOSS_COEF: f32 = 0.01;
/// Maximum number of held-out batches used per evaluation.
const MAX_EVAL_BATCHES: usize = 8;

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
    /// Continue from the latest `checkpoint-N.bin` in `output_dir` when its
    /// saved `config.json` matches the current model config.
    #[serde(default = "default_true")]
    pub resume: bool,
}

fn default_true() -> bool {
    true
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
            resume: true,
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

    // Look for a checkpoint to resume from before config.json is overwritten.
    let resume_from = if config.resume {
        find_resumable_checkpoint(&config.output_dir, &model_config)
    } else {
        None
    };

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
    let mut batches = if !corpus_files.is_empty() {
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
            .chunks(config.batch_size.max(1))
            .map(|chunk| collate_batch(chunk.to_vec(), PAD_ID))
            .collect::<Vec<_>>()
    } else {
        log!("⚠  No corpus files provided — running demo loop with random inputs");
        vec![]
    };

    // Hold out ~1% of batches for evaluation once there is enough data.
    let eval_batches = if batches.len() >= 20 {
        let n_eval = (batches.len() / 100).max(1);
        batches.split_off(batches.len() - n_eval)
    } else {
        Vec::new()
    };

    log!(
        "   {} training batches ready ({} held out for eval)",
        batches.len(),
        eval_batches.len()
    );

    // ── Initialise model ──────────────────────────────────────────────────────
    phase!("Initialising model…");

    use crate::backend::TrainBackend;
    type AB = TrainBackend;

    let device = <AB as Backend>::Device::default();
    <AB as Backend>::seed(config.seed);
    let mut model = QuarkModel::<AB>::new(&model_config, &device);
    log!("   Model initialised on {:?}", device);

    let mut step = 0u64;
    if let Some((path, saved_step)) = resume_from {
        match CheckpointRecorder::new().load(path.with_extension(""), &device) {
            Ok(record) => {
                model = model.load_record(record);
                step = saved_step;
                log!(
                    "↻  Resumed from {} (step {saved_step}); optimiser state starts fresh",
                    path.display()
                );
            }
            Err(e) => log!("⚠  Could not resume from {}: {e} — starting fresh", path.display()),
        }
    }
    if step >= config.max_steps {
        log!("✅  Already trained for {step} steps (max_steps={})", config.max_steps);
        phase!("Complete!");
        let _ = tx.send(TrainingEvent::Done);
        return;
    }

    // ── Initialise optimiser ──────────────────────────────────────────────────
    let mut optim = config.adamw.to_burn_config().init();

    // ── Main loop ─────────────────────────────────────────────────────────────
    phase!("Training…");
    log!("▶  Starting training loop");

    let accum_steps = config.grad_accum_steps.max(1);
    let start_step = step;
    let start_time = Instant::now();
    let total_batches = batches.len();
    let mut batch_idx = (step as usize * accum_steps) % total_batches.max(1);
    let mut epoch = (step as usize * accum_steps / total_batches.max(1)) as u32;
    let mut tokens_seen = 0u64;
    let ce_loss = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![PAD_ID as usize]))
        .init(&device);

    while step < config.max_steps && !stop.load(Ordering::SeqCst) {
        let lr = config.schedule.get_lr(step);
        let mut accumulator = GradientsAccumulator::new();
        let mut step_loss = 0.0f32;

        for micro in 0..accum_steps {
            // ── Get next batch ────────────────────────────────────────────────
            let (input_ids, label_ids, n_tokens) = if total_batches > 0 {
                if batch_idx >= total_batches {
                    batch_idx = 0;
                    epoch += 1;
                    log!("━━  Epoch {} started", epoch + 1);
                }
                let batch = &batches[batch_idx];
                batch_idx += 1;
                batch_tensors::<AB>(batch, &device)
            } else {
                demo_batch::<AB>(
                    config.batch_size,
                    model_config.max_position_embeddings.min(64),
                    vocab_size,
                    step as usize * accum_steps + micro,
                    &device,
                )
            };
            tokens_seen += n_tokens;

            // ── Forward + backward ────────────────────────────────────────────
            let (logits, aux) = model.forward_with_aux(input_ids); // [batch, seq, vocab]
            let [b, s, v] = logits.dims();
            let ce = ce_loss.forward(logits.reshape([b * s, v]), label_ids.reshape([b * s]));
            step_loss += ce.clone().into_scalar().elem::<f32>() / accum_steps as f32;

            let loss = match aux {
                Some(aux) => ce + aux * AUX_LOSS_COEF,
                None => ce,
            };
            let loss = loss / accum_steps as f32;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            accumulator.accumulate(&model, grads);
        }

        // ── Clip + optimiser step ─────────────────────────────────────────────
        let mut grads = accumulator.grads();
        let grad_norm = clip_grad_norm(&model, &mut grads, config.max_grad_norm);
        model = optim.step(lr, model, grads);
        step += 1;

        let elapsed = start_time.elapsed().as_secs_f32();
        let tokens_per_sec = if elapsed > 0.0 { tokens_seen as f32 / elapsed } else { 0.0 };
        let steps_this_run = step - start_step;
        let eta_secs =
            (elapsed / steps_this_run as f32 * (config.max_steps - step) as f32) as u64;

        let _ = tx.send(TrainingEvent::Metrics(TrainingMetrics {
            step,
            loss: step_loss,
            learning_rate: lr as f32,
            tokens_per_sec,
            grad_norm,
            vram_used_bytes: 0,
            ram_used_bytes: 0,
            disk_used_bytes: 0,
            epoch,
            eta_secs,
        }));

        if steps_this_run == 1 {
            log!("   First optimizer step complete");
        } else if step.is_multiple_of(100) {
            log!(
                "   step={step:>6}  loss={step_loss:.4}  lr={lr:.2e}  |g|={grad_norm:.3}  {:.0} tok/s",
                tokens_per_sec
            );
        }

        // ── Evaluate ──────────────────────────────────────────────────────────
        if !eval_batches.is_empty()
            && config.eval_every_steps > 0
            && step.is_multiple_of(config.eval_every_steps)
        {
            let loss = evaluate(&model.valid(), &eval_batches, &device);
            log!("   eval  step={step:>6}  loss={loss:.4}  ppl={:.1}", loss.exp());
            let _ = tx.send(TrainingEvent::Eval { step, loss });
        }

        // ── Save checkpoint ───────────────────────────────────────────────────
        if step < config.max_steps
            && config.save_every_steps > 0
            && step.is_multiple_of(config.save_every_steps)
        {
            save_burn_checkpoint(&model, &config.output_dir, step, &tx);
        }
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

// ── Batch helpers ─────────────────────────────────────────────────────────────

/// Convert a collated batch to `(input_ids, labels, non_pad_tokens)`.
fn batch_tensors<B: Backend>(
    batch: &DataBatch,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>, u64) {
    let b = batch.input_ids.len();
    let s = batch.input_ids.first().map(|r| r.len()).unwrap_or(1).max(1);

    let input_flat: Vec<i32> =
        batch.input_ids.iter().flat_map(|row| row.iter().map(|&id| id as i32)).collect();
    let label_flat: Vec<i32> =
        batch.labels.iter().flat_map(|row| row.iter().map(|&id| id as i32)).collect();
    let n_tokens = label_flat.iter().filter(|&&id| id != PAD_ID as i32).count() as u64;

    (
        Tensor::from_data(TensorData::new(input_flat, [b, s]), device),
        Tensor::from_data(TensorData::new(label_flat, [b, s]), device),
        n_tokens,
    )
}

/// Deterministic synthetic batch used when no corpus is provided.
fn demo_batch<B: Backend>(
    batch_size: usize,
    seq: usize,
    vocab_size: usize,
    offset: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>, u64) {
    let n = batch_size * seq;
    let ids: Vec<i32> = (0..n).map(|i| ((offset * n + i) % vocab_size) as i32).collect();
    let lbl: Vec<i32> = (0..n).map(|i| ((offset * n + i + 1) % vocab_size) as i32).collect();
    (
        Tensor::from_data(TensorData::new(ids, [batch_size, seq]), device),
        Tensor::from_data(TensorData::new(lbl, [batch_size, seq]), device),
        n as u64,
    )
}

/// Mean cross-entropy over (up to `MAX_EVAL_BATCHES`) held-out batches.
fn evaluate<B: Backend>(model: &QuarkModel<B>, batches: &[DataBatch], device: &B::Device) -> f32 {
    let ce_loss = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![PAD_ID as usize]))
        .init(device);
    let used = &batches[..batches.len().min(MAX_EVAL_BATCHES)];
    let total: f32 = used
        .iter()
        .map(|batch| {
            let (input_ids, label_ids, _) = batch_tensors::<B>(batch, device);
            let logits = model.forward(input_ids);
            let [b, s, v] = logits.dims();
            ce_loss
                .forward(logits.reshape([b * s, v]), label_ids.reshape([b * s]))
                .into_scalar()
                .elem::<f32>()
        })
        .sum();
    total / used.len() as f32
}

// ── Checkpoint helpers ────────────────────────────────────────────────────────

/// Latest `checkpoint-N.bin` in `dir`, if the `config.json` saved alongside it
/// matches `model_config` exactly.
fn find_resumable_checkpoint(dir: &Path, model_config: &QuarkConfig) -> Option<(PathBuf, u64)> {
    let saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).ok()?).ok()?;
    if saved != serde_json::to_value(model_config).ok()? {
        return None;
    }
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            let step = name.strip_prefix("checkpoint-")?.strip_suffix(".bin")?.parse().ok()?;
            Some((path, step))
        })
        .max_by_key(|(_, step)| *step)
}

fn save_burn_checkpoint(
    model: &QuarkModel<crate::backend::TrainBackend>,
    output_dir: &std::path::Path,
    step: u64,
    tx: &MetricsSender,
) {
    macro_rules! log {
        ($($t:tt)*) => {{ let _ = tx.send(TrainingEvent::Log(format!($($t)*))); }};
    }

    // The recorder appends ".bin" automatically.
    let stem = output_dir.join(format!("checkpoint-{step}"));

    let record = model.clone().into_record();
    match CheckpointRecorder::new().record(record, stem.clone()) {
        Ok(_) => log!("💾  Checkpoint saved → {}.bin", stem.display()),
        Err(e) => log!("⚠  Checkpoint save failed: {e}"),
    }
}
