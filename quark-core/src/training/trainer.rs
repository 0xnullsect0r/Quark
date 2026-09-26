use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use burn::{
    module::{AutodiffModule, Module},
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
use crate::data::sft::{collate_sft, load_conversations, tokenize_conversation};
use crate::memory::{budget::HardwareBudget, tier::TierConfig};
use crate::model::QuarkModel;
use crate::model::config::QuarkConfig;
use crate::tokenizer::bpe::{PAD_ID, QuarkTokenizer};
use crate::training::adamw::AdamWConfig;
use crate::training::loss::masked_cross_entropy;
use crate::training::grad_clip::clip_grad_norm;
use crate::training::lr_schedule::CosineSchedule;
use crate::training::metrics::{MetricsReceiver, MetricsSender, TrainingEvent, TrainingMetrics};

/// Weight of the MoE load-balancing loss added to the LM loss.
pub(crate) const AUX_LOSS_COEF: f32 = 0.01;
/// Maximum number of held-out batches used per evaluation.
pub(crate) const MAX_EVAL_BATCHES: usize = 8;

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
    /// Numeric precision for training compute.
    #[serde(default)]
    pub precision: Precision,
    /// Recompute cheap activations in the backward pass instead of storing
    /// them (Burn's balanced checkpointing): less memory, somewhat slower.
    #[serde(default = "default_true")]
    pub gradient_checkpointing: bool,
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
    /// Pretraining on raw text, or chat fine-tuning of an existing checkpoint.
    #[serde(default)]
    pub mode: TrainingMode,
    /// Stream the model layer by layer through RAM/disk (see
    /// `training::streamed`). `Auto` does so when the model doesn't fit.
    #[serde(default)]
    pub offload: OffloadMode,
    /// Optimizer used when offloading (the in-memory trainer uses AdamW).
    #[serde(default)]
    pub optimizer: crate::training::optim::OptimizerKind,
}

/// Whether to train in memory or streamed through RAM/disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum OffloadMode {
    /// Stream when the in-memory estimate doesn't fit.
    #[default]
    Auto,
    /// Always train in memory.
    Off,
    /// Always stream.
    On,
}

/// What kind of training run this is.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TrainingMode {
    /// Next-token prediction on raw `.txt` / `.jsonl` text.
    #[default]
    Pretrain,
    /// Supervised fine-tuning on chat conversations (`data::sft` JSONL),
    /// starting from `base_checkpoint`. The architecture and tokenizer come
    /// from the `config.json` / `tokenizer.json` next to it.
    FineTune { base_checkpoint: PathBuf },
}

/// Learning rate suggested for fine-tuning (lower than pretraining).
pub const FINE_TUNE_LR: f64 = 5e-5;

fn default_true() -> bool {
    true
}

/// Numeric precision for training compute. Checkpoints are always saved in
/// f32, so inference is unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Precision {
    #[default]
    F32,
    /// bfloat16 compute: half the memory. Only available in CUDA builds.
    Bf16,
}

impl Precision {
    pub fn bytes_per_elem(self) -> u64 {
        match self {
            Precision::F32 => 4,
            Precision::Bf16 => 2,
        }
    }

    /// Whether this build can train at this precision.
    pub fn is_supported(self) -> bool {
        match self {
            Precision::F32 => true,
            Precision::Bf16 => cfg!(feature = "backend-cuda"),
        }
    }
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
            precision: Precision::F32,
            gradient_checkpointing: true,
            max_grad_norm: 1.0,
            seed: 42,
            adamw: AdamWConfig::default(),
            schedule: CosineSchedule::default(),
            tier: TierConfig::default(),
            resume: true,
            mode: TrainingMode::Pretrain,
            offload: OffloadMode::Auto,
            optimizer: crate::training::optim::OptimizerKind::AdamW,
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
        let panic_tx = tx_clone.clone();
        let run = std::panic::AssertUnwindSafe(move || {
            dispatch_training(model_config, trainer_config, corpus_files, tokenizer_path, tx_clone, stop_clone)
        });
        // Report crashes (e.g. out of memory, no GPU adapter) instead of
        // leaving the UI waiting forever.
        if let Err(panic) = std::panic::catch_unwind(run) {
            let msg = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            let _ = panic_tx.send(TrainingEvent::Error(format!("Training crashed: {msg}")));
        }
    });

    (TrainingHandle { sender: tx, stop_flag }, rx)
}

/// Pick the concrete autodiff backend for the requested precision, then run
/// the loop on it.
///
fn dispatch_training(
    model_config: QuarkConfig,
    mut config: TrainerConfig,
    corpus_files: Vec<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    tx: MetricsSender,
    stop: Arc<AtomicBool>,
) {
    if !config.precision.is_supported() {
        let _ = tx.send(TrainingEvent::Log(format!(
            "⚠  {:?} training needs a CUDA build — using F32",
            config.precision
        )));
        config.precision = Precision::F32;
    }

    macro_rules! run {
        ($backend:ty) => {
            run_training_loop::<$backend>(model_config, config, corpus_files, tokenizer_path, tx, stop)
        };
    }

    use burn::backend::{autodiff::checkpoint::strategy::BalancedCheckpointing, Autodiff};

    match (config.precision, config.gradient_checkpointing) {
        #[cfg(feature = "backend-cuda")]
        (Precision::Bf16, true) => {
            run!(Autodiff<crate::backend::ComputeBackendBf16, BalancedCheckpointing>)
        }
        #[cfg(feature = "backend-cuda")]
        (Precision::Bf16, false) => run!(Autodiff<crate::backend::ComputeBackendBf16>),
        (_, true) => run!(Autodiff<crate::backend::ComputeBackend, BalancedCheckpointing>),
        (_, false) => run!(crate::backend::TrainBackend),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Training loop
// ─────────────────────────────────────────────────────────────────────────────

fn run_training_loop<AB: AutodiffBackend>(
    mut model_config: QuarkConfig,
    mut config: TrainerConfig,
    corpus_files: Vec<PathBuf>,
    mut tokenizer_path: Option<PathBuf>,
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

    // Fine-tuning continues the base model: same architecture and tokenizer,
    // and never writes into the base checkpoint's folder.
    if let TrainingMode::FineTune { base_checkpoint } = &config.mode {
        let Some(base_config) = QuarkConfig::for_checkpoint(base_checkpoint) else {
            bail!(
                "No config.json next to {} — fine-tuning needs a checkpoint trained by Quark",
                base_checkpoint.display()
            );
        };
        model_config = base_config;
        let base_dir = base_checkpoint.parent().map(Path::to_path_buf).unwrap_or_default();
        let own_tokenizer = base_checkpoint.join("tokenizer.json"); // sharded checkpoints
        tokenizer_path = Some(if own_tokenizer.exists() {
            own_tokenizer
        } else {
            base_dir.join("tokenizer.json")
        });
        if same_dir(&config.output_dir, &base_dir) {
            config.output_dir = base_dir.join("finetune");
        }
        log!("▶  Fine-tuning {}", base_checkpoint.display());
    }

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

    // ── Load and tokenize training data ───────────────────────────────────────
    let mut batches = match &config.mode {
        TrainingMode::FineTune { .. } => {
            match sft_batches(&corpus_files, &tokenizer, &model_config, &config, &tx) {
                Ok(b) => b,
                Err(e) => bail!("{e}"),
            }
        }
        TrainingMode::Pretrain if corpus_files.is_empty() => {
            log!("⚠  No corpus files provided — running demo loop with random inputs");
            vec![]
        }
        TrainingMode::Pretrain => {
            match pretrain_batches(corpus_files, &tokenizer, &model_config, &config, &tx) {
                Ok(b) => b,
                Err(e) => bail!("{e}"),
            }
        }
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

    let estimate = estimate_memory(&model_config, &config, &HardwareBudget::detect());
    log!(
        "   {:.1}M params, ≈{} needed to train in memory ({} {} available)",
        model_config.param_count() as f64 / 1e6,
        fmt_gb(estimate.needed_bytes),
        fmt_gb(estimate.available_bytes),
        estimate.device
    );
    let streamed = match config.offload {
        OffloadMode::On => true,
        OffloadMode::Off => false,
        OffloadMode::Auto => !estimate.fits(),
    };
    if streamed {
        log!("▶  Streaming the model layer by layer through RAM and disk (offload)");
        let run = crate::training::streamed::run_streamed::<AB>(
            model_config,
            &config,
            batches,
            eval_batches,
            &tx,
            &stop,
        );
        match run {
            Ok(()) => {
                let _ = tx.send(TrainingEvent::Done);
            }
            Err(e) => {
                let _ = tx.send(TrainingEvent::Error(format!("{e:#}")));
            }
        }
        return;
    }
    if !estimate.fits() {
        log!(
            "⚠  This probably won't fit in {} — if training crashes or swaps, use a smaller \
             preset, batch size or context length, or turn on offloading",
            estimate.device
        );
    }

    // ── Initialise model ──────────────────────────────────────────────────────
    phase!("Initialising model…");

    let device = burn::tensor::Device::<AB>::default();
    <AB as Backend>::seed(&device, config.seed);
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
    if let (0, TrainingMode::FineTune { base_checkpoint }) = (step, &config.mode) {
        let loaded = if crate::checkpoint::sharded::is_sharded(base_checkpoint) {
            crate::checkpoint::sharded::load_sharded::<AB>(base_checkpoint, &device)
                .map(|(_, m)| m)
        } else {
            CheckpointRecorder::new()
                .load(base_checkpoint.with_extension(""), &device)
                .map(|record| model.clone().load_record(record))
                .map_err(anyhow::Error::from)
        };
        match loaded {
            Ok(m) => {
                model = m;
                log!("   Loaded base weights from {}", base_checkpoint.display());
            }
            Err(e) => bail!("Could not load base checkpoint {}: {e:#}", base_checkpoint.display()),
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
    let mut ram_used_bytes = 0u64;
    let mut vram_used_bytes = 0u64;
    let mut sys = sysinfo::System::new();

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
            let ce = masked_cross_entropy(logits.reshape([b * s, v]), label_ids.reshape([b * s]), PAD_ID);
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

        if steps_this_run == 1 || step.is_multiple_of(10) {
            ram_used_bytes = process_rss(&mut sys).unwrap_or(ram_used_bytes);
            let (vram_total, vram_free) = crate::memory::budget::detect_vram();
            vram_used_bytes = vram_total.saturating_sub(vram_free);
        }

        let _ = tx.send(TrainingEvent::Metrics(TrainingMetrics {
            step,
            loss: step_loss,
            learning_rate: lr as f32,
            tokens_per_sec,
            grad_norm,
            vram_used_bytes,
            ram_used_bytes,
            disk_used_bytes: 0,
            disk_io_bytes: 0,
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

// ── Data helpers ──────────────────────────────────────────────────────────────

/// Raw text → packed next-token-prediction batches.
fn pretrain_batches(
    corpus_files: Vec<PathBuf>,
    tokenizer: &QuarkTokenizer,
    model_config: &QuarkConfig,
    config: &TrainerConfig,
    tx: &MetricsSender,
) -> Result<Vec<DataBatch>, String> {
    let log = |msg: String| {
        let _ = tx.send(TrainingEvent::Log(msg));
    };
    let _ = tx.send(TrainingEvent::Phase("Loading corpus…".into()));
    log(format!("   Loading {} corpus file(s)…", corpus_files.len()));

    let loader = TextLoader::new(corpus_files, model_config.max_position_embeddings);
    let texts = loader.load_texts().map_err(|e| format!("Corpus load failed: {e}"))?;
    log(format!("   Loaded {} documents", texts.len()));

    let _ = tx.send(TrainingEvent::Phase("Tokenizing…".into()));
    let token_seqs: Vec<Vec<u32>> = texts
        .iter()
        .filter_map(|text| tokenizer.encode(text).ok())
        .filter(|ids| !ids.is_empty())
        .collect();
    log(format!("   Tokenized {} sequences", token_seqs.len()));

    let packed = pack_sequences(token_seqs, model_config.max_position_embeddings);
    log(format!(
        "   Packed into {} chunks of {} tokens",
        packed.len(),
        model_config.max_position_embeddings
    ));
    if packed.is_empty() {
        return Err("No training tokens after packing. Check your corpus files.".into());
    }

    Ok(packed
        .chunks(config.batch_size.max(1))
        .map(|chunk| collate_batch(chunk.to_vec(), PAD_ID))
        .collect())
}

/// Chat conversations (JSONL) → shuffled fine-tuning batches that train only
/// on assistant turns.
fn sft_batches(
    files: &[PathBuf],
    tokenizer: &QuarkTokenizer,
    model_config: &QuarkConfig,
    config: &TrainerConfig,
    tx: &MetricsSender,
) -> Result<Vec<DataBatch>, String> {
    use rand::{SeedableRng, seq::SliceRandom};

    let log = |msg: String| {
        let _ = tx.send(TrainingEvent::Log(msg));
    };
    if files.is_empty() {
        return Err("Fine-tuning needs chat .jsonl files ({\"messages\": [...]} per line)".into());
    }
    let _ = tx.send(TrainingEvent::Phase("Loading conversations…".into()));
    let (conversations, skipped) =
        load_conversations(files).map_err(|e| format!("Conversation load failed: {e}"))?;
    log(format!("   Loaded {} conversations ({skipped} unparseable lines skipped)", conversations.len()));

    let _ = tx.send(TrainingEvent::Phase("Tokenizing…".into()));
    let mut examples = Vec::with_capacity(conversations.len());
    let mut truncated_away = 0;
    let mut failed = 0;
    let mut first_error = None;
    for messages in &conversations {
        match tokenize_conversation(tokenizer, messages, model_config.max_position_embeddings) {
            Ok(Some(ex)) => examples.push(ex),
            Ok(None) => truncated_away += 1,
            Err(e) => {
                failed += 1;
                first_error.get_or_insert(e.to_string());
            }
        }
    }
    if let Some(e) = first_error {
        log(format!("⚠  {failed} conversations could not be tokenized and were skipped: {e}"));
    }
    if truncated_away > 0 {
        log(format!(
            "⚠  {truncated_away} conversations had no assistant turn within {} tokens and were skipped",
            model_config.max_position_embeddings
        ));
    }
    if examples.is_empty() {
        return Err("No usable conversations: each needs at least one assistant message".into());
    }

    examples.shuffle(&mut rand::rngs::StdRng::seed_from_u64(config.seed));
    Ok(examples.chunks(config.batch_size.max(1)).map(|c| collate_sft(c, PAD_ID)).collect())
}

fn same_dir(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

// ── Memory helpers ────────────────────────────────────────────────────────────

/// Estimated training memory against what the training device has free.
#[derive(Debug, Clone, Copy)]
pub struct MemoryEstimate {
    pub needed_bytes: u64,
    pub available_bytes: u64,
    /// "VRAM" or "RAM".
    pub device: &'static str,
}

impl MemoryEstimate {
    pub fn fits(&self) -> bool {
        self.available_bytes == 0 || self.needed_bytes <= self.available_bytes
    }
}

/// Rough memory needed to train `model` with `config`, compared with free
/// VRAM (GPU builds, when detectable) or free RAM.
pub fn estimate_memory(
    model: &QuarkConfig,
    config: &TrainerConfig,
    budget: &HardwareBudget,
) -> MemoryEstimate {
    let precision = if config.precision.is_supported() { config.precision } else { Precision::F32 };
    let needed_bytes = model.training_memory_bytes(
        config.batch_size,
        model.max_position_embeddings,
        precision.bytes_per_elem(),
        config.gradient_checkpointing,
    );
    let gpu_build = cfg!(any(feature = "backend-cuda", feature = "backend-wgpu"));
    if gpu_build && budget.vram_total_bytes > 0 {
        MemoryEstimate { needed_bytes, available_bytes: budget.vram_free_bytes, device: "VRAM" }
    } else {
        MemoryEstimate { needed_bytes, available_bytes: budget.ram_free_bytes, device: "RAM" }
    }
}

pub(crate) fn fmt_gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

/// Resident memory of this process.
pub(crate) fn process_rss(sys: &mut sysinfo::System) -> Option<u64> {
    let pid = sysinfo::get_current_pid().ok()?;
    sys.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[pid]),
        false,
        sysinfo::ProcessRefreshKind::new().with_memory(),
    );
    sys.process(pid).map(|p| p.memory())
}

// ── Batch helpers ─────────────────────────────────────────────────────────────

/// Convert a collated batch to `(input_ids, labels, non_pad_tokens)`.
pub(crate) fn batch_tensors<B: Backend>(
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
    let used = &batches[..batches.len().min(MAX_EVAL_BATCHES)];
    let total: f32 = used
        .iter()
        .map(|batch| {
            let (input_ids, label_ids, _) = batch_tensors::<B>(batch, device);
            let logits = model.forward(input_ids);
            let [b, s, v] = logits.dims();
            masked_cross_entropy(logits.reshape([b * s, v]), label_ids.reshape([b * s]), PAD_ID)
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

fn save_burn_checkpoint<B: Backend>(
    model: &QuarkModel<B>,
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
