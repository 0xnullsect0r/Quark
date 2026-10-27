//! Headless Quark commands.
//!
//! ```text
//! quark-cli tokenizer <out.json> <corpus> [--vocab N]        train a BPE tokenizer
//! quark-cli train  <preset> <corpus> --out <dir> [...]        train (headless, resumable)
//! quark-cli export <checkpoint> <out-dir> [--q4 | --q8]       inference export (optionally quantized)
//! quark-cli infer  <checkpoint> "<prompt>" [--max N]           generate text
//! quark-cli bench  <preset> [--steps N] [--seq N] [--offload DIR] [--layer-only]
//! ```
//!
//! `<corpus>` is a `.txt` / `.jsonl` file or a folder of them; `<checkpoint>`
//! is a `.bin` file or a sharded `checkpoint-N/` folder.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use quark_core::{
    chat::{default_stop_strings, render_prompt, ChatMessage},
    checkpoint::export::{export_for_inference, tokenizer_for},
    data::loader::corpus_files_in,
    inference::{InferenceEngine, SamplingParams},
    model::{config::QuarkConfig, proj::QuantFormat},
    tokenizer::bpe::QuarkTokenizer,
    training::{
        metrics::TrainingEvent,
        start_training,
        trainer::{OffloadMode, Precision, FINE_TUNE_LR},
        TrainerConfig, TrainingMode,
    },
};

const USAGE: &str = "\
Usage:
  quark-cli tokenizer <out.json> <corpus> [--vocab N]
  quark-cli train  <preset> <corpus> --out <dir> [--tokenizer tokenizer.json]
                   [--steps N] [--batch N] [--accum N] [--seq N] [--lr X] [--bf16]
                   [--save-every N] [--eval-every N] [--keep N] [--offload auto|on|off]
                   [--finetune <base-checkpoint>]
  quark-cli export <checkpoint> <out-dir> [--q4 | --q8]
  quark-cli infer  <checkpoint> \"<prompt>\" [--max N] [--raw]
  quark-cli bench  <preset> [--steps N] [--seq N] [--offload DIR] [--layer-only]

<preset> is one of tiny, small, 1b, 3b, 7b, 10b, 20b, 30b.
<corpus> is a .txt / .jsonl file or a folder of them (searched recursively).
<checkpoint> is a .bin file or a sharded checkpoint-N/ folder.
train resumes from the newest checkpoint in --out; the tokenizer defaults to
<out>/tokenizer.json, then ~/.quark/datasets/tokenizer.json.
infer wraps the prompt in the chat template unless --raw is given.";

const VALUE_FLAGS: &[&str] = &[
    "--max", "--steps", "--seq", "--offload", "--vocab", "--out", "--tokenizer", "--batch",
    "--accum", "--lr", "--save-every", "--eval-every", "--keep", "--finetune",
];

fn preset(name: &str) -> Result<QuarkConfig> {
    Ok(match name {
        "tiny" => QuarkConfig::quark_tiny(),
        "small" => QuarkConfig::quark_small(),
        "1b" => QuarkConfig::quark_1b(),
        "3b" => QuarkConfig::quark_3b(),
        "7b" => QuarkConfig::quark_7b(),
        "10b" => QuarkConfig::quark_10b_a2b(),
        "20b" => QuarkConfig::quark_20b(),
        "30b" => QuarkConfig::quark_30b(),
        other => anyhow::bail!("unknown preset {other} (tiny, small, 1b, 3b, 7b, 10b, 20b, 30b)"),
    })
}

/// A corpus file, or every `.txt` / `.jsonl` file under a folder.
fn corpus_files(path: &str) -> Result<Vec<PathBuf>> {
    let path = PathBuf::from(path);
    let files = if path.is_dir() { corpus_files_in(&path)? } else { vec![path.clone()] };
    anyhow::ensure!(
        files.iter().all(|f| f.is_file()),
        "{} is not a file or folder of .txt / .jsonl files",
        path.display()
    );
    anyhow::ensure!(!files.is_empty(), "no .txt / .jsonl files under {}", path.display());
    Ok(files)
}

fn fmt_duration(secs: u64) -> String {
    let (h, m) = (secs / 3600, secs % 3600 / 60);
    if h > 0 { format!("{h}h{m:02}m") } else { format!("{m}m{:02}s", secs % 60) }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "quark=warn".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1));
    let positional: Vec<&String> = {
        let mut skip_next = false;
        args.iter()
            .skip(1)
            .filter(|a| {
                if skip_next {
                    skip_next = false;
                    return false;
                }
                if VALUE_FLAGS.contains(&a.as_str()) {
                    skip_next = true;
                }
                !a.starts_with("--")
            })
            .collect()
    };

    match args.first().map(String::as_str) {
        Some("tokenizer") => {
            let [out, corpus] = positional[..] else {
                anyhow::bail!("{USAGE}");
            };
            let files = corpus_files(corpus)?;
            let vocab: usize = value("--vocab").map(|v| v.parse()).transpose()?.unwrap_or(32_000);
            let bytes: u64 = files.iter().filter_map(|f| f.metadata().ok()).map(|m| m.len()).sum();
            eprintln!(
                "Training a {vocab}-token BPE tokenizer on {} file(s), {:.2} GB…",
                files.len(),
                bytes as f64 / 1e9
            );
            let t = std::time::Instant::now();
            let tok = QuarkTokenizer::train(&files, vocab, Path::new(out))?;
            eprintln!("Saved {out} ({} tokens) in {:.0}s", tok.vocab_size(), t.elapsed().as_secs_f32());
        }
        Some("train") => {
            let [preset_name, corpus] = positional[..] else {
                anyhow::bail!("{USAGE}");
            };
            let mut model = preset(preset_name)?;
            if let Some(seq) = value("--seq") {
                model.max_position_embeddings = seq.parse()?;
            }
            let files = corpus_files(corpus)?;
            let out = value("--out")
                .map(PathBuf::from)
                .unwrap_or_else(|| quark_core::paths::checkpoints_dir().join(preset_name));
            let mut config = TrainerConfig { output_dir: out.clone(), ..TrainerConfig::default() };
            let num = |name: &str| value(name).map(|v| v.parse::<u64>()).transpose();
            if let Some(v) = num("--steps")? {
                config.max_steps = v;
            }
            if let Some(v) = num("--batch")? {
                config.batch_size = v as usize;
            }
            if let Some(v) = num("--accum")? {
                config.grad_accum_steps = v as usize;
            }
            if let Some(v) = num("--save-every")? {
                config.save_every_steps = v;
            }
            if let Some(v) = num("--eval-every")? {
                config.eval_every_steps = v;
            }
            if let Some(v) = num("--keep")? {
                config.keep_checkpoints = v as usize;
            }
            if let Some(base) = value("--finetune") {
                config.mode = TrainingMode::FineTune { base_checkpoint: PathBuf::from(base) };
                config.schedule.max_lr = FINE_TUNE_LR;
                config.schedule.min_lr = FINE_TUNE_LR / 10.0;
            }
            if let Some(lr) = value("--lr") {
                config.schedule.max_lr = lr.parse()?;
                config.schedule.min_lr = config.schedule.max_lr / 10.0;
            }
            if flag("--bf16") {
                config.precision = Precision::Bf16;
            }
            config.offload = match value("--offload").map(String::as_str) {
                None | Some("auto") => OffloadMode::Auto,
                Some("on") => OffloadMode::On,
                Some("off") => OffloadMode::Off,
                Some(other) => anyhow::bail!("--offload {other}: expected auto, on or off"),
            };
            let tokenizer = value("--tokenizer").map(PathBuf::from).or_else(|| {
                let own = out.join("tokenizer.json");
                own.exists().then_some(own)
            });

            quark_core::backend::check_device()?;
            eprintln!("Backend: {}", quark_core::backend::selected().label());
            eprintln!("Corpus:  {} file(s) from {corpus}", files.len());
            eprintln!("Output:  {}", out.display());
            let (_handle, rx) = start_training(model, config, files, tokenizer);
            for event in rx {
                match event {
                    TrainingEvent::Log(line) => eprintln!("{line}"),
                    TrainingEvent::Phase(phase) => eprintln!("▸ {phase}"),
                    TrainingEvent::Metrics(m) => {
                        if m.step == 1 || m.step.is_multiple_of(10) {
                            eprintln!(
                                "step {:>6}  loss {:.4}  lr {:.2e}  |g| {:.3}  {:.0} tok/s  epoch {}  eta {}",
                                m.step,
                                m.loss,
                                m.learning_rate,
                                m.grad_norm,
                                m.tokens_per_sec,
                                m.epoch + 1,
                                fmt_duration(m.eta_secs)
                            );
                        }
                    }
                    TrainingEvent::Eval { .. } => {} // already in the trainer's log line
                    TrainingEvent::Error(e) => anyhow::bail!("{e}"),
                    TrainingEvent::Done => break,
                }
            }
        }
        Some("export") => {
            let [src, dst] = positional[..] else {
                anyhow::bail!("{USAGE}");
            };
            let quant = if flag("--q4") {
                Some(QuantFormat::Q4)
            } else if flag("--q8") {
                Some(QuantFormat::Q8)
            } else {
                None
            };
            let (src, dst) = (PathBuf::from(src), PathBuf::from(dst));
            eprintln!("Exporting {} → {} ({quant:?})…", src.display(), dst.display());
            export_for_inference(&src, &dst, quant)?;
            let size: u64 = std::fs::read_dir(&dst)?.flatten().filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum();
            eprintln!("Done: {:.2} GB", size as f64 / 1e9);
        }
        Some("infer") => {
            let [ckpt, prompt] = positional[..] else {
                anyhow::bail!("{USAGE}");
            };
            let ckpt = PathBuf::from(ckpt);
            let cfg = QuarkConfig::for_checkpoint(&ckpt).context("config.json not found for checkpoint")?;
            let tok = tokenizer_for(&ckpt).context("tokenizer.json not found for checkpoint")?;
            let max = value("--max").map(|v| v.parse()).transpose()?.unwrap_or(256);
            let t = std::time::Instant::now();
            let engine = InferenceEngine::load(&ckpt, &cfg, &tok)?;
            eprintln!(
                "Loaded in {:.1}s on {}",
                t.elapsed().as_secs_f32(),
                quark_core::backend::selected().label()
            );

            let (text, params) = if flag("--raw") {
                (prompt.clone(), SamplingParams { max_new_tokens: max, ..SamplingParams::default() })
            } else {
                let params = SamplingParams {
                    max_new_tokens: max,
                    stop_strings: default_stop_strings(),
                    ..SamplingParams::default()
                };
                (render_prompt(&[ChatMessage::user(prompt.as_str())]), params)
            };
            let (tx, rx) = std::sync::mpsc::channel::<String>();
            let t = std::time::Instant::now();
            let printer = std::thread::spawn(move || {
                let mut pieces = 0;
                for piece in rx {
                    print!("{piece}");
                    let _ = std::io::stdout().flush();
                    pieces += 1;
                }
                pieces
            });
            let out = engine.generate_streaming(&text, params, tx)?;
            let _ = printer.join();
            let secs = t.elapsed().as_secs_f32();
            println!();
            eprintln!("{} chars in {secs:.1}s", out.chars().count());
        }
        Some("bench") => {
            let [preset] = positional[..] else {
                anyhow::bail!("{USAGE}");
            };
            let mut cfg = self::preset(preset)?;
            if let Some(seq) = value("--seq") {
                cfg.max_position_embeddings = seq.parse()?;
            }
            quark_core::backend::check_device()?;
            println!("Backend: {}", quark_core::backend::selected().label());
            let steps: u64 = value("--steps").map(|v| v.parse()).transpose()?.unwrap_or(3);
            if flag("--layer-only") {
                bench::layer(&cfg)?;
            } else {
                let dir = value("--offload").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("quark-bench"));
                bench::streamed(&cfg, steps, &dir)?;
            }
        }
        _ => {
            println!("Quark LLM CLI\n{USAGE}");
        }
    }
    Ok(())
}

mod bench {
    use std::path::Path;
    use std::time::Instant;

    use anyhow::Result;
    use quark_core::{
        backend::TrainBackend,
        data::batch::collate_batch,
        memory::store::TensorStore,
        model::{block::DecoderBlock, config::QuarkConfig},
        tokenizer::bpe::PAD_ID,
        training::{
            adamw::AdamWConfig,
            optim::OptimizerKind,
            streamed::{StreamedOptim, StreamedTrainer},
        },
    };
    use burn::tensor::{backend::Backend, Distribution, Tensor};

    fn gb(bytes: u64) -> String {
        format!("{:.1} GB", bytes as f64 / 1e9)
    }

    fn rss() -> u64 {
        let mut sys = sysinfo::System::new();
        let pid = sysinfo::get_current_pid().ok();
        pid.and_then(|pid| {
            sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), false);
            sys.process(pid).map(|p| p.memory())
        })
        .unwrap_or(0)
    }

    /// One decoder layer at the preset's real shape: forward + backward
    /// (what the streamed trainer does per layer and micro-batch).
    pub fn layer(cfg: &QuarkConfig) -> Result<()> {
        let device = quark_core::backend::device();
        let seq = cfg.max_position_embeddings;
        let moe = cfg.is_moe_layer(0);
        println!(
            "Layer benchmark: hidden {}, seq {seq}, {} (preset {:.2}B params, {:.2}B active)",
            cfg.hidden_size,
            if moe { format!("MoE {} experts top-{}", cfg.num_experts, cfg.num_experts_per_tok) } else { "dense".into() },
            cfg.param_count() as f64 / 1e9,
            cfg.active_param_count() as f64 / 1e9,
        );
        let before = rss();
        let t = Instant::now();
        let layer = DecoderBlock::<TrainBackend>::new(cfg, moe, &device);
        let stage = quark_core::memory::stage::module_to_stage(&layer)?; // materialise
        println!("  init: {:.1}s, {} of weights", t.elapsed().as_secs_f32(), gb(stage.bytes()));

        let x = Tensor::<TrainBackend, 3>::random([1, seq, cfg.hidden_size], Distribution::Normal(0.0, 1.0), &device).require_grad();
        let pass = || {
            let t = Instant::now();
            let (y, _) = layer.forward_with_aux(x.clone(), true);
            let _ = y.clone().sum().into_scalar();
            let fwd = t.elapsed().as_secs_f32();
            let t = Instant::now();
            let grads = y.sum().backward();
            let _ = x.grad(&grads).map(|g| g.sum().into_scalar());
            (fwd, t.elapsed().as_secs_f32())
        };
        // The first pass compiles and autotunes the GPU kernels; time the next ones.
        let (fwd0, bwd0) = pass();
        println!("  warm-up (kernel compilation): forward {fwd0:.2}s, backward {bwd0:.2}s");
        const RUNS: usize = 3;
        let (mut fwd, mut bwd) = (0.0, 0.0);
        for _ in 0..RUNS {
            let (f, b) = pass();
            fwd += f / RUNS as f32;
            bwd += b / RUNS as f32;
        }
        let peak = rss().saturating_sub(before);

        // Matmul FLOPs of one forward pass (active experts only, causal attention).
        let (h, s) = (cfg.hidden_size as f64, seq as f64);
        let head_dim = h / cfg.num_attention_heads as f64;
        let kv = cfg.num_key_value_heads as f64 * head_dim;
        let ffn = 3.0 * h * cfg.intermediate_size as f64;
        let ffn = if moe { cfg.num_experts_per_tok as f64 * ffn } else { ffn };
        let flops = 2.0 * s * (2.0 * h * h + 2.0 * h * kv + ffn) + 2.0 * s * s * h;
        let tflops = |flops: f64, secs: f32| flops / secs as f64 / 1e12;
        println!(
            "  forward {fwd:.3}s ({:.2} TFLOP/s), backward {bwd:.3}s ({:.2} TFLOP/s), process memory +{}",
            tflops(flops, fwd),
            tflops(2.0 * flops, bwd),
            gb(peak)
        );
        let per_layer = fwd * 2.0 + bwd; // forward, recompute forward, backward
        println!(
            "  ⇒ compute per micro-batch of {seq} tokens through all {} layers: ≈{:.0}s (plus disk I/O)",
            cfg.num_hidden_layers,
            per_layer * cfg.num_hidden_layers as f32
        );
        Ok(())
    }

    /// Full streamed training steps on synthetic data.
    pub fn streamed(cfg: &QuarkConfig, steps: u64, dir: &Path) -> Result<()> {
        let device = quark_core::backend::device();
        <TrainBackend as Backend>::seed(&device, 0);
        let _ = std::fs::remove_dir_all(dir);
        let budget = quark_core::memory::budget::HardwareBudget::detect();
        let ram = (budget.ram_free_bytes / 2).max(1 << 30);
        let store = TensorStore::new(dir.join("weights"), ram / 2)?;
        let acts = TensorStore::new(dir.join("activations"), ram / 4)?;
        println!(
            "Streamed benchmark: {:.2}B params, seq {}, offload {} (RAM cache {})",
            cfg.param_count() as f64 / 1e9,
            cfg.max_position_embeddings,
            dir.display(),
            gb(ram / 2)
        );
        let t = Instant::now();
        let optim = StreamedOptim { kind: OptimizerKind::AdamWCompact, hyper: AdamWConfig::default(), max_grad_norm: 1.0 };
        let mut trainer = StreamedTrainer::<TrainBackend>::new(cfg.clone(), store, acts, device, optim, 0)?;
        println!("  initialised all layers in {:.0}s", t.elapsed().as_secs_f32());

        let seq = cfg.max_position_embeddings;
        let batch = collate_batch(
            vec![(0..seq).map(|t| (t * 7 % (cfg.vocab_size - 4) + 4) as u32).collect()],
            PAD_ID,
        );
        for _ in 0..steps {
            let io0 = io(&trainer);
            let t = Instant::now();
            let stats = trainer.train_step(std::slice::from_ref(&batch), 1e-4)?;
            println!(
                "  step {}: loss {:.3}, {:.1}s, disk {} , RSS {}",
                trainer.step,
                stats.loss,
                t.elapsed().as_secs_f32(),
                gb(io(&trainer) - io0),
                gb(rss())
            );
        }
        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }

    fn io(trainer: &StreamedTrainer<TrainBackend>) -> u64 {
        use std::sync::atomic::Ordering::Relaxed;
        let s = trainer.store().stats();
        s.read_bytes.load(Relaxed) + s.write_bytes.load(Relaxed)
    }
}
