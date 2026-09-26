//! Headless Quark commands.
//!
//! ```text
//! quark-cli export <checkpoint> <out-dir> [--q4 | --q8]   inference export (optionally quantized)
//! quark-cli infer  <checkpoint> "<prompt>" [--max N]       generate text
//! quark-cli bench  <tiny|small|10b> [--steps N] [--seq N] [--offload DIR] [--layer-only]
//! ```
//!
//! `<checkpoint>` is a `.bin` file or a sharded `checkpoint-N/` folder.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use quark_core::{
    chat::{default_stop_strings, render_prompt, ChatMessage},
    checkpoint::export::{export_for_inference, tokenizer_for},
    inference::{InferenceEngine, SamplingParams},
    model::{config::QuarkConfig, proj::QuantFormat},
};

const USAGE: &str = "\
Usage:
  quark-cli export <checkpoint> <out-dir> [--q4 | --q8]
  quark-cli infer  <checkpoint> \"<prompt>\" [--max N] [--raw]
  quark-cli bench  <tiny|small|10b> [--steps N] [--seq N] [--offload DIR] [--layer-only]

<checkpoint> is a .bin file or a sharded checkpoint-N/ folder.
infer wraps the prompt in the chat template unless --raw is given.";

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
                if ["--max", "--steps", "--seq", "--offload"].contains(&a.as_str()) {
                    skip_next = true;
                }
                !a.starts_with("--")
            })
            .collect()
    };

    match args.first().map(String::as_str) {
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
            eprintln!("Loaded in {:.1}s", t.elapsed().as_secs_f32());

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
            let mut cfg = match preset.as_str() {
                "tiny" => QuarkConfig::quark_tiny(),
                "small" => QuarkConfig::quark_small(),
                "10b" => QuarkConfig::quark_10b_a2b(),
                other => anyhow::bail!("unknown preset {other} (tiny, small, 10b)"),
            };
            if let Some(seq) = value("--seq") {
                cfg.max_position_embeddings = seq.parse()?;
            }
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
        let device = burn::tensor::Device::<TrainBackend>::default();
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
        let t = Instant::now();
        let (y, _) = layer.forward_with_aux(x.clone(), true);
        let _ = y.clone().sum().into_scalar();
        let fwd = t.elapsed().as_secs_f32();
        let t = Instant::now();
        let grads = y.sum().backward();
        let _ = x.grad(&grads);
        let bwd = t.elapsed().as_secs_f32();
        let peak = rss().saturating_sub(before);
        let per_layer = fwd * 2.0 + bwd; // forward, recompute forward, backward
        println!("  forward {fwd:.2}s, backward {bwd:.2}s, process memory +{}", gb(peak));
        println!(
            "  ⇒ compute per micro-batch of {seq} tokens through all {} layers: ≈{:.0}s (plus disk I/O)",
            cfg.num_hidden_layers,
            per_layer * cfg.num_hidden_layers as f32
        );
        Ok(())
    }

    /// Full streamed training steps on synthetic data.
    pub fn streamed(cfg: &QuarkConfig, steps: u64, dir: &Path) -> Result<()> {
        let device = burn::tensor::Device::<TrainBackend>::default();
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
