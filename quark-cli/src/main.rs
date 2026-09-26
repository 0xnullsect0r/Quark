//! Headless Quark commands.
//!
//! ```text
//! quark-cli export <checkpoint> <out-dir> [--q4 | --q8]   inference export (optionally quantized)
//! quark-cli infer  <checkpoint> "<prompt>" [--max N]       generate text
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
                if *a == "--max" {
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
        _ => {
            println!("Quark LLM CLI\n{USAGE}");
        }
    }
    Ok(())
}
