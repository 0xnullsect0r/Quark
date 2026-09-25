//! End-to-end: train a tiny model, reload the checkpoint for inference, then
//! resume training from it.

use std::path::{Path, PathBuf};
use std::time::Duration;

use quark_core::{
    inference::{InferenceEngine, SamplingParams},
    model::config::QuarkConfig,
    tokenizer::bpe::QuarkTokenizer,
    training::{
        lr_schedule::CosineSchedule,
        metrics::{MetricsReceiver, TrainingEvent, TrainingMetrics},
        start_training, TrainerConfig,
    },
};

fn tiny_config() -> QuarkConfig {
    QuarkConfig {
        vocab_size: 0, // replaced with the tokenizer's vocab size by the trainer
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        intermediate_size: 64,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        num_experts: 4,
        num_experts_per_tok: 2,
        num_moe_layers: 1,
        moe_layer_freq: 2,
        tie_word_embeddings: false,
    }
}

fn trainer_config(output_dir: &Path, max_steps: u64) -> TrainerConfig {
    TrainerConfig {
        output_dir: output_dir.to_path_buf(),
        max_steps,
        batch_size: 2,
        grad_accum_steps: 2,
        save_every_steps: 10,
        eval_every_steps: 10,
        schedule: CosineSchedule { warmup_steps: 2, max_steps: 60, max_lr: 3e-3, min_lr: 3e-4 },
        ..TrainerConfig::default()
    }
}

struct Run {
    metrics: Vec<TrainingMetrics>,
    evals: Vec<(u64, f32)>,
    logs: Vec<String>,
}

fn collect(rx: MetricsReceiver) -> Run {
    let mut run = Run { metrics: vec![], evals: vec![], logs: vec![] };
    loop {
        match rx.recv_timeout(Duration::from_secs(600)).expect("training timed out") {
            TrainingEvent::Metrics(m) => run.metrics.push(m),
            TrainingEvent::Eval { step, loss } => run.evals.push((step, loss)),
            TrainingEvent::Log(l) => run.logs.push(l),
            TrainingEvent::Phase(_) => {}
            TrainingEvent::Done => return run,
            TrainingEvent::Error(e) => panic!("training failed: {e}"),
        }
    }
}

fn setup(dir: &Path) -> (PathBuf, PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let corpus = dir.join("corpus.txt");
    let text: String = (0..400)
        .map(|i| format!("fn add_{i}(a: u32, b: u32) -> u32 {{ a + b }}\n"))
        .collect();
    std::fs::write(&corpus, text).unwrap();
    let tokenizer = dir.join("tok.json");
    QuarkTokenizer::train(std::slice::from_ref(&corpus), 400, &tokenizer).unwrap();
    (corpus, tokenizer)
}

#[test]
fn train_load_generate_and_resume() {
    let dir = std::env::temp_dir().join(format!("quark-e2e-{}", std::process::id()));
    let (corpus, tokenizer) = setup(&dir);
    let out = dir.join("ckpt");

    // ── Train ────────────────────────────────────────────────────────────────
    let (_handle, rx) = start_training(
        tiny_config(),
        trainer_config(&out, 30),
        vec![corpus.clone()],
        Some(tokenizer.clone()),
    );
    let run = collect(rx);

    assert_eq!(run.metrics.len(), 30);
    assert_eq!(run.metrics.last().unwrap().step, 30);
    let first = run.metrics[0].loss;
    let last = run.metrics[25..].iter().map(|m| m.loss).sum::<f32>() / 5.0;
    assert!(last < first * 0.7, "loss did not decrease: {first} → {last}");
    assert!(run.metrics.iter().all(|m| m.grad_norm.is_finite() && m.grad_norm > 0.0));
    assert!(!run.evals.is_empty(), "no eval events");
    assert!(run.evals.iter().all(|(_, l)| l.is_finite()));

    for f in ["config.json", "tokenizer.json", "checkpoint-10.bin", "checkpoint-30.bin"] {
        assert!(out.join(f).exists(), "missing {f}");
    }

    // ── Load for inference ───────────────────────────────────────────────────
    let ckpt = out.join("checkpoint-30.bin");
    let config = QuarkConfig::for_checkpoint(&ckpt).expect("config.json next to checkpoint");
    let engine = InferenceEngine::load(&ckpt, &config, &out.join("tokenizer.json")).unwrap();
    let params = SamplingParams { max_new_tokens: 40, ..SamplingParams::default() };

    let (tx, rx) = std::sync::mpsc::channel();
    let text = engine.generate_streaming("fn add_7(", params, tx).unwrap();
    let streamed: String = rx.try_iter().collect();
    assert_eq!(streamed, text, "streamed pieces must add up to the full response");

    // ── Resume ───────────────────────────────────────────────────────────────
    let (_handle, rx) =
        start_training(tiny_config(), trainer_config(&out, 35), vec![corpus], Some(tokenizer));
    let resumed = collect(rx);
    assert!(resumed.logs.iter().any(|l| l.contains("Resumed")), "{:#?}", resumed.logs);
    assert_eq!(resumed.metrics.first().unwrap().step, 31);
    assert_eq!(resumed.metrics.len(), 5);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn crash_is_reported_not_hung() {
    let dir = std::env::temp_dir().join(format!("quark-e2e-crash-{}", std::process::id()));
    let (corpus, tokenizer) = setup(&dir);
    // 4 query heads cannot be grouped over 3 KV heads: the forward pass panics.
    let bad = QuarkConfig { num_key_value_heads: 3, ..tiny_config() };
    let (_handle, rx) =
        start_training(bad, trainer_config(&dir.join("ckpt"), 5), vec![corpus], Some(tokenizer));
    let error = loop {
        match rx.recv_timeout(Duration::from_secs(120)).expect("no event: training hung") {
            TrainingEvent::Error(e) => break e,
            TrainingEvent::Done => panic!("expected a crash"),
            _ => {}
        }
    };
    assert!(error.starts_with("Training crashed"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}
