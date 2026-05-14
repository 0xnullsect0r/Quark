//! HuggingFace dataset downloader.
//!
//! Streams rows from curated HuggingFace datasets into JSONL files using a
//! bundled Python helper script (`hf_download.py`).  Progress and log lines
//! flow through an [`mpsc`] channel so the GUI can render a live scrolling log
//! and progress bar.
//!
//! The Python script is embedded at compile time via [`include_str!`] so the
//! application is fully self-contained — no external scripts are needed.

#![allow(dead_code)]

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

// ─── Embedded Python script ───────────────────────────────────────────────────

const HF_DOWNLOAD_PY: &str = include_str!("../../scripts/hf_download.py");

// ─── Public message type ──────────────────────────────────────────────────────

/// Messages sent from the background download thread to the GUI.
#[derive(Debug, Clone)]
pub enum HfMessage {
    /// A human-readable log line.
    Log(String),
    /// Overall download progress in \[0.0, 1.0\].
    Progress(f32),
    /// Short description of the current phase for the status bar.
    Phase(String),
    /// Current download speed in bytes per second.
    Speed(f32),
    /// Cumulative bytes downloaded and estimated total bytes.
    ByteProgress { downloaded: u64, total: u64 },
    /// Basename of the file currently being downloaded.
    CurrentFile(String),
    /// All selected datasets downloaded successfully.
    Done,
    /// Download failed; contains a human-readable description.
    Error(String),
}

// ─── Dataset catalogue ────────────────────────────────────────────────────────

/// Category tag used to group datasets in the GUI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HfDatasetCategory {
    Code,
    Knowledge,
    Instructions,
}

impl HfDatasetCategory {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Code => "🖥  Code",
            Self::Knowledge => "📚  Knowledge",
            Self::Instructions => "💬  Instructions",
        }
    }
}

/// A dataset entry in the curated catalogue.
#[derive(Debug, Clone)]
pub struct HfDataset {
    /// Short stable identifier used in `HfConfig::selected_ids`.
    pub id: &'static str,
    /// HuggingFace repository / dataset ID.
    pub hf_id: &'static str,
    /// Human-readable label shown in the GUI.
    pub label: &'static str,
    pub category: HfDatasetCategory,
    /// Approximate uncompressed size in GiB (used for display and planning).
    pub approx_size_gib: f32,
    /// `true` if downloading requires a valid HuggingFace API token.
    pub hf_token_required: bool,
    /// Optional dataset subset / configuration name (e.g. `"20231101.en"`).
    pub subset: Option<&'static str>,
    /// Dataset split to stream (usually `"train"`).
    pub split: &'static str,
    /// Override the field name extracted as text.  `None` = auto-detect.
    pub text_field: Option<&'static str>,
}

/// Return the full curated dataset catalogue.
pub fn hf_datasets() -> Vec<HfDataset> {
    vec![
        // ── Code ──────────────────────────────────────────────────────────
        HfDataset {
            id: "github_code",
            hf_id: "codeparrot/github-code-clean",
            label: "GitHub Code — 24 languages, deduplicated",
            category: HfDatasetCategory::Code,
            approx_size_gib: 115.0,
            hf_token_required: false,
            subset: None,
            split: "train",
            text_field: Some("code"),
        },
        HfDataset {
            id: "openwebmath",
            hf_id: "open-web-math/open-web-math",
            label: "OpenWebMath — math from the web",
            category: HfDatasetCategory::Code,
            approx_size_gib: 14.0,
            hf_token_required: false,
            subset: None,
            split: "train",
            text_field: Some("text"),
        },
        HfDataset {
            id: "code_feedback",
            hf_id: "m-a-p/CodeFeedback-Filtered-Instruction",
            label: "Code Feedback — filtered code instruction pairs",
            category: HfDatasetCategory::Code,
            approx_size_gib: 1.0,
            hf_token_required: false,
            subset: None,
            split: "train",
            text_field: Some("answer"),
        },
        HfDataset {
            id: "the_stack_smol",
            hf_id: "bigcode/the-stack-smol",
            label: "The Stack Smol — code, deduplicated  🔑 token required",
            category: HfDatasetCategory::Code,
            approx_size_gib: 35.0,
            hf_token_required: true,
            subset: None,
            split: "train",
            text_field: Some("content"),
        },
        // ── Knowledge ─────────────────────────────────────────────────────
        HfDataset {
            id: "wikipedia_en",
            hf_id: "wikimedia/wikipedia",
            label: "Wikipedia English (2023-11-01)",
            category: HfDatasetCategory::Knowledge,
            approx_size_gib: 20.0,
            hf_token_required: false,
            subset: Some("20231101.en"),
            split: "train",
            text_field: Some("text"),
        },
        HfDataset {
            id: "openwebtext",
            hf_id: "Skylion007/openwebtext",
            label: "OpenWebText — Common Crawl filtered web text (Pile-CC equivalent)",
            category: HfDatasetCategory::Knowledge,
            approx_size_gib: 40.0,
            hf_token_required: false,
            subset: None,
            split: "train",
            text_field: Some("text"),
        },
        HfDataset {
            id: "scientific_papers",
            hf_id: "allenai/peS2o",
            label: "peS2o — ArXiv + Semantic Scholar papers (covers arxiv & PubMed)",
            category: HfDatasetCategory::Knowledge,
            approx_size_gib: 70.0,
            hf_token_required: false,
            subset: None,
            split: "train",
            text_field: Some("text"),
        },
        HfDataset {
            id: "gutenberg_pg19",
            hf_id: "deepmind/pg19",
            label: "PG-19 — Project Gutenberg books (pre-1919 public domain)",
            category: HfDatasetCategory::Knowledge,
            approx_size_gib: 11.0,
            hf_token_required: false,
            subset: None,
            split: "train",
            text_field: Some("text"),
        },
        HfDataset {
            id: "stackexchange",
            hf_id: "HuggingFaceH4/stack-exchange-preferences",
            label: "Stack Exchange Q&A — voted answers",
            category: HfDatasetCategory::Knowledge,
            approx_size_gib: 2.0,
            hf_token_required: false,
            subset: None,
            split: "train",
            text_field: None,
        },
        HfDataset {
            id: "legal",
            hf_id: "pile-of-law/pile-of-law",
            label: "Pile of Law — court opinions & legal documents (FreeLaw equivalent)",
            category: HfDatasetCategory::Knowledge,
            approx_size_gib: 50.0,
            hf_token_required: false,
            subset: None,
            split: "train",
            text_field: Some("text"),
        },
        HfDataset {
            id: "patents",
            hf_id: "HUPD/hupd",
            label: "HUPD — Harvard USPTO Patent Dataset (USPTO equivalent)",
            category: HfDatasetCategory::Knowledge,
            approx_size_gib: 50.0,
            hf_token_required: false,
            subset: None,
            split: "train",
            text_field: Some("abstract"),
        },
        HfDataset {
            id: "slimpajama",
            hf_id: "cerebras/SlimPajama-627B",
            label: "SlimPajama — diverse deduplicated web text (stream subset)",
            category: HfDatasetCategory::Knowledge,
            approx_size_gib: 627.0,
            hf_token_required: false,
            subset: None,
            split: "train",
            text_field: Some("text"),
        },
        HfDataset {
            id: "c4_en",
            hf_id: "allenai/c4",
            label: "C4 English — cleaned Common Crawl (stream subset)",
            category: HfDatasetCategory::Knowledge,
            approx_size_gib: 750.0,
            hf_token_required: false,
            subset: Some("en"),
            split: "train",
            text_field: Some("text"),
        },
        // ── Instructions ──────────────────────────────────────────────────
        HfDataset {
            id: "ultrachat",
            hf_id: "HuggingFaceH4/ultrachat_200k",
            label: "UltraChat 200K — multi-turn instruction following",
            category: HfDatasetCategory::Instructions,
            approx_size_gib: 1.0,
            hf_token_required: false,
            subset: None,
            split: "train_sft",
            text_field: Some("messages"),
        },
        HfDataset {
            id: "openhermes",
            hf_id: "teknium/OpenHermes-2.5",
            label: "OpenHermes 2.5 — diverse instruction/chat pairs",
            category: HfDatasetCategory::Instructions,
            approx_size_gib: 1.0,
            hf_token_required: false,
            subset: None,
            split: "train",
            text_field: Some("conversations"),
        },
    ]
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a HuggingFace download job.
#[derive(Debug, Clone)]
pub struct HfConfig {
    /// Directory where the venv and JSONL output files are stored.
    pub target_dir: PathBuf,
    /// Python executable to use (`"python3"`, `"python"`, …).
    pub python_cmd: String,
    /// Ids from `hf_datasets()` to download.
    pub selected_ids: Vec<String>,
    /// Maximum GB to download per dataset (0 = unlimited / stream cap).
    pub max_gb_per_dataset: f32,
    /// Number of parallel HTTP connections per file (1–20, default 10).
    pub parallel_workers: u8,
    /// Optional HuggingFace API token for gated datasets.
    /// Not persisted to disk.
    pub hf_token: String,
}

impl Default for HfConfig {
    fn default() -> Self {
        Self {
            target_dir: crate::paths::app_data_dir(),
            python_cmd: detect_python(),
            selected_ids: vec![
                "github_code".into(),
                "wikipedia_en".into(),
                "scientific_papers".into(),
                "ultrachat".into(),
                "openhermes".into(),
            ],
            max_gb_per_dataset: 10.0,
            parallel_workers: 10,
            hf_token: String::new(),
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Spawn the download pipeline in a background thread.
///
/// Returns a [`Receiver`] that yields [`HfMessage`]s.  The channel closes when
/// the thread finishes (either [`HfMessage::Done`] or [`HfMessage::Error`]).
pub fn start_hf_build(config: HfConfig) -> Receiver<HfMessage> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || run_pipeline(config, tx));
    rx
}

/// Detect the best Python executable name on the current PATH.
pub fn detect_python() -> String {
    for candidate in ["python3", "python", "python3.12", "python3.11", "python3.10"] {
        if cmd_exists(candidate) {
            return candidate.to_owned();
        }
    }
    "python3".to_owned()
}

// ─── Pipeline ─────────────────────────────────────────────────────────────────

fn run_pipeline(cfg: HfConfig, tx: Sender<HfMessage>) {
    macro_rules! log {
        ($($t:tt)*) => { let _ = tx.send(HfMessage::Log(format!($($t)*))); };
    }
    macro_rules! phase {
        ($p:expr, $($t:tt)*) => {
            let _ = tx.send(HfMessage::Phase(format!($($t)*)));
            let _ = tx.send(HfMessage::Progress($p));
        };
    }
    macro_rules! bail {
        ($($t:tt)*) => {{
            let msg = format!($($t)*);
            log!("❌  {}", msg);
            let _ = tx.send(HfMessage::Error(msg));
            return;
        }};
    }

    // ── 1. Prerequisites ──────────────────────────────────────────────────
    phase!(0.01, "Checking prerequisites…");
    log!("Looking for {} …", cfg.python_cmd);
    if !cmd_exists(&cfg.python_cmd) {
        bail!("'{}' not found on PATH. Install Python 3 and try again.", cfg.python_cmd);
    }
    log!("✔  {} OK", cfg.python_cmd);

    if cfg.selected_ids.is_empty() {
        bail!("No datasets selected.");
    }

    // ── 2. Create / reuse virtual environment ────────────────────────────
    let venv_dir = cfg.target_dir.join("hf-venv");
    #[cfg(target_os = "windows")]
    let venv_python = venv_dir.join("Scripts").join("python.exe");
    #[cfg(not(target_os = "windows"))]
    let venv_python = venv_dir.join("bin").join("python");

    if venv_dir.exists() {
        log!("ℹ  Virtual environment already at {}; reusing.", venv_dir.display());
    } else {
        phase!(0.03, "Creating Python virtual environment…");
        log!("Creating venv at {} …", venv_dir.display());
        if let Err(e) = std::fs::create_dir_all(&cfg.target_dir) {
            bail!("Cannot create target directory: {e}");
        }
        let ok = stream_command(
            Command::new(&cfg.python_cmd)
                .args(["-m", "venv", venv_dir.to_str().unwrap_or("hf-venv")]),
            &tx,
            0.03,
            0.06,
        );
        if !ok {
            bail!("python -m venv failed — see log above.");
        }
        log!("✔  Virtual environment created.");
    }

    // ── 3. Install Python deps ────────────────────────────────────────────
    const DEPS_VERSION: &str = "hf-v1";
    let deps_stamp = venv_dir.join(".quark-hf-deps");
    let stamp_ok = deps_stamp.exists()
        && std::fs::read_to_string(&deps_stamp).unwrap_or_default().trim() == DEPS_VERSION;

    if stamp_ok {
        log!("ℹ  Dependencies already installed; skipping.");
    } else {
        phase!(0.06, "Installing Python dependencies…");
        log!("pip install datasets huggingface_hub tqdm …");
        let ok = stream_command(
            Command::new(&venv_python).args([
                "-m",
                "pip",
                "install",
                "--upgrade",
                "datasets",
                "huggingface_hub",
                "tqdm",
            ]),
            &tx,
            0.06,
            0.10,
        );
        if !ok {
            bail!("pip install failed — see log above.");
        }
        log!("✔  Python dependencies installed.");
        let _ = std::fs::write(&deps_stamp, DEPS_VERSION);
    }

    // ── 4. Write bundled Python script to disk ────────────────────────────
    let script_path = cfg.target_dir.join("hf_download.py");
    if let Err(e) = std::fs::write(&script_path, HF_DOWNLOAD_PY) {
        bail!("Could not write hf_download.py: {e}");
    }
    log!("ℹ  Script written to {}.", script_path.display());

    // ── 5. Download each selected dataset ─────────────────────────────────
    let catalogue = hf_datasets();
    let selected: Vec<&HfDataset> = cfg
        .selected_ids
        .iter()
        .filter_map(|id| catalogue.iter().find(|d| d.id == id.as_str()))
        .collect();

    let n = selected.len();
    let band_start = 0.10_f32;
    let band_end = 0.99_f32;
    let step = (band_end - band_start) / n as f32;
    let out_dir = cfg.target_dir.join("datasets");
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        bail!("Cannot create datasets directory: {e}");
    }

    for (idx, ds) in selected.iter().enumerate() {
        let p_start = band_start + idx as f32 * step;
        let p_end = p_start + step;

        phase!(
            p_start,
            "Downloading {}/{}: {}…",
            idx + 1,
            n,
            ds.label
        );
        log!(
            "━━  Dataset {}/{}: {} ({})",
            idx + 1,
            n,
            ds.label,
            ds.hf_id
        );

        if ds.hf_token_required && cfg.hf_token.is_empty() {
            log!("⚠  '{}' requires a HuggingFace token. Skipping.", ds.hf_id);
            log!("   Enter your token in the Dataset panel and re-run.");
            continue;
        }

        let mut cmd = Command::new(&venv_python);
        cmd.arg(script_path.to_str().unwrap_or("hf_download.py"));
        cmd.args(["--dataset-id", ds.hf_id]);
        cmd.args(["--output-dir", out_dir.to_str().unwrap_or("datasets")]);
        cmd.args(["--split", ds.split]);
        if cfg.max_gb_per_dataset > 0.0 {
            cmd.args(["--max-gb", &format!("{:.2}", cfg.max_gb_per_dataset)]);
        }
        if let Some(subset) = ds.subset {
            cmd.args(["--subset", subset]);
        }
        if let Some(field) = ds.text_field {
            cmd.args(["--text-field", field]);
        }
        if !cfg.hf_token.is_empty() {
            cmd.args(["--hf-token", &cfg.hf_token]);
        }
        cmd.args(["--workers", &cfg.parallel_workers.to_string()]);

        let ok = stream_command(&mut cmd, &tx, p_start, p_end);
        if !ok {
            log!("⚠  '{}' failed — skipping and continuing with next dataset.", ds.hf_id);
        } else {
            log!("✔  '{}' done ({}/{}).", ds.label, idx + 1, n);
        }
    }

    // ── Done ─────────────────────────────────────────────────────────────
    phase!(1.0, "Complete!");
    log!("✅  Download complete.  JSONL files are in: {}", out_dir.display());
    let _ = tx.send(HfMessage::Done);
}

// ─── Subprocess helpers ───────────────────────────────────────────────────────

/// Spawn `cmd`, stream its stdout + stderr to the channel, and return whether
/// it exited with status 0.
///
/// Lines from `hf_download.py` beginning with `PROGRESS:` are parsed as
/// floating-point values and mapped into the `[p_start, p_end]` range.
/// Lines beginning with `LOG:` are forwarded as log messages.
/// All other non-empty lines are also forwarded as log messages.
fn stream_command(cmd: &mut Command, tx: &Sender<HfMessage>, p_start: f32, p_end: f32) -> bool {
    let mut child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(HfMessage::Log(format!("Failed to spawn: {e}")));
            return false;
        }
    };

    let err_tx = tx.clone();
    let stderr = child.stderr.take().unwrap();
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let _ = err_tx.send(HfMessage::Log(format!("[err] {line}")));
        }
    });

    let stdout = child.stdout.take().unwrap();
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        if let Some(rest) = line.strip_prefix("PROGRESS:") {
            if let Ok(p) = rest.trim().parse::<f32>() {
                let mapped = p_start + p * (p_end - p_start);
                let _ = tx.send(HfMessage::Progress(mapped));
            }
        } else if let Some(rest) = line.strip_prefix("SPEED:") {
            if let Ok(bps) = rest.trim().parse::<f32>() {
                let _ = tx.send(HfMessage::Speed(bps));
            }
        } else if let Some(rest) = line.strip_prefix("BYTES:") {
            if let Some((d, t)) = rest.trim().split_once('/') {
                if let (Ok(dl), Ok(tot)) = (d.parse::<u64>(), t.parse::<u64>()) {
                    let _ = tx.send(HfMessage::ByteProgress { downloaded: dl, total: tot });
                }
            }
        } else if let Some(rest) = line.strip_prefix("FILE:") {
            let _ = tx.send(HfMessage::CurrentFile(rest.trim().to_owned()));
        } else if let Some(rest) = line.strip_prefix("LOG:") {
            let _ = tx.send(HfMessage::Log(rest.to_owned()));
        } else if line.trim() == "DONE" {
            // handled by process exit code
        } else if !line.trim().is_empty() {
            let _ = tx.send(HfMessage::Log(line));
        }
    }

    child.wait().map(|s| s.success()).unwrap_or(false)
}

/// Returns `true` if `name` resolves to an executable on PATH.
fn cmd_exists(name: &str) -> bool {
    let path_var = std::env::var("PATH").unwrap_or_default();
    std::env::split_paths(&path_var).any(|dir| {
        let p = dir.join(name);
        if p.is_file() {
            return true;
        }
        #[cfg(windows)]
        if dir.join(format!("{name}.exe")).is_file() {
            return true;
        }
        false
    })
}
