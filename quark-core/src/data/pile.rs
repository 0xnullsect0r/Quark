//! The Pile dataset downloader / builder.
//!
//! Clones `EleutherAI/the-pile`, installs its Python dependencies, and runs
//! the pile replication script — all in a background thread.  Progress and
//! log lines flow through an [`mpsc`] channel so the GUI can render a live
//! scrolling log and progress bar.

#![allow(dead_code)]

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

// ─── Public message type ──────────────────────────────────────────────────────

/// Messages sent from the background build thread to the GUI.
#[derive(Debug, Clone)]
pub enum PileMessage {
    /// A human-readable log line (subprocess stdout / stderr or status text).
    Log(String),
    /// Overall build progress in \[0.0, 1.0\].
    Progress(f32),
    /// Short description of the current phase shown in the status bar.
    Phase(String),
    /// Build completed successfully.
    Done,
    /// Build failed; the string contains a human-readable description.
    Error(String),
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a Pile build job.
#[derive(Debug, Clone)]
pub struct PileConfig {
    /// Directory where the repo will be cloned and data stored.
    pub target_dir: PathBuf,
    /// Python executable to use (`"python3"`, `"python"`, …).
    pub python_cmd: String,
    /// One or more component ids passed to `pile.py --using`.
    /// Use `["pile_reprod"]` to download the full Pile in a single pass.
    /// Specify individual component ids to download a subset; the pipeline
    /// will invoke `pile.py` once per component in sequence.
    pub components: Vec<String>,
    /// `--interleave_output` argument (default 30).
    pub interleave_output: u32,
    /// If `true`, skip the `git clone` step when the repo directory already
    /// exists.
    pub skip_clone_if_exists: bool,
}

impl Default for PileConfig {
    fn default() -> Self {
        Self {
            target_dir: crate::paths::app_data_dir(),
            python_cmd: detect_python(),
            components: vec!["pile_reprod".into()],
            interleave_output: 30,
            skip_clone_if_exists: true,
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Spawn the pile build pipeline in a background thread.
///
/// Returns a [`Receiver`] that yields [`PileMessage`]s.  The channel is
/// dropped when the thread finishes (either [`PileMessage::Done`] or
/// [`PileMessage::Error`]).
pub fn start_pile_build(config: PileConfig) -> Receiver<PileMessage> {
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

// ─── Component catalogue ─────────────────────────────────────────────────────

/// A known Pile component selectable by the user.
#[derive(Debug, Clone, PartialEq)]
pub struct PileComponent {
    pub id: &'static str,
    pub label: &'static str,
    pub approx_size_gib: f32,
}

/// Return the list of all known Pile components.
pub fn pile_components() -> Vec<PileComponent> {
    vec![
        PileComponent { id: "pile_reprod",     label: "Full Pile (all components)",  approx_size_gib: 825.0 },
        PileComponent { id: "pile_cc",          label: "Pile-CC (Common Crawl)",      approx_size_gib: 227.0 },
        PileComponent { id: "pubmed_central",   label: "PubMed Central",              approx_size_gib:  90.0 },
        PileComponent { id: "books3",           label: "Books3",                      approx_size_gib: 101.0 },
        PileComponent { id: "openwebtext2",     label: "OpenWebText2",                approx_size_gib:  63.0 },
        PileComponent { id: "arxiv",            label: "ArXiv",                       approx_size_gib:  56.0 },
        PileComponent { id: "github",           label: "GitHub Code",                 approx_size_gib:  95.0 },
        PileComponent { id: "freelaw",          label: "FreeLaw",                     approx_size_gib:  51.0 },
        PileComponent { id: "stackexchange",    label: "StackExchange",               approx_size_gib:  32.0 },
        PileComponent { id: "uspto",            label: "USPTO Backgrounds",           approx_size_gib:  23.0 },
        PileComponent { id: "pubmed_abstracts", label: "PubMed Abstracts",            approx_size_gib:  19.0 },
        PileComponent { id: "gutenberg",        label: "Gutenberg (PG-19)",           approx_size_gib:  11.0 },
        PileComponent { id: "opensubtitles",    label: "OpenSubtitles",               approx_size_gib:  13.0 },
        PileComponent { id: "wikipedia_en",     label: "Wikipedia (en)",              approx_size_gib:   6.0 },
        PileComponent { id: "dm_mathematics",   label: "DM Mathematics",              approx_size_gib:   8.0 },
        PileComponent { id: "ubuntu_irc",       label: "Ubuntu IRC",                  approx_size_gib:   6.0 },
        PileComponent { id: "hackerNews",       label: "HackerNews",                  approx_size_gib:   4.0 },
        PileComponent { id: "enron_emails",     label: "Enron Emails",                approx_size_gib:   1.0 },
    ]
}

// ─── Pipeline ─────────────────────────────────────────────────────────────────

fn run_pipeline(cfg: PileConfig, tx: Sender<PileMessage>) {
    // Convenience macros so every send failure is silently ignored (GUI might
    // have closed the window).
    macro_rules! log {
        ($($t:tt)*) => { let _ = tx.send(PileMessage::Log(format!($($t)*))); };
    }
    macro_rules! phase {
        ($p:expr, $($t:tt)*) => {
            let _ = tx.send(PileMessage::Phase(format!($($t)*)));
            let _ = tx.send(PileMessage::Progress($p));
        };
    }
    macro_rules! bail {
        ($($t:tt)*) => {{
            let msg = format!($($t)*);
            log!("❌  {}", msg);
            let _ = tx.send(PileMessage::Error(msg));
            return;
        }};
    }

    // ── 1. Prerequisites ──────────────────────────────────────────────────
    phase!(0.02, "Checking prerequisites…");
    log!("Looking for git…");
    if !cmd_exists("git") {
        bail!("'git' not found on PATH. Install git and try again.");
    }
    log!("✔  git OK");

    log!("Looking for {} …", cfg.python_cmd);
    if !cmd_exists(&cfg.python_cmd) {
        bail!(
            "'{}' not found on PATH. Install Python 3 and try again.",
            cfg.python_cmd
        );
    }
    log!("✔  {} OK", cfg.python_cmd);

    // ── 2. Clone repo ─────────────────────────────────────────────────────
    let repo_dir = cfg.target_dir.join("the-pile");
    if repo_dir.exists() && cfg.skip_clone_if_exists {
        log!("ℹ  Repo already at {}; skipping clone.", repo_dir.display());
    } else {
        phase!(0.05, "Cloning EleutherAI/the-pile…");
        log!(
            "git clone https://github.com/EleutherAI/the-pile.git → {}",
            repo_dir.display()
        );
        if let Err(e) = std::fs::create_dir_all(&cfg.target_dir) {
            bail!("Cannot create target directory: {e}");
        }
        let ok = stream_command(
            Command::new("git")
                .args(["clone", "https://github.com/EleutherAI/the-pile.git"])
                .arg(&repo_dir),
            &tx,
            0.05,
            0.12,
        );
        if !ok {
            bail!("git clone failed — see log above.");
        }
        log!("✔  Repository cloned.");
    }

    // ── 3. Create / reuse virtual environment ────────────────────────────
    let venv_dir = cfg.target_dir.join("pile-venv");
    #[cfg(target_os = "windows")]
    let venv_python = venv_dir.join("Scripts").join("python.exe");
    #[cfg(not(target_os = "windows"))]
    let venv_python = venv_dir.join("bin").join("python");

    if venv_dir.exists() {
        log!("ℹ  Virtual environment already at {}; reusing.", venv_dir.display());
    } else {
        phase!(0.10, "Creating Python virtual environment…");
        log!("Creating venv at {} …", venv_dir.display());
        let ok = stream_command(
            Command::new(&cfg.python_cmd).args(["-m", "venv", venv_dir.to_str().unwrap_or("pile-venv")]),
            &tx,
            0.10,
            0.12,
        );
        if !ok {
            bail!("python -m venv failed — see log above.");
        }
        log!("✔  Virtual environment created.");
    }

    // Track whether deps have already been installed so re-runs are fast.
    // Bump this string whenever the required Python deps change.
    const DEPS_VERSION: &str = "v3"; // added pytablewriter
    let deps_stamp = venv_dir.join(".quark-deps-installed");

    // ── 4. Install Python deps into the venv ─────────────────────────────
    let stamp_ok = deps_stamp.exists()
        && std::fs::read_to_string(&deps_stamp).unwrap_or_default().trim() == DEPS_VERSION;
    if stamp_ok {
        log!("ℹ  Dependencies already installed ({}); skipping.", deps_stamp.display());
    } else {
        phase!(0.12, "Installing Python dependencies (pip install -e .)…");
        log!("Using venv Python: {}", venv_python.display());
        let ok = stream_command(
            Command::new(&venv_python)
                .args(["-m", "pip", "install", "-e", "."])
                .current_dir(&repo_dir),
            &tx,
            0.12,
            0.16,
        );
        if !ok {
            bail!("pip install failed — see log above.");
        }
        log!("✔  Base Python dependencies installed.");

        // The Pile imports fasttext unconditionally but doesn't list it as a dep.
        // Try fasttext from PyPI first; if the C++ extension fails to compile
        // (common on Python 3.13+ / new g++ versions), fall back to a pure-Python
        // stub that always predicts English — enough for downloading data.
        phase!(0.16, "Installing fasttext + supplemental dependencies…");
        log!("pip install fasttext zstandard datasets pytablewriter …");
        let fasttext_ok = stream_command(
            Command::new(&venv_python)
                .args(["-m", "pip", "install", "fasttext", "zstandard", "datasets", "pytablewriter"])
                .current_dir(&repo_dir),
            &tx,
            0.16,
            0.19,
        );
        if fasttext_ok {
            log!("✔  fasttext + supplemental dependencies installed.");
        } else {
            log!("⚠  fasttext pip install failed (likely C++ build error on Python 3.13+).");
            log!("ℹ  Writing pure-Python fasttext stub — language filtering disabled, download unaffected.");
            // Locate site-packages inside the venv.
            let site_packages = find_venv_site_packages(&venv_dir);
            let stub = site_packages.join("fasttext.py");
            // Stub implements only what pile.py uses: load_model() + predict().
            // Returning __label__en / 1.0 disables language filtering so all
            // text passes through — the only effect is no lang-ID filtering.
            let stub_src = "\
# Auto-generated by Quark: pure-Python fasttext stub.\n\
# The real fasttext C extension could not be compiled on this Python version.\n\
# predict() always returns English so the Pile download is unaffected.\n\
class _Model:\n\
    def predict(self, text, k=1):\n\
        return (['__label__en'], [1.0])\n\
    def get_word_vector(self, word):\n\
        return [0.0] * 100\n\
def load_model(path):\n\
    return _Model()\n";
            match std::fs::write(&stub, stub_src) {
                Ok(_) => { log!("✔  fasttext stub written to {}.", stub.display()); }
                Err(e) => { log!("⚠  Could not write fasttext stub: {e}"); }
            }

            // Still install zstandard + datasets (pure Python / has wheels).
            log!("pip install zstandard datasets pytablewriter …");
            let ok2 = stream_command(
                Command::new(&venv_python)
                    .args(["-m", "pip", "install", "zstandard", "datasets", "pytablewriter"])
                    .current_dir(&repo_dir),
                &tx,
                0.19,
                0.20,
            );
            if ok2 {
                log!("✔  zstandard + datasets installed.");
            } else {
                log!("⚠  zstandard/datasets install failed; some components may not work.");
            }
        }

        // Write the stamp so subsequent runs skip reinstallation.
        let _ = std::fs::write(&deps_stamp, DEPS_VERSION);
    }

    // ── 4. Download / generate pile components ────────────────────────────
    let components = if cfg.components.is_empty() {
        vec!["pile_reprod".to_owned()]
    } else {
        cfg.components.clone()
    };
    let n_components = components.len();

    // Divide the 0.20 → 0.90 band evenly across each component.
    let band_start = 0.20_f32;
    let band_end = 0.90_f32;
    let band = band_end - band_start;
    let step = band / n_components as f32;

    for (idx, component) in components.iter().enumerate() {
        let p_start = band_start + idx as f32 * step;
        let p_end = p_start + step;

        if n_components > 1 {
            phase!(
                p_start,
                "Downloading component {}/{}: {}…",
                idx + 1,
                n_components,
                component
            );
        } else {
            phase!(p_start, "Downloading & building The Pile — this can take many hours…");
        }

        log!(
            "Running: {} the_pile/pile.py --interleave_output {} --using {}",
            venv_python.display(), cfg.interleave_output, component,
        );
        if component == "pile_reprod" {
            log!("⚠  The full Pile is ~825 GiB compressed. Make sure you have enough disk space.");
        }

        let ok = stream_command(
            Command::new(&venv_python)
                .args([
                    "the_pile/pile.py",
                    "--interleave_output",
                    &cfg.interleave_output.to_string(),
                    "--using",
                    component,
                ])
                .current_dir(&repo_dir),
            &tx,
            p_start,
            p_end,
        );
        if !ok {
            bail!("pile.py failed for component '{}' — see log above.", component);
        }
        log!("✔  Component '{}' complete ({}/{}).", component, idx + 1, n_components);
    }
    log!("✔  Pile data generation complete.");

    // ── 5. Pass-2 shuffle (if present) ───────────────────────────────────
    let pass2 = repo_dir.join("processing_scripts").join("pass2.py");
    if pass2.exists() {
        phase!(0.90, "Running pass-2 shuffle script…");
        log!("Running: {} {}", venv_python.display(), pass2.display());
        let ok = stream_command(
            Command::new(&venv_python)
                .arg(&pass2)
                .current_dir(&repo_dir),
            &tx,
            0.90,
            0.99,
        );
        if ok {
            log!("✔  Pass-2 shuffle complete.");
        } else {
            log!("⚠  pass2.py returned non-zero; continuing anyway.");
        }
    } else {
        log!("ℹ  No pass-2 script found; skipping shuffle step.");
    }

    // ── Done ─────────────────────────────────────────────────────────────
    phase!(1.0, "Complete!");
    log!("✅  The Pile build finished.  Data is in: {}", repo_dir.display());
    let _ = tx.send(PileMessage::Done);
}

// ─── Subprocess helpers ───────────────────────────────────────────────────────

/// Spawn `cmd`, stream its stdout + stderr to the channel, and return whether
/// it exited with status 0.  Progress is linearly interpolated from
/// `p_start` to `p_end` based on line count (heuristic: ~10 000 lines).
fn stream_command(
    cmd: &mut Command,
    tx: &Sender<PileMessage>,
    p_start: f32,
    p_end: f32,
) -> bool {
    let mut child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(PileMessage::Log(format!("Failed to spawn: {e}")));
            return false;
        }
    };

    // Forward stderr on a separate thread so it doesn't deadlock with stdout.
    let err_tx = tx.clone();
    let stderr = child.stderr.take().unwrap();
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let _ = err_tx.send(PileMessage::Log(format!("[err] {line}")));
        }
    });

    let stdout = child.stdout.take().unwrap();
    let mut n: u64 = 0;
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        n += 1;
        let _ = tx.send(PileMessage::Log(line));
        if p_end > p_start {
            let frac = (n as f32 / 10_000.0).min(1.0);
            let _ = tx.send(PileMessage::Progress(p_start + frac * (p_end - p_start)));
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

/// Locate the `site-packages` directory inside a venv.
///
/// On Unix the layout is `lib/pythonX.Y/site-packages/`;
/// on Windows it is `Lib/site-packages/`.
fn find_venv_site_packages(venv: &std::path::Path) -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        return venv.join("Lib").join("site-packages");
    }
    #[cfg(not(target_os = "windows"))]
    {
        // Walk lib/ looking for a pythonX.Y directory.
        let lib = venv.join("lib");
        if let Ok(entries) = std::fs::read_dir(&lib) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let s = name.to_string_lossy();
                if s.starts_with("python") {
                    return entry.path().join("site-packages");
                }
            }
        }
        // Fallback guess (e.g. python3.11)
        lib.join("python3").join("site-packages")
    }
}
