//! quark-code — terminal AI coding agent powered by a bundled Quark model.
//!
//! Usage:
//!   quark-code [--model <bundle-dir>] [--plan] [--allow-shell] [project_dir]
//!
//! `--plan` starts in Plan mode (the model explains changes but cannot apply
//! them); the default is Build mode.
//!
//! `run_shell` is disabled unless `--allow-shell` is passed or `model/mcp.json`
//! enables it.
//!
//! The model is loaded from `--model <bundle-dir>` or, by default, from a
//! `model/` directory next to the executable (written by quark-gui's Export
//! tab or `quark-cli export`).  Without a loadable model quark-code exits with
//! an error: there is no fallback mode.
//!
//! Features:
//! - Full TUI (ratatui + crossterm) with Plan/Build mode toggle
//! - Slash commands: /init /plan /build /undo /redo /diff /mcp /help /exit
//! - MCP tools: read_file, write_file, list_dir, search_files, run_shell
//! - Extended tools: git_status, git_diff, git_log, git_add, git_commit,
//!   grep_code, find_files, read_lines, write_lines
//! - @file context injection
//! - Undo/redo stack for all file changes
//! - Project context scanner + AGENTS.md writer (/init)
//! - Streaming token output

#![allow(dead_code)]

mod agent;
mod app;
mod context;
mod tools;
mod tui;

use anyhow::Result;
use quark_core::inference::InferenceEngine;
use quark_core::mcp::McpConfig;
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> Result<()> {
    // Initialise tracing (suppress most output; TUI owns the screen)
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("error"))
        .with_writer(std::io::stderr)
        .init();

    // ── Command-line flags ─────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag_value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };

    // ── Locate model directory ─────────────────────────────────────────────
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let model_dir = flag_value("--model")
        .map(PathBuf::from)
        .unwrap_or_else(|| exe_dir.join("model"));

    // ── Load config from model/config.json ────────────────────────────────
    let config_path = model_dir.join("config.json");
    let model_name = if config_path.exists() {
        let txt = std::fs::read_to_string(&config_path).unwrap_or_default();
        let v: serde_json::Value = serde_json::from_str(&txt).unwrap_or_default();
        v["name"].as_str().unwrap_or("Quark").to_owned()
    } else {
        "Quark".to_owned()
    };

    // ── Load MCP config ────────────────────────────────────────────────────
    let mcp_path = model_dir.join("mcp.json");
    let mut mcp_cfg: McpConfig = if mcp_path.exists() {
        let txt = std::fs::read_to_string(&mcp_path)?;
        serde_json::from_str(&txt).unwrap_or_default()
    } else {
        // Default: file tools on; shell stays off unless --allow-shell is passed
        McpConfig {
            read_file: true,
            write_file: true,
            list_dir: true,
            search_files: true,
            get_cwd: true,
            run_shell: false,
            working_dir: std::env::current_dir().unwrap_or_else(|_| exe_dir.clone()),
        }
    };

    // ── Load system prompt ─────────────────────────────────────────────────
    let system_prompt_path = model_dir.join("system_prompt.txt");
    let system_prompt = if system_prompt_path.exists() {
        std::fs::read_to_string(&system_prompt_path).unwrap_or_default()
    } else {
        DEFAULT_SYSTEM_PROMPT.to_owned()
    };

    if args.iter().any(|a| a == "--allow-shell") {
        mcp_cfg.run_shell = true;
    }

    // ── Project directory (first positional arg or cwd) ───────────────────
    let model_flag_value = flag_value("--model");
    let project_root = args
        .iter()
        .find(|a| !a.starts_with("--") && Some(*a) != model_flag_value)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // Resolve to absolute path
    let project_root = project_root.canonicalize().unwrap_or(project_root);

    // Point MCP working_dir at the project root
    mcp_cfg.working_dir = project_root.clone();

    // ── Load inference engine ─────────────────────────────────────────────
    let engine = match InferenceEngine::load_bundle(&model_dir) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            if has_bundled_model(&model_dir) {
                eprintln!(
                    "quark-code: model in {} failed to load: {e:#}",
                    model_dir.display()
                );
            } else {
                eprintln!("{}", no_model_message(&model_dir));
            }
            std::process::exit(2);
        }
    };

    // ── Build App state ────────────────────────────────────────────────────
    let mut app = app::App::new(model_name, system_prompt, mcp_cfg, engine, project_root);
    if args.iter().any(|a| a == "--plan") {
        app.mode = app::Mode::Plan;
    }

    // ── Run TUI ────────────────────────────────────────────────────────────
    tui::run(app).map_err(|e| anyhow::anyhow!("TUI error: {e}"))
}

const DEFAULT_SYSTEM_PROMPT: &str = "\
You are Quark Code, an expert AI coding assistant running locally on the user's machine.

You help with:
- Reading, understanding, and explaining code
- Adding features and fixing bugs
- Refactoring and code review
- Git operations (status, diff, commit)
- Shell commands and project management

You have access to a comprehensive set of tools for interacting with the filesystem and git.
Always read relevant files before making changes.
In Plan mode, explain what you would do without actually doing it.
In Build mode, apply changes directly using write_file or write_lines.
Prefer small, targeted edits. After making changes, summarise what was done.";

fn has_bundled_model(model_dir: &std::path::Path) -> bool {
    quark_core::checkpoint::export::find_bundled_model(model_dir).is_some()
}

/// What to do when no model bundle is present.
fn no_model_message(model_dir: &std::path::Path) -> String {
    format!(
        "quark-code: no model found in {}\n\n\
         quark-code needs a trained Quark model to run. To get one:\n\
           1. Train a model in the Quark GUI (Training tab), then export it from\n\
              the Export tab — or run: quark-cli export <checkpoint> <bundle-dir>\n\
           2. Start quark-code with: quark-code --model <bundle-dir>\n\
              (a bundle is a directory holding checkpoint/ (or checkpoint.bin),\n\
              config.json and tokenizer.json)\n\n\
         Without --model, quark-code looks for a model/ directory next to the executable.",
        model_dir.display()
    )
}
