//! quark-code — terminal AI coding agent powered by a bundled Quark model.
//!
//! Usage:
//!   quark-code [project_dir]
//!
//! Looks for model files in a `model/` directory next to the executable
//! (set by quark-gui's Export panel).  Falls back to a demo/stub mode if
//! no model is found.
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

use std::path::PathBuf;
use anyhow::Result;
use quark_core::mcp::McpConfig;

fn main() -> Result<()> {
    // Initialise tracing (suppress most output; TUI owns the screen)
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("error"))
        .with_writer(std::io::stderr)
        .init();

    // ── Locate model directory ─────────────────────────────────────────────
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let model_dir = exe_dir.join("model");

    // ── Load config from model/config.json ────────────────────────────────
    let config_path = model_dir.join("config.json");
    let model_name  = if config_path.exists() {
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
        // Default: enable all read tools + shell for coding use
        McpConfig {
            read_file:    true,
            write_file:   true,
            list_dir:     true,
            search_files: true,
            get_cwd:      true,
            run_shell:    true,
            working_dir:  std::env::current_dir().unwrap_or_else(|_| exe_dir.clone()),
        }
    };

    // ── Load system prompt ─────────────────────────────────────────────────
    let system_prompt_path = model_dir.join("system_prompt.txt");
    let system_prompt = if system_prompt_path.exists() {
        std::fs::read_to_string(&system_prompt_path).unwrap_or_default()
    } else {
        DEFAULT_SYSTEM_PROMPT.to_owned()
    };

    // ── Project directory (first arg or cwd) ──────────────────────────────
    let project_root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // Resolve to absolute path
    let project_root = project_root.canonicalize().unwrap_or(project_root);

    // Point MCP working_dir at the project root
    mcp_cfg.working_dir = project_root.clone();

    // ── Model loaded? ──────────────────────────────────────────────────────
    let model_loaded = model_dir.join("checkpoint.safetensors").exists()
        && model_dir.join("tokenizer.json").exists();

    // ── Build App state ────────────────────────────────────────────────────
    let app = app::App::new(
        model_name,
        system_prompt,
        mcp_cfg,
        model_loaded,
        project_root,
    );

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
