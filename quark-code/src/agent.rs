//! Agent loop: builds prompts, calls inference (or stub), parses tool calls.

use std::sync::mpsc::{self, Receiver};
use std::thread;

use quark_core::mcp::{parse_tool_calls, McpConfig, ToolCall};

use crate::app::{App, FileChange, Message, Mode};
use crate::tools::{err_result, execute_extended, expand_mentions};

// ─── Response token channel ───────────────────────────────────────────────────

pub struct StreamHandle {
    pub rx: Receiver<AgentEvent>,
}

pub enum AgentEvent {
    Token(String),
    ToolCall { name: String, preview: String },
    ToolResult { name: String, ok: bool, preview: String },
    FileChanged(FileChange),
    Done(String), // final assembled response
    Error(String),
}

// ─── Start agent turn ─────────────────────────────────────────────────────────

/// Kick off an agent turn in a background thread.
/// The caller should drain `StreamHandle::rx` every UI frame.
pub fn start_turn(app: &App) -> StreamHandle {
    let mcp_cfg       = app.mcp_cfg.clone();
    let system_prompt = build_system_prompt(app);
    let prompt        = build_prompt(app);
    let mode          = app.mode;
    let model_loaded  = app.model_loaded;

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        run_agent_turn(tx, system_prompt, prompt, mcp_cfg, mode, model_loaded);
    });

    StreamHandle { rx }
}

// ─── Prompt construction ──────────────────────────────────────────────────────

fn build_system_prompt(app: &App) -> String {
    let mut s = app.system_prompt.clone();
    s.push_str("\n\n## Available Tools\n\n");
    s.push_str("You have access to these tools. Call them using XML tags:\n");
    s.push_str("<tool_call>{\"tool\": \"TOOL_NAME\", ...args}</tool_call>\n\n");
    s.push_str("### File Tools\n");
    if app.mcp_cfg.read_file   { s.push_str("- read_file(path) — read file contents\n"); }
    if app.mcp_cfg.write_file  { s.push_str("- write_file(path, content) — create or overwrite a file\n"); }
    if app.mcp_cfg.list_dir    { s.push_str("- list_dir(path?) — list directory\n"); }
    if app.mcp_cfg.search_files { s.push_str("- search_files(pattern, path?) — find files by name\n"); }
    if app.mcp_cfg.get_cwd     { s.push_str("- get_cwd() — current working directory\n"); }
    if app.mcp_cfg.run_shell   { s.push_str("- run_shell(command) — run shell command\n"); }
    s.push_str("- read_lines(path, start, end) — read specific lines\n");
    s.push_str("- write_lines(path, content) — write file content\n");
    s.push_str("- grep_code(pattern, path?) — search code with grep\n");
    s.push_str("- find_files(pattern) — find files matching pattern\n");
    s.push_str("\n### Git Tools\n");
    s.push_str("- git_status() — show working tree status\n");
    s.push_str("- git_diff(path?) — show uncommitted changes\n");
    s.push_str("- git_log(n?) — recent commit history\n");
    s.push_str("- git_add(path?) — stage changes\n");
    s.push_str("- git_commit(message) — commit staged changes\n");
    s.push_str("- git_branch() — current branch name\n");
    s.push_str("\n## Mode\n\n");
    match app.mode {
        Mode::Plan  => s.push_str("You are in **PLAN mode**. Suggest changes only. Do NOT call write_file, write_lines, git_add, or git_commit.\n"),
        Mode::Build => s.push_str("You are in **BUILD mode**. You may read and write files and run git commands.\n"),
    }
    if let Some(agents) = &app.agents_md {
        s.push_str("\n## Project Context (AGENTS.md)\n\n");
        s.push_str(agents);
    }
    s
}

fn build_prompt(app: &App) -> String {
    let mut prompt = String::new();
    // Last N messages for context (keep it reasonable)
    let history: Vec<&Message> = app.messages
        .iter()
        .filter(|m| m.role != crate::app::Role::System)
        .rev()
        .take(30)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    for msg in &history {
        let tag = match msg.role {
            crate::app::Role::User      => "user",
            crate::app::Role::Assistant => "assistant",
            crate::app::Role::Tool      => "tool_result",
            crate::app::Role::System    => continue,
        };
        prompt.push_str(&format!("<{tag}>\n{}\n</{tag}>\n\n", msg.content));
    }
    prompt.push_str("<assistant>\n");
    prompt
}

// ─── Agent turn execution ─────────────────────────────────────────────────────

fn run_agent_turn(
    tx:            mpsc::Sender<AgentEvent>,
    system_prompt: String,
    prompt:        String,
    mcp_cfg:       McpConfig,
    _mode:         Mode,
    model_loaded:  bool,
) {
    // ── Generate response ────────────────────────────────────────────────────
    let response = if model_loaded {
        // TODO: Wire in quark-core inference once weights are loaded.
        // This requires: tokenize(system_prompt + prompt) → generate() → detokenize
        // For now produce a stub response that still exercises the tool loop.
        generate_stub_response(&prompt, &mcp_cfg)
    } else {
        generate_stub_response(&prompt, &mcp_cfg)
    };

    // Stream the response token-by-token (word-by-word for the stub)
    for word in response.split_inclusive(' ') {
        let _ = tx.send(AgentEvent::Token(word.to_owned()));
        std::thread::sleep(std::time::Duration::from_millis(8)); // typing feel
    }

    // ── Parse and execute tool calls ─────────────────────────────────────────
    let calls = parse_tool_calls(&response);
    let mut changes: Vec<FileChange> = Vec::new();

    for call in &calls {
        let preview = format!("{} {:?}", call.tool, call.args);
        let _ = tx.send(AgentEvent::ToolCall {
            name:    call.tool.clone(),
            preview: preview.clone(),
        });

        // Snapshot before-state for undo if write tool
        let before = snapshot_before(&call, &mcp_cfg);

        let result = execute_extended(call, &mcp_cfg);

        // Capture after-state for undo
        if let Some(fc) = build_file_change(call, before, &mcp_cfg) {
            changes.push(fc.clone());
            let _ = tx.send(AgentEvent::FileChanged(fc));
        }

        let preview_result: String = result.content.chars().take(300).collect();
        let _ = tx.send(AgentEvent::ToolResult {
            name:    result.tool.clone(),
            ok:      result.ok,
            preview: preview_result,
        });
    }

    // Signal done with full response + any file changes
    let _ = tx.send(AgentEvent::Done(response));
}

// ─── Inference stub ───────────────────────────────────────────────────────────

/// Generate a heuristic response demonstrating tool use without a real model.
fn generate_stub_response(prompt: &str, cfg: &McpConfig) -> String {
    let prompt_lower = prompt.to_lowercase();

    // Try to be helpful based on keywords
    if prompt_lower.contains("git status") || prompt_lower.contains("what changed") {
        r#"Let me check the current git status for you.
<tool_call>{"tool":"git_status"}</tool_call>

I'll also check the recent commit history.
<tool_call>{"tool":"git_log","n":5}</tool_call>"#.to_owned()
    } else if prompt_lower.contains("read") || prompt_lower.contains("show") || prompt_lower.contains("what is in") {
        // Try to extract a filename
        let file = extract_likely_path(prompt).unwrap_or("src/main.rs".to_owned());
        format!(
            r#"Let me read that file for you.
<tool_call>{{"tool":"read_file","path":"{file}"}}</tool_call>"#
        )
    } else if prompt_lower.contains("list") || prompt_lower.contains("files") || prompt_lower.contains("directory") {
        r#"Let me list the project structure.
<tool_call>{"tool":"list_dir","path":"."}</tool_call>"#.to_owned()
    } else if prompt_lower.contains("search") || prompt_lower.contains("find") || prompt_lower.contains("where is") {
        let pattern = extract_search_pattern(prompt);
        format!(
            r#"Searching the codebase for that pattern.
<tool_call>{{"tool":"grep_code","pattern":"{pattern}"}}</tool_call>"#
        )
    } else {
        format!(
            "I'm running in stub mode — model weights are not yet loaded. \
            Once you train a model in Quark GUI and export it with `quark-code`, \
            I'll respond intelligently to: \"{}\"\n\n\
            In the meantime, try commands like:\n\
            - `/init` — analyse this project\n\
            - `/plan` — switch to plan mode\n\
            - `/build` — switch to build mode\n\
            - `@src/main.rs explain this file`",
            prompt.lines().last().unwrap_or("").trim()
        )
    }
}

fn extract_likely_path(prompt: &str) -> Option<String> {
    for word in prompt.split_whitespace() {
        let w = word.trim_matches(|c| c == '\'' || c == '"' || c == '`');
        if w.contains('/') || w.contains('.') {
            if !w.starts_with("http") {
                return Some(w.to_owned());
            }
        }
    }
    None
}

fn extract_search_pattern(prompt: &str) -> String {
    // Very naive: grab the last quoted or backtick string
    for delim in &['"', '\'', '`'] {
        let s: Vec<&str> = prompt.split(*delim).collect();
        if s.len() >= 3 {
            return s[1].to_owned();
        }
    }
    "fn ".to_owned()
}

// ─── Undo helpers ─────────────────────────────────────────────────────────────

fn snapshot_before(call: &ToolCall, cfg: &McpConfig) -> Option<String> {
    let path = call.args.get("path")?.as_str()?;
    match call.tool.as_str() {
        "write_file" | "write_lines" | "apply_diff" => {
            let full = cfg.working_dir.join(path);
            std::fs::read_to_string(full).ok()
        }
        _ => None,
    }
}

fn build_file_change(call: &ToolCall, before: Option<String>, cfg: &McpConfig) -> Option<FileChange> {
    let path = call.args.get("path")?.as_str()?;
    match call.tool.as_str() {
        "write_file" | "write_lines" | "apply_diff" => {
            let full = cfg.working_dir.join(path);
            let after = std::fs::read_to_string(&full).ok();
            Some(FileChange {
                path: full,
                before,
                after,
            })
        }
        _ => None,
    }
}
