//! Agent loop: builds prompts, calls inference (or stub), runs tool calls and
//! feeds their results back to the model until it answers without tools.

use std::sync::mpsc::{self, Receiver};
use std::thread;

use quark_core::mcp::{parse_tool_calls, McpConfig, ToolCall, ToolResult};

use crate::app::{App, FileChange, Message, Mode};
use crate::tools::execute_extended;

// ─── Response token channel ───────────────────────────────────────────────────

pub struct StreamHandle {
    pub rx: Receiver<AgentEvent>,
}

pub enum AgentEvent {
    Token(String),
    ToolCall { name: String, preview: String },
    ToolResult { name: String, ok: bool, preview: String },
    FileChanged(FileChange),
    /// Text of an intermediate round (the one that issued tool calls).
    Segment(String),
    Done(String), // final round's response (may be empty)
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
    let engine        = app.engine.clone();

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        run_agent_turn(tx, system_prompt, prompt, mcp_cfg, mode, engine);
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

/// Maximum model → tools → model round trips in one turn.
const MAX_TOOL_ROUNDS: usize = 8;
/// Tool output fed back to the model is truncated to this many characters.
const MAX_TOOL_RESULT_CHARS: usize = 4000;

/// Tools that modify the project (or can, like `run_shell`); blocked in Plan mode.
fn is_mutating_tool(name: &str) -> bool {
    matches!(
        name,
        "write_file" | "write_lines" | "apply_diff" | "git_add" | "git_commit" | "run_shell"
    )
}

fn run_agent_turn(
    tx:             mpsc::Sender<AgentEvent>,
    system_prompt:  String,
    prompt:         String,
    mcp_cfg:        McpConfig,
    mode:           Mode,
    engine:         Option<std::sync::Arc<quark_core::inference::InferenceEngine>>,
) {
    // `transcript` ends with an open `<assistant>` tag; each round appends the
    // model's reply and the tool results, then reopens `<assistant>`.
    let mut transcript = prompt;
    let mut response = String::new();

    for round in 0..MAX_TOOL_ROUNDS {
        // ── Generate response ────────────────────────────────────────────────
        response = match &engine {
            Some(engine) => match generate_streamed(engine, &system_prompt, &transcript, &tx) {
                Ok(text) => text,
                Err(e) => {
                    let _ = tx.send(AgentEvent::Error(format!("Inference error: {e}")));
                    return;
                }
            },
            None => {
                let text = generate_stub_response(&transcript, &mcp_cfg);
                // Stream word-by-word for a typing feel.
                for word in text.split_inclusive(' ') {
                    let _ = tx.send(AgentEvent::Token(word.to_owned()));
                    std::thread::sleep(std::time::Duration::from_millis(8));
                }
                text
            }
        };

        let calls = parse_tool_calls(&response);
        if calls.is_empty() {
            break;
        }
        // Show this round's text before its tool calls.
        let _ = tx.send(AgentEvent::Segment(response.clone()));

        // ── Execute tool calls ───────────────────────────────────────────────
        let mut results = String::new();
        for call in &calls {
            let preview = format!("{} {:?}", call.tool, call.args);
            let _ = tx.send(AgentEvent::ToolCall {
                name:    call.tool.clone(),
                preview,
            });

            let (result, change) = run_tool(call, mode, &mcp_cfg);
            if let Some(fc) = change {
                let _ = tx.send(AgentEvent::FileChanged(fc));
            }

            let _ = tx.send(AgentEvent::ToolResult {
                name:    result.tool.clone(),
                ok:      result.ok,
                preview: result.content.chars().take(300).collect(),
            });

            let status = if result.ok { "ok" } else { "error" };
            let content: String = result.content.chars().take(MAX_TOOL_RESULT_CHARS).collect();
            results.push_str(&format!(
                "<tool_result>\n{} ({status}):\n{content}\n</tool_result>\n",
                result.tool
            ));
        }

        // The stub can't read tool results, so a second round would repeat itself.
        if engine.is_none() {
            response.clear();
            break;
        }
        if round + 1 == MAX_TOOL_ROUNDS {
            response = format!("(stopped after {MAX_TOOL_ROUNDS} tool rounds)");
            break;
        }

        transcript.push_str(&response);
        transcript.push_str("\n</assistant>\n\n");
        transcript.push_str(&results);
        transcript.push_str("\n<assistant>\n");
    }

    let _ = tx.send(AgentEvent::Done(response));
}

/// Execute one tool call (unless Plan mode blocks it), returning its result
/// and the file change to record for undo, if any.
fn run_tool(call: &ToolCall, mode: Mode, cfg: &McpConfig) -> (ToolResult, Option<FileChange>) {
    if mode == Mode::Plan && is_mutating_tool(&call.tool) {
        let blocked = ToolResult {
            tool:    call.tool.clone(),
            ok:      false,
            content: "blocked: Plan mode is read-only (switch with /build)".into(),
        };
        return (blocked, None);
    }
    // Snapshot before-state for undo if write tool
    let before = snapshot_before(call, cfg);
    let result = execute_extended(call, cfg);
    let change = if result.ok { build_file_change(call, before, cfg) } else { None };
    (result, change)
}

/// Run the model on `system_prompt` + `transcript`, forwarding text pieces to
/// the UI as they are generated.
fn generate_streamed(
    engine:        &quark_core::inference::InferenceEngine,
    system_prompt: &str,
    transcript:    &str,
    tx:            &mpsc::Sender<AgentEvent>,
) -> anyhow::Result<String> {
    let full_prompt = format!("{system_prompt}\n\n{transcript}");
    let params = quark_core::inference::SamplingParams::default();
    let (token_tx, token_rx) = mpsc::channel::<String>();
    thread::scope(|scope| {
        scope.spawn(|| {
            for piece in token_rx {
                let _ = tx.send(AgentEvent::Token(piece));
            }
        });
        engine.generate_streaming(&full_prompt, params, token_tx)
    })
}

// ─── Inference stub ───────────────────────────────────────────────────────────

/// Generate a heuristic response demonstrating tool use without a real model.
fn generate_stub_response(prompt: &str, _cfg: &McpConfig) -> String {
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
        if (w.contains('/') || w.contains('.')) && !w.starts_with("http") {
            return Some(w.to_owned());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cfg(name: &str) -> McpConfig {
        let dir = std::env::temp_dir().join(format!("quark-code-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("hello.txt"), "hi").unwrap();
        McpConfig { working_dir: dir, ..McpConfig::default() }
    }

    fn write_call() -> ToolCall {
        let args = serde_json::json!({ "path": "new.txt", "content": "x" });
        ToolCall { tool: "write_file".into(), args }
    }

    #[test]
    fn plan_mode_blocks_writes() {
        let cfg = McpConfig { write_file: true, ..temp_cfg("plan") };
        let (result, change) = run_tool(&write_call(), Mode::Plan, &cfg);
        assert!(!result.ok);
        assert!(change.is_none());
        assert!(!cfg.working_dir.join("new.txt").exists());
    }

    #[test]
    fn build_mode_writes_and_records_change() {
        let cfg = McpConfig { write_file: true, ..temp_cfg("build") };
        let (result, change) = run_tool(&write_call(), Mode::Build, &cfg);
        assert!(result.ok, "{}", result.content);
        let change = change.expect("file change recorded");
        assert_eq!(change.before, None);
        assert_eq!(change.after.as_deref(), Some("x"));
    }

    #[test]
    fn stub_turn_runs_tools_and_finishes() {
        let cfg = McpConfig { list_dir: true, ..temp_cfg("stub") };
        let (tx, rx) = mpsc::channel();
        run_agent_turn(
            tx,
            String::new(),
            "<user>\nlist the files\n</user>\n\n<assistant>\n".into(),
            cfg,
            Mode::Build,
            None,
        );
        let events: Vec<AgentEvent> = rx.into_iter().collect();
        assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolCall { name, .. } if name == "list_dir")));
        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::ToolResult { ok: true, preview, .. } if preview.contains("hello.txt")
        )));
        assert!(matches!(events.last(), Some(AgentEvent::Done(_))));
    }
}
