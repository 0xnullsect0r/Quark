//! Agent loop: builds prompts, calls inference, runs tool calls and feeds
//! their results back to the model until it answers without tools.

use std::sync::mpsc::{self, Receiver};
use std::thread;

use quark_core::chat::{default_stop_strings, render_prompt, ChatMessage, ChatRole};
use quark_core::mcp::{parse_tool_calls, McpConfig, ToolCall, ToolResult};

use crate::app::{App, FileChange, Message, Mode};
use crate::tools::execute_extended;

// ─── Response token channel ───────────────────────────────────────────────────

pub struct StreamHandle {
    pub rx: Receiver<AgentEvent>,
}

pub enum AgentEvent {
    Token(String),
    ToolCall {
        name: String,
        preview: String,
    },
    ToolResult {
        name: String,
        ok: bool,
        preview: String,
    },
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
    let mcp_cfg = app.mcp_cfg.clone();
    let system_prompt = build_system_prompt(app);
    let history = build_history(app);
    let mode = app.mode;
    let engine = app.engine.clone();

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let mut generate =
            |system_prompt: &str, history: &[ChatMessage], tx: &mpsc::Sender<AgentEvent>| {
                generate_streamed(&engine, system_prompt, history, tx)
            };
        run_agent_turn(tx, system_prompt, history, mcp_cfg, mode, &mut generate);
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
    if app.mcp_cfg.read_file {
        s.push_str("- read_file(path) — read file contents\n");
    }
    if app.mcp_cfg.write_file {
        s.push_str("- write_file(path, content) — create or overwrite a file\n");
    }
    if app.mcp_cfg.list_dir {
        s.push_str("- list_dir(path?) — list directory\n");
    }
    if app.mcp_cfg.search_files {
        s.push_str("- search_files(pattern, path?) — find files by name\n");
    }
    if app.mcp_cfg.get_cwd {
        s.push_str("- get_cwd() — current working directory\n");
    }
    if app.mcp_cfg.run_shell {
        s.push_str("- run_shell(command) — run shell command\n");
    }
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

/// Recent conversation (without the system prompt) in the shared chat format.
fn build_history(app: &App) -> Vec<ChatMessage> {
    // Last N messages for context (keep it reasonable)
    let recent: Vec<&Message> = app
        .messages
        .iter()
        .filter(|m| m.role != crate::app::Role::System)
        .rev()
        .take(30)
        .collect();

    recent
        .into_iter()
        .rev()
        .map(|msg| {
            let role = match msg.role {
                crate::app::Role::User => ChatRole::User,
                crate::app::Role::Assistant => ChatRole::Assistant,
                crate::app::Role::Tool => ChatRole::Tool,
                crate::app::Role::System => unreachable!("filtered above"),
            };
            ChatMessage::new(role, msg.content.as_str())
        })
        .collect()
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

/// Produces the model's reply for one round: the system prompt, the
/// transcript so far, and the channel for streamed tokens.
type Generate<'a> =
    dyn FnMut(&str, &[ChatMessage], &mpsc::Sender<AgentEvent>) -> anyhow::Result<String> + 'a;

fn run_agent_turn(
    tx: mpsc::Sender<AgentEvent>,
    system_prompt: String,
    mut history: Vec<ChatMessage>,
    mcp_cfg: McpConfig,
    mode: Mode,
    generate: &mut Generate<'_>,
) {
    // Each round appends the model's reply and the tool results to `history`
    // and asks the model again.
    let mut response = String::new();

    for round in 0..MAX_TOOL_ROUNDS {
        // ── Generate response ────────────────────────────────────────────────
        response = match generate(&system_prompt, &history, &tx) {
            Ok(text) => text,
            Err(e) => {
                let _ = tx.send(AgentEvent::Error(format!("Inference error: {e}")));
                return;
            }
        };

        let calls = parse_tool_calls(&response);
        if calls.is_empty() {
            break;
        }
        // Show this round's text before its tool calls.
        let _ = tx.send(AgentEvent::Segment(response.clone()));

        // ── Execute tool calls ───────────────────────────────────────────────
        let mut results = Vec::with_capacity(calls.len());
        for call in &calls {
            let preview = format!("{} {:?}", call.tool, call.args);
            let _ = tx.send(AgentEvent::ToolCall {
                name: call.tool.clone(),
                preview,
            });

            let (result, change) = run_tool(call, mode, &mcp_cfg);
            if let Some(fc) = change {
                let _ = tx.send(AgentEvent::FileChanged(fc));
            }

            let _ = tx.send(AgentEvent::ToolResult {
                name: result.tool.clone(),
                ok: result.ok,
                preview: result.content.chars().take(300).collect(),
            });

            let content: String = result.content.chars().take(MAX_TOOL_RESULT_CHARS).collect();
            results.push(ChatMessage::tool_result(&ToolResult { content, ..result }));
        }

        if round + 1 == MAX_TOOL_ROUNDS {
            response = format!("(stopped after {MAX_TOOL_ROUNDS} tool rounds)");
            break;
        }

        history.push(ChatMessage::assistant(response.as_str()));
        history.extend(results);
    }

    let _ = tx.send(AgentEvent::Done(response));
}

/// Execute one tool call (unless Plan mode blocks it), returning its result
/// and the file change to record for undo, if any.
fn run_tool(call: &ToolCall, mode: Mode, cfg: &McpConfig) -> (ToolResult, Option<FileChange>) {
    if mode == Mode::Plan && is_mutating_tool(&call.tool) {
        let blocked = ToolResult {
            tool: call.tool.clone(),
            ok: false,
            content: "blocked: Plan mode is read-only (switch with /build)".into(),
        };
        return (blocked, None);
    }
    // Snapshot before-state for undo if write tool
    let before = snapshot_before(call, cfg);
    let result = execute_extended(call, cfg);
    let change = if result.ok {
        build_file_change(call, before, cfg)
    } else {
        None
    };
    (result, change)
}

/// Run the model on `system_prompt` + `transcript`, forwarding text pieces to
/// the UI as they are generated.
fn generate_streamed(
    engine: &quark_core::inference::InferenceEngine,
    system_prompt: &str,
    history: &[ChatMessage],
    tx: &mpsc::Sender<AgentEvent>,
) -> anyhow::Result<String> {
    let mut messages = vec![ChatMessage::system(system_prompt)];
    messages.extend_from_slice(history);
    let full_prompt = render_prompt(&messages);
    let params = quark_core::inference::SamplingParams {
        stop_strings: default_stop_strings(),
        ..Default::default()
    };
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

fn build_file_change(
    call: &ToolCall,
    before: Option<String>,
    cfg: &McpConfig,
) -> Option<FileChange> {
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
        McpConfig {
            working_dir: dir,
            ..McpConfig::default()
        }
    }

    fn write_call() -> ToolCall {
        let args = serde_json::json!({ "path": "new.txt", "content": "x" });
        ToolCall {
            tool: "write_file".into(),
            args,
        }
    }

    #[test]
    fn plan_mode_blocks_writes() {
        let cfg = McpConfig {
            write_file: true,
            ..temp_cfg("plan")
        };
        let (result, change) = run_tool(&write_call(), Mode::Plan, &cfg);
        assert!(!result.ok);
        assert!(change.is_none());
        assert!(!cfg.working_dir.join("new.txt").exists());
    }

    #[test]
    fn build_mode_writes_and_records_change() {
        let cfg = McpConfig {
            write_file: true,
            ..temp_cfg("build")
        };
        let (result, change) = run_tool(&write_call(), Mode::Build, &cfg);
        assert!(result.ok, "{}", result.content);
        let change = change.expect("file change recorded");
        assert_eq!(change.before, None);
        assert_eq!(change.after.as_deref(), Some("x"));
    }

    /// A scripted "model": one reply per round, in order.
    fn scripted(
        replies: Vec<&'static str>,
    ) -> impl FnMut(&str, &[ChatMessage], &mpsc::Sender<AgentEvent>) -> anyhow::Result<String> {
        let mut replies = replies.into_iter();
        move |_, _, tx| {
            let text = replies.next().unwrap_or("done");
            let _ = tx.send(AgentEvent::Token(text.to_owned()));
            Ok(text.to_owned())
        }
    }

    #[test]
    fn turn_runs_tools_feeds_results_back_and_finishes() {
        let cfg = McpConfig {
            list_dir: true,
            ..temp_cfg("turn")
        };
        let (tx, rx) = mpsc::channel();
        let mut generate = scripted(vec![
            "Let me look.\n<tool_call>{\"tool\":\"list_dir\",\"path\":\".\"}</tool_call>",
            "The directory holds hello.txt.",
        ]);
        run_agent_turn(
            tx,
            String::new(),
            vec![ChatMessage::user("list the files")],
            cfg,
            Mode::Build,
            &mut generate,
        );
        let events: Vec<AgentEvent> = rx.into_iter().collect();
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolCall { name, .. } if name == "list_dir")));
        assert!(events.iter().any(|e| matches!(
            e,
            AgentEvent::ToolResult { ok: true, preview, .. } if preview.contains("hello.txt")
        )));
        assert!(
            matches!(events.iter().find(|e| matches!(e, AgentEvent::Segment(_))), Some(AgentEvent::Segment(t)) if t.starts_with("Let me look."))
        );
        assert!(
            matches!(events.last(), Some(AgentEvent::Done(t)) if t == "The directory holds hello.txt.")
        );
    }

    #[test]
    fn turn_stops_after_max_tool_rounds() {
        let cfg = McpConfig {
            list_dir: true,
            ..temp_cfg("rounds")
        };
        let (tx, rx) = mpsc::channel();
        let mut rounds = 0usize;
        let mut generate = |_: &str, _: &[ChatMessage], _: &mpsc::Sender<AgentEvent>| {
            rounds += 1;
            Ok("<tool_call>{\"tool\":\"list_dir\",\"path\":\".\"}</tool_call>".to_owned())
        };
        run_agent_turn(
            tx,
            String::new(),
            vec![ChatMessage::user("loop")],
            cfg,
            Mode::Build,
            &mut generate,
        );
        let events: Vec<AgentEvent> = rx.into_iter().collect();
        assert_eq!(rounds, MAX_TOOL_ROUNDS);
        assert!(matches!(events.last(), Some(AgentEvent::Done(t)) if t.contains("stopped after")));
    }
}
