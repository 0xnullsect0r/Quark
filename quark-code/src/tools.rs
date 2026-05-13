//! Extended tool set for Quark Code (beyond basic MCP tools).
//! Adds git operations, code search, project context, and shell execution.

use std::path::Path;
use std::process::Command;

use quark_core::mcp::{execute_tool, McpConfig, ToolCall, ToolResult};

// ─── Extended tool call ───────────────────────────────────────────────────────

/// Execute a tool call, supporting both standard MCP tools and Quark Code
/// extended tools (git_*, search_code, etc.).
pub fn execute_extended(call: &ToolCall, cfg: &McpConfig) -> ToolResult {
    match call.tool.as_str() {
        // Git tools
        "git_status"  => git_tool(cfg, &["status", "--short"]),
        "git_diff"    => {
            let file = call.args.get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if file.is_empty() {
                git_tool(cfg, &["diff"])
            } else {
                git_tool(cfg, &["diff", "--", file])
            }
        }
        "git_log"     => {
            let n = call.args.get("n")
                .and_then(|v| v.as_u64())
                .unwrap_or(10)
                .to_string();
            git_tool(cfg, &["log", "--oneline", &format!("-{n}")])
        }
        "git_add"     => {
            let path = get_str(&call.args, "path");
            git_tool(cfg, &["add", if path.is_empty() { "." } else { &path }])
        }
        "git_commit"  => {
            let msg = get_str(&call.args, "message");
            if msg.is_empty() {
                err_result(&call.tool, "message arg required")
            } else {
                git_tool(cfg, &["commit", "-m", &msg])
            }
        }
        "git_branch"  => git_tool(cfg, &["branch", "--show-current"]),

        // Code analysis
        "grep_code"   => {
            let pattern = get_str(&call.args, "pattern");
            let path    = get_str(&call.args, "path");
            if pattern.is_empty() {
                err_result(&call.tool, "pattern arg required")
            } else {
                grep_code(cfg, &pattern, if path.is_empty() { "." } else { &path })
            }
        }
        "find_files"  => {
            let pattern = get_str(&call.args, "pattern");
            find_files(cfg, &pattern)
        }
        "read_lines"  => {
            let path  = get_str(&call.args, "path");
            let start = call.args.get("start").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
            let end   = call.args.get("end").and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as usize;
            read_lines(cfg, &path, start, end)
        }
        "write_lines" => {
            let path    = get_str(&call.args, "path");
            let content = get_str(&call.args, "content");
            write_file_checked(cfg, &path, &content)
        }
        "apply_diff"  => {
            let path  = get_str(&call.args, "path");
            let patch = get_str(&call.args, "patch");
            apply_patch(cfg, &path, &patch)
        }

        // Delegate everything else to quark-core MCP
        _ => execute_tool(call, cfg),
    }
}

// ─── Git ──────────────────────────────────────────────────────────────────────

fn git_tool(cfg: &McpConfig, args: &[&str]) -> ToolResult {
    let output = Command::new("git")
        .args(args)
        .current_dir(&cfg.working_dir)
        .output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
            let content = if stdout.is_empty() { stderr } else { stdout };
            ToolResult {
                tool: format!("git {}", args.join(" ")),
                ok:      out.status.success(),
                content: if content.is_empty() { "(no output)".into() } else { content },
            }
        }
        Err(e) => err_result(&format!("git {}", args.join(" ")), &e.to_string()),
    }
}

// ─── Code search ─────────────────────────────────────────────────────────────

fn grep_code(cfg: &McpConfig, pattern: &str, path: &str) -> ToolResult {
    let full = cfg.working_dir.join(path);
    let out  = Command::new("grep")
        .args(["-rn", "--include=*.rs", "--include=*.py", "--include=*.ts",
               "--include=*.js", "--include=*.go", "--include=*.java",
               "--include=*.c", "--include=*.cpp", "--include=*.h",
               "-e", pattern])
        .arg(full)
        .output();
    match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout).into_owned();
            ToolResult {
                tool:    "grep_code".into(),
                ok:      true,
                content: if s.is_empty() { "No matches found.".into() } else { s },
            }
        }
        Err(e) => err_result("grep_code", &e.to_string()),
    }
}

fn find_files(cfg: &McpConfig, pattern: &str) -> ToolResult {
    // Use walkdir-style find
    let root = &cfg.working_dir;
    let mut results = Vec::new();
    find_recursive(root, root, pattern, &mut results, 0);
    results.sort();
    ToolResult {
        tool:    "find_files".into(),
        ok:      true,
        content: if results.is_empty() {
            "No files matched.".into()
        } else {
            results.join("\n")
        },
    }
}

fn find_recursive(root: &Path, dir: &Path, pattern: &str, out: &mut Vec<String>, depth: usize) {
    if depth > 8 { return; }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        // Skip hidden dirs and common noise
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with('.') || name == "target" || name == "node_modules" { continue; }
        }
        if path.is_dir() {
            find_recursive(root, &path, pattern, out, depth + 1);
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if glob_match(pattern, name) {
                let rel = path.strip_prefix(root).unwrap_or(&path);
                out.push(rel.to_string_lossy().into_owned());
            }
        }
    }
}

/// Simple glob: `*` matches any chars, `?` matches one char.
fn glob_match(pattern: &str, name: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let n: Vec<char> = name.chars().collect();
    glob_match_inner(&p, &n)
}

fn glob_match_inner(p: &[char], n: &[char]) -> bool {
    match (p.first(), n.first()) {
        (None, None)     => true,
        (Some('*'), _)   => glob_match_inner(&p[1..], n) || (!n.is_empty() && glob_match_inner(p, &n[1..])),
        (Some('?'), Some(_)) => glob_match_inner(&p[1..], &n[1..]),
        (Some(a), Some(b)) if a == b => glob_match_inner(&p[1..], &n[1..]),
        _ => false,
    }
}

// ─── Read/write with range ────────────────────────────────────────────────────

fn read_lines(cfg: &McpConfig, path: &str, start: usize, end: usize) -> ToolResult {
    if path.is_empty() {
        return err_result("read_lines", "path arg required");
    }
    let full = cfg.working_dir.join(path);
    match std::fs::read_to_string(&full) {
        Ok(content) => {
            let lines: Vec<&str> = content.lines().collect();
            let s = start.saturating_sub(1);
            let e = end.min(lines.len());
            let slice = if s < lines.len() && s < e {
                lines[s..e].join("\n")
            } else {
                content.clone()
            };
            let shown_start = s + 1;
            let shown_end   = e;
            ToolResult {
                tool:    "read_lines".into(),
                ok:      true,
                content: format!("// {path} lines {shown_start}–{shown_end}\n{slice}"),
            }
        }
        Err(e) => err_result("read_lines", &e.to_string()),
    }
}

fn write_file_checked(cfg: &McpConfig, path: &str, content: &str) -> ToolResult {
    if path.is_empty() {
        return err_result("write_lines", "path arg required");
    }
    let full = cfg.working_dir.join(path);
    if let Some(parent) = full.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&full, content) {
        Ok(_)  => ToolResult { tool: "write_lines".into(), ok: true, content: format!("Wrote {path}") },
        Err(e) => err_result("write_lines", &e.to_string()),
    }
}

fn apply_patch(_cfg: &McpConfig, path: &str, _patch: &str) -> ToolResult {
    // Very simple line-based patch: lines starting with '+' are additions,
    // '-' are removals, everything else is context.  Not full unified diff —
    // just best-effort for small model-generated patches.
    ToolResult {
        tool:    "apply_diff".into(),
        ok:      false,
        content: format!("apply_diff not yet implemented for {path}; use write_file instead."),
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn get_str(args: &serde_json::Value, key: &str) -> String {
    args.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned()
}

pub fn err_result(tool: &str, msg: &str) -> ToolResult {
    ToolResult { tool: tool.into(), ok: false, content: msg.into() }
}

// ─── Context expansion: @file mentions ───────────────────────────────────────

/// Expand `@path/to/file` mentions in `input` by inlining their contents.
pub fn expand_mentions(input: &str, working_dir: &Path) -> String {
    let mut result = String::with_capacity(input.len() + 512);
    for word in input.split_whitespace() {
        if let Some(rel) = word.strip_prefix('@') {
            let path = working_dir.join(rel);
            if let Ok(content) = std::fs::read_to_string(&path) {
                result.push_str(&format!(
                    "\n<file path=\"{rel}\">\n{content}\n</file>\n"
                ));
            } else {
                result.push_str(word);
            }
        } else {
            result.push_str(word);
        }
        result.push(' ');
    }
    result
}
