//! MCP (Model Context Protocol) tool definitions.
//! quark-chat uses these to intercept tool calls in model output and execute them.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A tool call parsed from model output: <tool_call>{"tool":"...","arg":"..."}</tool_call>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool: String,
    #[serde(flatten)]
    pub args: serde_json::Value,
}

/// Result returned after executing a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool: String,
    pub ok: bool,
    pub content: String,
}

/// Config controlling which MCP tools are enabled
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    pub read_file: bool,
    pub write_file: bool,
    pub list_dir: bool,
    pub search_files: bool,
    pub get_cwd: bool,
    pub run_shell: bool,
    pub working_dir: PathBuf,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            read_file: true,
            write_file: true,
            list_dir: true,
            search_files: true,
            get_cwd: true,
            run_shell: false, // off by default — potentially dangerous
            working_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }
}

/// Parse all <tool_call>…</tool_call> blocks from a model output string
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after_open = &rest[start + "<tool_call>".len()..];
        if let Some(end) = after_open.find("</tool_call>") {
            let json_str = after_open[..end].trim();
            if let Ok(call) = serde_json::from_str::<ToolCall>(json_str) {
                calls.push(call);
            }
            rest = &after_open[end + "</tool_call>".len()..];
        } else {
            break;
        }
    }
    calls
}

/// Execute a single tool call according to the given McpConfig
pub fn execute_tool(call: &ToolCall, cfg: &McpConfig) -> ToolResult {
    let result = match call.tool.as_str() {
        "read_file" => {
            if !cfg.read_file {
                return disabled(&call.tool);
            }
            let path = get_str(&call.args, "path");
            match std::fs::read_to_string(resolve(&cfg.working_dir, &path)) {
                Ok(content) => Ok(content),
                Err(e) => Err(e.to_string()),
            }
        }
        "write_file" => {
            if !cfg.write_file {
                return disabled(&call.tool);
            }
            let path = get_str(&call.args, "path");
            let content = get_str(&call.args, "content");
            let full = resolve(&cfg.working_dir, &path);
            if let Some(parent) = full.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match std::fs::write(&full, content) {
                Ok(()) => Ok(format!(
                    "Written {} bytes to {}",
                    full.display().to_string().len(),
                    full.display()
                )),
                Err(e) => Err(e.to_string()),
            }
        }
        "list_dir" => {
            if !cfg.list_dir {
                return disabled(&call.tool);
            }
            let path = get_str(&call.args, "path");
            let dir = if path.is_empty() {
                cfg.working_dir.clone()
            } else {
                resolve(&cfg.working_dir, &path)
            };
            match std::fs::read_dir(&dir) {
                Ok(rd) => {
                    let mut entries: Vec<String> = rd
                        .flatten()
                        .map(|e| {
                            let name = e.file_name().to_string_lossy().to_string();
                            let kind = if e.path().is_dir() { "/" } else { "" };
                            format!("{name}{kind}")
                        })
                        .collect();
                    entries.sort();
                    Ok(entries.join("\n"))
                }
                Err(e) => Err(e.to_string()),
            }
        }
        "search_files" => {
            if !cfg.search_files {
                return disabled(&call.tool);
            }
            let pattern = get_str(&call.args, "pattern");
            let dir = resolve(&cfg.working_dir, &get_str(&call.args, "dir"));
            let results = glob_search(&dir, &pattern, 200);
            Ok(results.join("\n"))
        }
        "get_cwd" => {
            if !cfg.get_cwd {
                return disabled(&call.tool);
            }
            Ok(cfg.working_dir.display().to_string())
        }
        "run_shell" => {
            if !cfg.run_shell {
                return disabled(&call.tool);
            }
            let cmd = get_str(&call.args, "command");
            #[cfg(unix)]
            let output = std::process::Command::new("sh").arg("-c").arg(&cmd).output();
            #[cfg(windows)]
            let output = std::process::Command::new("cmd").arg("/C").arg(&cmd).output();
            match output {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    Ok(format!("{stdout}{stderr}").chars().take(4000).collect())
                }
                Err(e) => Err(e.to_string()),
            }
        }
        other => Err(format!("Unknown tool: {other}")),
    };
    match result {
        Ok(content) => ToolResult {
            tool: call.tool.clone(),
            ok: true,
            content,
        },
        Err(e) => ToolResult {
            tool: call.tool.clone(),
            ok: false,
            content: e,
        },
    }
}

fn disabled(tool: &str) -> ToolResult {
    ToolResult {
        tool: tool.to_string(),
        ok: false,
        content: format!("Tool '{tool}' is disabled in this app's MCP config."),
    }
}

fn get_str(val: &serde_json::Value, key: &str) -> String {
    val.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn resolve(base: &Path, path: &str) -> PathBuf {
    if path.is_empty() {
        return base.to_path_buf();
    }
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

fn glob_search(dir: &Path, pattern: &str, limit: usize) -> Vec<String> {
    let mut results = Vec::new();
    search_recursive(dir, dir, pattern, &mut results, limit);
    results
}

fn search_recursive(base: &Path, dir: &Path, pattern: &str, out: &mut Vec<String>, limit: usize) {
    if out.len() >= limit {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            search_recursive(base, &path, pattern, out, limit);
        } else if pattern.is_empty() || name.contains(pattern) {
            let rel = path.strip_prefix(base).unwrap_or(&path);
            out.push(rel.display().to_string());
        }
        if out.len() >= limit {
            return;
        }
    }
}

/// Format a tool result for injection back into the conversation as context
pub fn format_tool_result(result: &ToolResult) -> String {
    let status = if result.ok { "ok" } else { "error" };
    format!(
        "<tool_result tool=\"{}\" status=\"{}\">\n{}\n</tool_result>",
        result.tool, status, result.content
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_calls() {
        let text = r#"I will read that file.
<tool_call>{"tool":"read_file","path":"README.md"}</tool_call>
Let me also check the directory.
<tool_call>{"tool":"list_dir","path":"."}</tool_call>"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool, "read_file");
        assert_eq!(calls[1].tool, "list_dir");
    }
}
