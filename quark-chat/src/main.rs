//! quark-chat — standalone terminal chat app exported from Quark GUI.
//!
//! Expects model files next to the executable in a `model/` directory:
//!   model/config.json              QuarkConfig
//!   model/checkpoint.safetensors   weights
//!   model/tokenizer.json           BPE tokenizer
//!   model/mcp.json                 McpConfig  (optional)
//!   model/system_prompt.txt        system prompt (optional)

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::Result;
use quark_core::mcp::{execute_tool, format_tool_result, parse_tool_calls, McpConfig};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .init();

    // Locate model dir relative to executable
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let model_dir = exe_dir.join("model");

    if !model_dir.exists() {
        eprintln!("Error: no model/ directory found next to this executable.");
        eprintln!("Expected: {}", model_dir.display());
        std::process::exit(1);
    }

    // Load MCP config
    let mcp_path = model_dir.join("mcp.json");
    let mut mcp_cfg: McpConfig = if mcp_path.exists() {
        let txt = std::fs::read_to_string(&mcp_path)?;
        serde_json::from_str(&txt).unwrap_or_default()
    } else {
        McpConfig::default()
    };
    // Set working dir to cwd so relative tool paths work naturally
    mcp_cfg.working_dir = std::env::current_dir().unwrap_or_else(|_| exe_dir.clone());

    // Load system prompt
    let system_prompt_path = model_dir.join("system_prompt.txt");
    let system_prompt = if system_prompt_path.exists() {
        std::fs::read_to_string(&system_prompt_path).unwrap_or_default()
    } else {
        "You are a helpful coding assistant with access to MCP tools for reading and writing files."
            .to_string()
    };

    // Load config.json for display
    let config_path = model_dir.join("config.json");
    let model_name = if config_path.exists() {
        let txt = std::fs::read_to_string(&config_path).unwrap_or_default();
        let v: serde_json::Value = serde_json::from_str(&txt).unwrap_or_default();
        v["name"].as_str().unwrap_or("Quark").to_string()
    } else {
        "Quark".to_string()
    };

    println!("╔══════════════════════════════════════════╗");
    println!("║  {} — Chat                               ", model_name);
    println!("╚══════════════════════════════════════════╝");
    println!();
    println!(
        "System: {}",
        system_prompt.lines().next().unwrap_or("")
    );
    println!();
    print_mcp_status(&mcp_cfg);
    println!();
    println!("Type your message and press Enter. Ctrl+C to exit.");
    println!("─────────────────────────────────────────────────");
    println!();

    // Conversation history (simple string accumulation for context)
    let mut history = format!("<system>\n{system_prompt}\n</system>\n\n");

    let stdin = io::stdin();
    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("Input error: {e}");
                break;
            }
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/exit" || input == "/quit" {
            break;
        }
        if input == "/clear" {
            history = format!("<system>\n{system_prompt}\n</system>\n\n");
            println!("[Conversation cleared]");
            continue;
        }
        if input == "/help" {
            print_help();
            continue;
        }
        if input.starts_with("/mcp") {
            print_mcp_status(&mcp_cfg);
            continue;
        }

        history.push_str(&format!("<user>\n{input}\n</user>\n\n<assistant>\n"));

        // NOTE: Real inference requires a loaded Burn model.
        // This binary is designed to be bundled by quark-gui's export feature,
        // which will wire up actual model loading when the full inference pipeline
        // is integrated. For now we show the tool-calling loop correctly and
        // emit a placeholder response so the MCP dispatch machinery is exercisable.
        println!();
        println!("Quark: [Model weights not loaded — this binary was exported without a compiled");
        println!(
            "       backend wired to inference. Re-export from Quark GUI after training"
        );
        println!("       completes to get a functional model.]");

        // Demo: if user mentions a filename, do a read_file to show MCP works
        let simulated_response =
            if input.contains(".rs") || input.contains(".txt") || input.contains(".py") {
                // Extract a plausible filename from input
                let word = input
                    .split_whitespace()
                    .find(|w| w.contains('.') && !w.starts_with("http"))
                    .unwrap_or("");
                if !word.is_empty() {
                    format!(
                        "Let me read that file for you.\n<tool_call>{{\"tool\":\"read_file\",\"path\":\"{word}\"}}</tool_call>"
                    )
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

        if !simulated_response.is_empty() {
            let calls = parse_tool_calls(&simulated_response);
            for call in &calls {
                println!();
                println!("[MCP] Calling tool: {} {:?}", call.tool, call.args);
                let result = execute_tool(call, &mcp_cfg);
                let formatted = format_tool_result(&result);
                println!(
                    "[MCP] Result ({}):",
                    if result.ok { "ok" } else { "error" }
                );
                // Show a preview (first 500 chars)
                let preview: String = result.content.chars().take(500).collect();
                println!("{preview}");
                history.push_str(&formatted);
                history.push('\n');
            }
        }

        history.push_str("</assistant>\n\n");
        println!();
    }

    println!("\nGoodbye!");
    Ok(())
}

fn print_mcp_status(cfg: &McpConfig) {
    println!("MCP Tools enabled:");
    println!(
        "  read_file:    {}",
        if cfg.read_file { "✓" } else { "✗" }
    );
    println!(
        "  write_file:   {}",
        if cfg.write_file { "✓" } else { "✗" }
    );
    println!(
        "  list_dir:     {}",
        if cfg.list_dir { "✓" } else { "✗" }
    );
    println!(
        "  search_files: {}",
        if cfg.search_files { "✓" } else { "✗" }
    );
    println!(
        "  get_cwd:      {}",
        if cfg.get_cwd { "✓" } else { "✗" }
    );
    println!(
        "  run_shell:    {}",
        if cfg.run_shell { "✓" } else { "✗" }
    );
    println!("  working_dir:  {}", cfg.working_dir.display());
}

fn print_help() {
    println!("Commands:");
    println!("  /clear  — clear conversation history");
    println!("  /mcp    — show MCP tool status");
    println!("  /help   — show this help");
    println!("  /exit   — quit");
    println!();
    println!("MCP Tool Calling:");
    println!("  The model can call tools by emitting:");
    println!("  <tool_call>{{\"tool\":\"read_file\",\"path\":\"file.txt\"}}</tool_call>");
    println!(
        "  Available tools: read_file, write_file, list_dir, search_files, get_cwd, run_shell"
    );
}
