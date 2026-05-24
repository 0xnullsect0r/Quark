//! quark-chat — standalone terminal chat app exported from Quark GUI.
//!
//! Expects model files next to the executable in a `model/` directory:
//!   model/config.json              QuarkConfig
//!   model/checkpoint.bin           weights (CompactRecorder format)
//!   model/tokenizer.json           BPE tokenizer
//!   model/mcp.json                 McpConfig  (optional)
//!   model/system_prompt.txt        system prompt (optional)

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::Result;
use quark_core::inference::sampling::SamplingParams;
use quark_core::inference::InferenceEngine;
use quark_core::mcp::{execute_tool, format_tool_result, parse_tool_calls, McpConfig};
use quark_core::model::config::QuarkConfig;

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
    mcp_cfg.working_dir = std::env::current_dir().unwrap_or_else(|_| exe_dir.clone());

    // Load system prompt
    let system_prompt_path = model_dir.join("system_prompt.txt");
    let system_prompt = if system_prompt_path.exists() {
        std::fs::read_to_string(&system_prompt_path).unwrap_or_default()
    } else {
        "You are a helpful coding assistant with access to MCP tools for reading and writing files."
            .to_string()
    };

    // Load model config
    let config_path = model_dir.join("config.json");
    let model_config: QuarkConfig = if config_path.exists() {
        let txt = std::fs::read_to_string(&config_path)?;
        serde_json::from_str(&txt).unwrap_or_else(|_| QuarkConfig::quark_1b())
    } else {
        QuarkConfig::quark_1b()
    };

    let model_name = "Quark".to_string();

    // Load inference engine
    let checkpoint_path = model_dir.join("checkpoint.bin");
    let tokenizer_path = model_dir.join("tokenizer.json");

    let engine = if checkpoint_path.exists() && tokenizer_path.exists() {
        eprintln!("Loading model from {}…", checkpoint_path.display());
        match InferenceEngine::load(&checkpoint_path, &model_config, &tokenizer_path) {
            Ok(e) => {
                eprintln!("Model loaded.");
                Some(e)
            }
            Err(e) => {
                eprintln!("Warning: model load failed: {e}");
                None
            }
        }
    } else {
        eprintln!("Warning: checkpoint.bin or tokenizer.json not found — running without model.");
        None
    };

    let sampling = SamplingParams::default();

    println!("╔══════════════════════════════════════════╗");
    println!("║  {} — Chat                               ", model_name);
    println!("╚══════════════════════════════════════════╝");
    println!();
    println!("System: {}", system_prompt.lines().next().unwrap_or(""));
    println!();
    print_mcp_status(&mcp_cfg);
    println!();
    println!("Type your message and press Enter. Ctrl+C to exit.");
    println!("─────────────────────────────────────────────────");
    println!();

    let mut history = format!("<system>\n{system_prompt}\n</system>\n\n");

    let stdin = io::stdin();
    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
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

        let response = match &engine {
            Some(e) => {
                print!("Quark: ");
                io::stdout().flush()?;
                match e.generate(&history, sampling.clone()) {
                    Ok(text) => {
                        println!("{text}");
                        text
                    }
                    Err(err) => {
                        let msg = format!("[Generation error: {err}]");
                        println!("{msg}");
                        msg
                    }
                }
            }
            None => {
                let msg = "[Model not loaded — export from Quark GUI after training to get a functional model.]";
                println!("Quark: {msg}");
                msg.to_string()
            }
        };

        let calls = parse_tool_calls(&response);
        for call in &calls {
            println!();
            println!("[MCP] Calling tool: {} {:?}", call.tool, call.args);
            let result = execute_tool(call, &mcp_cfg);
            let formatted = format_tool_result(&result);
            println!("[MCP] Result ({}):", if result.ok { "ok" } else { "error" });
            let preview: String = result.content.chars().take(500).collect();
            println!("{preview}");
            history.push_str(&formatted);
            history.push('\n');
        }

        history.push_str(&format!("{response}\n</assistant>\n\n"));
        println!();
    }

    println!("\nGoodbye!");
    Ok(())
}

fn print_mcp_status(cfg: &McpConfig) {
    println!("MCP Tools enabled:");
    println!("  read_file:    {}", if cfg.read_file { "✓" } else { "✗" });
    println!("  write_file:   {}", if cfg.write_file { "✓" } else { "✗" });
    println!("  list_dir:     {}", if cfg.list_dir { "✓" } else { "✗" });
    println!("  search_files: {}", if cfg.search_files { "✓" } else { "✗" });
    println!("  get_cwd:      {}", if cfg.get_cwd { "✓" } else { "✗" });
    println!("  run_shell:    {}", if cfg.run_shell { "✓" } else { "✗" });
    println!("  working_dir:  {}", cfg.working_dir.display());
}

fn print_help() {
    println!("Commands:");
    println!("  /clear  — clear conversation history");
    println!("  /mcp    — show MCP tool status");
    println!("  /help   — show this help");
    println!("  /exit   — quit");
}
