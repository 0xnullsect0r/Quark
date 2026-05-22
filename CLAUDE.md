# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Quark is a desktop application for training and running Llama 4-style Mixture-of-Experts LLMs entirely on local hardware — no cloud required. It ships three binaries: `quark` (GUI), `quark-chat` (terminal REPL), and `quark-code` (AI coding agent TUI).

## Build Commands

All builds require a backend feature flag. `backend-cpu` works everywhere; `backend-wgpu` adds GPU on macOS/Linux; `backend-cuda` adds NVIDIA GPU support.

```bash
# Development build (fast compile, opt-level 1)
cargo build --package quark-gui --features backend-cpu

# Release build (lto, strip, codegen-units=1 — slow)
cargo build --release --package quark-gui --features backend-cpu

# Build all three binaries
cargo build --release --features backend-cpu

# GPU variants
cargo build --release --package quark-gui --features "backend-cpu backend-wgpu"   # macOS/Linux
cargo build --release --package quark-gui --features "backend-cpu backend-cuda"   # NVIDIA
```

Binaries land in `target/release/`: `quark`, `quark-chat`, `quark-code`.

## Development Commands

```bash
# Type-check the whole workspace
cargo check --workspace --features backend-cpu

# Lint (CI enforces zero warnings)
cargo clippy --workspace --features backend-cpu -- -D warnings

# Run all tests
cargo test --workspace --features backend-cpu

# Run a single test by name
cargo test --workspace --features backend-cpu -- test_parse_tool_calls

# Format
cargo fmt --all
```

Linux dev machines need system libs before any GUI build:
```bash
sudo apt-get install -y libgtk-3-dev libxcb-render0-dev libxcb-shape0-dev \
  libxcb-xfixes0-dev libxkbcommon-dev libssl-dev
```

## Architecture

### Workspace Layout

```
quark-core/   — shared library: model, training, inference, MCP tools, paths, updater
quark-gui/    — egui/eframe desktop GUI (panels for config, dataset, training, chat, export)
quark-chat/   — terminal REPL wrapper around quark-core inference
quark-code/   — ratatui TUI coding agent with MCP tool loop
quark-cli/    — thin CLI entry point
```

`quark-core` is the single source of truth for all ML logic. All other crates depend on it.

### quark-core modules

| Module | Purpose |
|---|---|
| `model` | GQA + MoE transformer (Burn backend) |
| `training` | Training loop, optimizer, gradient checkpointing |
| `inference` | Token generation, sampling |
| `data` | Dataset loading and tokenization |
| `checkpoint` | Safetensors serialization |
| `tokenizer` | HuggingFace BPE tokenizer wrapper |
| `memory` | Three-tier VRAM/RAM/disk spilling |
| `mcp` | MCP tool definitions and XML tool-call parser |
| `paths` | Platform-specific app data directories |
| `updater` | Background GitHub release check |

### MCP Tool System (`quark-core/src/mcp.rs`)

The model emits tool calls as `<tool_call>{"tool":"name",...}</tool_call>` XML in its output. `parse_tool_calls()` scans raw text for these tags and deserializes them. `execute_tool()` dispatches on the tool name and respects the `McpConfig` enable flags. `run_shell` is disabled by default.

`quark-code` extends this with additional git tools via `execute_extended()` in `quark-code/src/tools.rs`.

### quark-code agent loop (`quark-code/src/agent.rs`)

`start_turn()` spawns a background thread that:
1. Builds a system prompt injecting available tools, current mode (Plan/Build), and `AGENTS.md` context
2. Runs inference (currently a keyword-matching stub — real model wiring is a TODO)
3. Streams `AgentEvent` tokens back over `mpsc::channel`
4. Parses tool calls from the response, executes them, and emits `FileChanged` events for undo tracking

Plan mode blocks all write/git tools. Build mode allows full filesystem access.

### GUI panels (`quark-gui/src/panels/`)

Each panel is a struct implementing a `ui(&mut self, ui: &mut egui::Ui)` method. `QuarkApp` in `app.rs` owns all panel instances and dispatches to the active one. The `DatasetPanel` has a background `update(ctx)` call for polling live download logs.

### Backend feature flags

The Burn backend is selected at compile time via Cargo features:
- `backend-cpu` → `burn-ndarray`
- `backend-wgpu` → `burn-wgpu` (Metal on macOS, Vulkan/WGPU elsewhere)
- `backend-cuda` → `burn-cuda` (NVIDIA sm_70+)

Multiple backends can coexist; the fastest available is selected at runtime.

### Data directories

`quark-core::paths` provides canonical platform paths:
- Linux: `~/.quark/`
- macOS: `~/Library/Application Support/Quark/`
- Windows: `%APPDATA%\Quark\`

Subdirectories: `checkpoints/`, `datasets/`, `the-pile/`, `settings.toml`.

## Code Conventions

- `rustfmt.toml`: `max_width=100`, `imports_granularity="Crate"`, `group_imports="StdExternalCrate"`
- Many files carry `#![allow(dead_code, unused_imports)]` — this is intentional while the ML pipeline is being wired up; don't remove these attributes unless the code is actually complete
- Error handling uses `anyhow` for applications and `thiserror` for library types
- Logging uses `tracing`; log level is controlled by `RUST_LOG` env var (default: `quark=info`)

## Key TODOs in the Codebase

The inference path in `quark-code/src/agent.rs` is a heuristic stub (`generate_stub_response`). The comment at line 121 marks where real `quark-core` inference needs to be wired in once model weights are loadable.
