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
2. Runs inference with the loaded `InferenceEngine` and streams real tokens as `AgentEvent::Token`, falling back to a keyword-matching stub (`generate_stub_response`) when no model is bundled
3. Parses tool calls, executes them, emits `FileChanged` events for undo tracking, and feeds `<tool_result>` blocks back to the model for up to `MAX_TOOL_ROUNDS` rounds

Plan mode blocks mutating tools (`is_mutating_tool`) in code, not just in the prompt. Build mode allows full filesystem access. `run_shell` is off unless `--allow-shell` is passed or `mcp.json` enables it.

### GUI panels (`quark-gui/src/panels/`)

Each panel is a struct implementing a `ui(&mut self, ui: &mut egui::Ui)` method. `QuarkApp` in `app.rs` owns all panel instances and dispatches to the active one. The `DatasetPanel` has a background `update(ctx)` call for polling live download logs.

### Backend feature flags

The Burn backend is selected at compile time via Cargo features:
- `backend-cpu` → Burn's `Flex` (pure-Rust CPU; replaced ndarray in the 0.21 upgrade)
- `backend-wgpu` → `burn/wgpu` (Metal on macOS, Vulkan/WGPU elsewhere)
- `backend-cuda` → `burn/cuda` (NVIDIA sm_70+)

The backend is chosen at compile time (`quark-core/src/backend.rs`): CUDA if `backend-cuda` is enabled, otherwise WGPU if `backend-wgpu` is, otherwise Flex. Burn is 0.21; code only names backends through `backend.rs` aliases (tests use `InferBackend`). CI clippy-checks the wgpu and CUDA builds (compile only; no GPU on hosted runners).

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

## Model, training and inference notes

- `DecoderBlock` holds either a dense FFN or a `MoeBlock` (as `Option`s), never both. MoE uses sparse top-k dispatch and returns a Switch-style load-balancing loss via `forward_with_aux`.
- Training (`training/trainer.rs`) supports grad accumulation, global-norm clipping (`training/grad_clip.rs`), held-out eval (`TrainingEvent::Eval`), and resume from the latest `checkpoint-N.bin`. It writes `config.json` and `tokenizer.json` next to the checkpoints; `QuarkConfig::for_checkpoint` reads the config back.
- `start_training` runs `dispatch_training`, which picks the autodiff backend for `TrainerConfig::precision` (`Bf16` only in CUDA builds) and runs the generic `run_training_loop::<AB>`. It also wraps the backend in `BalancedCheckpointing` when `gradient_checkpointing` is on (the default). Panics in the training thread become `TrainingEvent::Error("Training crashed: …")`. Burn 0.21 has no autodiff `topk`, so the MoE mask uses `sort_descending`.
- `QuarkConfig::param_count()` is exact (a test checks it against `num_params()`). `training_memory_bytes` and `trainer::estimate_memory` give the rough memory estimate that the GUI and trainer show.
- `chat.rs` holds the one chat template (`render_prompt`, `render_training_segments`, `STOP_STRINGS`). quark-chat, quark-code, the GUI chat and SFT data all use it, and `SamplingParams::stop_strings` ends generation at the assistant's closing tag.
- `TrainingMode::FineTune { base_checkpoint }` runs chat SFT (`data/sft.rs`: JSONL `{"messages": [...]}`, loss only on assistant tokens via byte-range masks from `QuarkTokenizer::encode_with_offsets`). It takes the architecture and tokenizer from the base folder and writes to `<base>/finetune/`.
- Tokenizers are trained with the full `ByteLevel::alphabet()`. Older ones silently drop unseen characters, and `encode_with_offsets` reports this as an error.
- **Offloaded ("streamed") training** (`training/streamed.rs`): `TrainerConfig::offload` (`Auto` = when the in-memory estimate doesn't fit). Weights and optimizer state live in `memory::store::TensorStore` (an LRU RAM cache over per-stage safetensors files under `<output>/offload/`). Each step runs a forward pass stage by stage (saving each layer's input activations to a second store), then a layer-major backward that recomputes each stage with autodiff, and updates it right away with the per-stage optimizers in `training/optim.rs` (`AdamW` matches Burn's exactly; `AdamWCompact` = companded int8 m + bf16 v; `Adafactor`). Clipping is per stage (`max_grad_norm/sqrt(stages)`). Checkpoints are sharded (`checkpoint/sharded.rs`, `checkpoint-N/` dirs incl. optimizer state; only the last 2 are kept). `streamed_step_matches_in_memory_step` checks one streamed step against the in-memory maths to 1e-5.
- Model stages (`model/stages.rs`): `EmbedStage` / `DecoderBlock` / `HeadStage` keep `QuarkModel`'s parameter paths. `memory/stage.rs` moves them in and out of `StageTensors` via burn-store. Don't clone a model before its lazily initialised params have been read: the clone draws new random weights.
- Checkpoints use `checkpoint::CheckpointRecorder` (Burn `BinFileRecorder`, full precision, `.bin`). Optimizer state is not saved, so it restarts fresh on resume.
- Generation (`inference/generate.rs`) uses per-layer KV caches (`QuarkModel::forward_cached`). When the context window fills, it re-prefills the most recent half window.
- `quark-core/tests/train_e2e.rs` trains a tiny model end-to-end (train → eval → checkpoint → load → stream → resume → chat fine-tune → stop at `</assistant>`). Run it after touching the model, trainer, tokenizer or inference code.
