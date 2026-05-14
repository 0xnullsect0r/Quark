# Quark

**Train and run your own Llama 4-style MoE coding LLM — entirely on your own hardware.**

[![CI](https://github.com/0xnullsect0r/Quark/actions/workflows/ci.yml/badge.svg)](https://github.com/0xnullsect0r/Quark/actions/workflows/ci.yml)
[![Release](https://github.com/0xnullsect0r/Quark/actions/workflows/release.yml/badge.svg)](https://github.com/0xnullsect0r/Quark/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Quark is a downloadable desktop GUI that guides you from raw data → trained LLM → deployable app with no cloud account, no subscriptions, and no vendor lock-in. Pick a model size, point it at your dataset, hit **Start Training**, and walk away. When it's done, export a standalone **quark-chat** REPL or a full **quark-code** AI coding agent.

---

## Table of Contents

- [Installation](#installation)
- [Architecture](#architecture)
- [Model Presets](#model-presets)
- [Backends](#backends)
- [Memory Tiering](#memory-tiering)
- [GUI Panels](#gui-panels)
- [Training Guide](#training-guide)
- [The Pile Dataset](#the-pile-dataset)
- [Resource Limits & Settings](#resource-limits--settings)
- [Exporting Your Model](#exporting-your-model)
  - [quark-chat](#quark-chat)
  - [quark-code](#quark-code)
- [Build from Source](#build-from-source)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### macOS

Download **Quark-\<version\>-macos.dmg** from the [Releases](../../releases) page.

1. Open the DMG — drag **Quark.app** to `/Applications`.
2. `quark-chat` and `quark-code` are also in the DMG root; copy them to `/usr/local/bin` for CLI access.
3. On first launch macOS may show a security dialog — open **System Settings → Privacy & Security** and click **Open Anyway**.

### Windows

Download **Quark-\<version\>-windows-setup.exe** from the [Releases](../../releases) page.

1. Run the installer — choose which components to install:
   - **Quark GUI** (required) — the main training + inference app
   - **Quark Chat** — terminal REPL
   - **Quark Code** — AI coding agent (optionally added to `PATH`)
2. Optional: enable "Add Quark Code to PATH" to use `quark-code` from any terminal.
3. A desktop shortcut and Start Menu entry are created for the GUI.

### Linux — .deb (Debian / Ubuntu)

```bash
# Download the .deb from the Releases page, then:
sudo dpkg -i quark_0.1.0_amd64.deb
sudo apt-get install -f          # resolve any missing dependencies
```

### Linux — .rpm (Fedora / RHEL / openSUSE)

```bash
# Download the .rpm from the Releases page, then:
sudo rpm -i quark-0.1.0-1.x86_64.rpm
# or with dnf:
sudo dnf localinstall quark-0.1.0-1.x86_64.rpm
```

### Linux — Arch (AUR / makepkg)

```bash
# Clone the PKGBUILD from this repo, then:
cd installer/arch
makepkg -si
```

`makepkg` will automatically resolve the **latest GitHub release** at build time — no
manual version bumping required. The package is named `quark-bin` (AUR convention for
pre-built binaries).

> Once Quark is on the AUR, you can also use an AUR helper: `yay -S quark-bin`

### Linux — Raw binary

Download **quark-linux-amd64** from the [Releases](../../releases) page and make it executable:

```bash
chmod +x quark-linux-amd64
./quark-linux-amd64
```

---

## Architecture

Quark implements a **Llama 4-style Transformer + Mixture-of-Experts** decoder architecture in pure Rust using the [Burn](https://burn.dev) deep learning framework.

| Component | Implementation |
|-----------|---------------|
| **Attention** | Grouped-Query Attention (GQA) — fewer KV heads than Q heads, reducing KV cache memory |
| **FFN** | Mixture-of-Experts (MoE) — sparse top-K routing, only K of N experts activate per token |
| **Activation** | SwiGLU (Swish-gated Linear Unit) inside each expert |
| **Position** | Rotary Position Embeddings (RoPE) — relative position encoding in attention |
| **Normalization** | RMSNorm (pre-norm placement, no bias) |
| **Dtype** | bfloat16 on CUDA/Metal; float32 on CPU |
| **Vocab** | BPE tokenizer (HuggingFace `tokenizers` format) |

### MoE Routing

Each transformer block alternates between **dense SwiGLU** layers and **sparse MoE** layers. The router is a learned linear projection that produces per-expert logits; a softmax + top-K select the active experts. An auxiliary load-balancing loss encourages uniform expert utilization.

---

## Model Presets

All presets use the same GQA + MoE architecture. Hardware estimates assume `bfloat16` weights, `batch=1` inference.

| Preset | Params | Layers | Hidden | Heads | KV heads | MoE layers | Experts | Top-K | Min VRAM | Min RAM |
|--------|-------:|-------:|-------:|------:|---------:|-----------:|--------:|------:|---------:|--------:|
| **Quark-1B**   |   1 B |  16 |  2048 | 16 |  4 |  4 |  8 | 2 |  2 GB |  4 GB |
| **Quark-3B**   |   3 B |  28 |  3072 | 24 |  8 |  6 |  8 | 2 |  6 GB |  8 GB |
| **Quark-7B**   |   7 B |  32 |  4096 | 32 |  8 |  8 |  8 | 2 |  14 GB | 16 GB |
| **Quark-20B**  |  20 B |  40 |  5120 | 40 | 10 | 10 | 16 | 4 |  40 GB | 48 GB |
| **Quark-30B**  |  30 B |  48 |  6144 | 48 | 12 | 12 | 16 | 4 |  60 GB | 80 GB |
| **Quark-48B**  |  48 B |  56 |  7168 | 56 | 14 | 14 | 32 | 4 |  96 GB | 128 GB |
| **Quark-74B**  |  74 B |  64 |  8192 | 64 | 16 | 16 | 32 | 4 | 148 GB | 192 GB |
| **Quark-120B** | 120 B |  80 | 10240 | 80 | 20 | 20 | 64 | 8 | 240 GB | 320 GB |
| **Quark-249B** | 249 B |  96 | 12288 | 96 | 24 | 24 | 64 | 8 | 498 GB | 640 GB |
| **Quark-300B** | 300 B | 104 | 14336 | 112| 28 | 28 | 64 | 8 | 600 GB | 768 GB |
| **Quark-400B** | 400 B | 120 | 16384 | 128| 32 | 32 | 128| 8 | 800 GB |   1 TB |
| **Custom**     |  —    |   — |    — |  — |   — |   — |   — | — | — | — |

> **Tip:** For consumer hardware (≤ 24 GB VRAM) start with **Quark-1B** or **Quark-3B**. Enable memory tiering in Settings to spill layers to RAM/disk and train models that exceed your VRAM.

---

## Backends

Quark selects the fastest backend available at runtime. Multiple backends can be compiled in simultaneously.

| Backend | Hardware | Feature flag | Notes |
|---------|----------|--------------|-------|
| **CPU** (ndarray) | Any x86-64 / ARM64 | `backend-cpu` | Default; AVX2 auto-detected |
| **WGPU / Metal** | Apple Silicon, AMD/Intel GPU | `backend-wgpu` | Recommended for macOS |
| **CUDA** | NVIDIA GPU (sm_70+) | `backend-cuda` | Best performance on NVIDIA |

Pre-built releases ship the `backend-cpu` binary. Build from source with `backend-wgpu` or `backend-cuda` for GPU acceleration.

---

## Memory Tiering

Quark implements a three-tier memory system that lets you train models larger than your VRAM — or even larger than your RAM.

```
┌──────────┐    ┌──────────┐    ┌──────────────────────┐
│  VRAM    │ ←→ │  RAM     │ ←→ │  Disk (mmap)         │
│ active   │    │ inactive │    │ optimizer state /    │
│ layers   │    │ weights  │    │ weight shards        │
└──────────┘    └──────────┘    └──────────────────────┘
```

- **Layer streaming** — the next layer is prefetched to VRAM while the current one executes.
- **Gradient checkpointing** — activations are recomputed during backprop instead of stored.
- **Offloaded optimizer** — Adam m/v buffers (2× model size) live in RAM; only the current shard is on the GPU during the update step.
- **JIT quantization** — weights are stored as NF4 (4-bit) or INT8 on disk/RAM, dequantized to bf16 per-layer just before the forward pass.
- **Memory-mapped shards** — `.safetensors` weight files are `mmap`'d so the OS pages them in/out transparently.

---

## GUI Panels

| Panel | What it does |
|-------|-------------|
| **Config** | Select a model preset or fully customize every architecture parameter (layers, hidden size, heads, experts, context length, dtype). |
| **Dataset** | Add files or folders; preview tokenized samples; set train/validation split; optionally download and build **The Pile** (see below). |
| **Training** | Start / pause / stop training. Live loss and learning-rate charts. Tokens/sec throughput, ETA to completion, per-tier memory bars (VRAM / RAM / disk). |
| **Checkpoints** | Browse all saved checkpoints with timestamps and loss values. Load any checkpoint to resume training or run inference. Export weights as `.safetensors`. |
| **Chat** | Stream tokens from your trained model. Adjust temperature, top-p, top-k, and max tokens. Edit the system prompt. |
| **Settings** | Configure resource limits, hardware backend, disk offload path, theme, and log level. |
| **Export** | Package your model into a standalone **quark-chat** REPL or **quark-code** AI coding agent. |

---

## Training Guide

1. **Configure** — open the **Config** panel, pick a preset, and optionally tweak context length or dtype.
2. **Load data** — open **Dataset**, click **Add Files/Folder** and point Quark at your code corpus. Quark will tokenize and pack sequences automatically.
3. **Set resource limits** — open **Settings** and drag the VRAM / RAM / CPU sliders to leave headroom for other applications.
4. **Start training** — open **Training** and click **▶ Start**. Quark runs the training loop on a background thread; the UI stays responsive.
5. **Monitor** — watch the loss curve converge. The ETA and tokens/sec update in real time. Checkpoints auto-save every N steps (configurable).
6. **Chat** — once loss converges (or at any checkpoint), open **Chat** and start a conversation.

### Hyperparameters (Config panel)

| Parameter | Default | Notes |
|-----------|---------|-------|
| Learning rate | 3e-4 | Cosine decay with linear warmup |
| Warmup steps | 200 | |
| Batch size | 4 | Effective batch = batch × grad_accum |
| Grad accumulation | 8 | |
| Max seq length | 2048 | |
| Gradient clip | 1.0 | L2 norm clipping |

---

## The Pile Dataset

Quark has built-in support for [The Pile](https://github.com/EleutherAI/the-pile) — an 825 GiB diverse open-source text corpus from EleutherAI, excellent for general-purpose LLM pretraining.

### How to use it

1. Open the **Dataset** panel and select **The Pile** from the source dropdown.
2. Quark will:
   - Clone / update the Pile downloader scripts into your app-data directory
   - Present a component selector (choose subsets: code, books, Wikipedia, etc.)
   - Download and extract selected shards in the background
   - Show a progress bar with the current download step and a live scrolling log
3. Once downloaded, the dataset is cached and reused for future runs.

### Storage location

| Platform | Path |
|----------|------|
| Windows  | `%APPDATA%\Quark\datasets\pile\` |
| macOS    | `~/Library/Application Support/Quark/datasets/pile/` |
| Linux    | `~/.quark/datasets/pile/` |

> The full Pile is ~825 GB. You can download individual subsets (e.g., just "GitHub" for a coding-focused model) to save space.

---

## Resource Limits & Settings

All resource constraints are in the **Settings** panel and take effect immediately without restarting.

| Setting | Default | Description |
|---------|---------|-------------|
| **VRAM %** | 75% | Maximum fraction of GPU memory Quark may allocate |
| **RAM %** | 70% | Maximum fraction of system RAM Quark may allocate |
| **CPU thread %** | 80% | Fraction of logical cores handed to training workers |
| **GPU compute %** | 90% | Fraction of GPU compute budget reserved for Quark |
| **Disk offload path** | app-data dir | Where to write weight shards that don't fit in RAM |
| **Backend** | Auto | Force a specific backend (CPU / WGPU / CUDA) |
| **Theme** | System | Light / Dark / System |
| **Log level** | Info | Trace / Debug / Info / Warn / Error |

Reducing VRAM % or RAM % forces more layers to spill to the next tier (slower but prevents OOM).

---

## Exporting Your Model

Once training is complete, open the **Export** panel to package your model as a standalone application.

### quark-chat

A lightweight terminal REPL that spins up your model locally and lets you chat with it.

```
Usage: quark-chat [OPTIONS]

Options:
  --model <DIR>       Path to model directory (default: bundled)
  --system <TEXT>     System prompt override
  --temperature <F>   Sampling temperature (default: 0.7)
  --top-p <F>         Nucleus sampling threshold (default: 0.9)
  --max-tokens <N>    Max tokens per response (default: 512)
```

The exported binary includes the weights, tokenizer, and a hardened config. No Python, no internet access, no dependencies — just run the binary.

### quark-code

A full-featured AI coding agent for the terminal, inspired by Claude Code and GitHub Copilot CLI. It spins up your bundled Quark model locally and provides an interactive TUI.

```
Usage: quark-code [OPTIONS] [DIRECTORY]

Options:
  --model <DIR>    Path to model directory (default: bundled)
  --plan           Start in Plan mode (architect before coding)
```

#### Slash Commands

| Command | Description |
|---------|-------------|
| `/init` | Scan the project and write an `AGENTS.md` context file |
| `/plan` | Switch to Plan mode — architect changes before implementing |
| `/build` | Switch to Build mode — implement the current plan |
| `/diff` | Show a unified diff of all pending changes |
| `/undo` | Undo the last file change |
| `/redo` | Redo an undone change |
| `/mcp` | Show available MCP tools and their status |
| `/help` | Print all commands |
| `/exit` | Quit quark-code |

#### MCP Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read any file in the project |
| `write_file` | Write or overwrite a file |
| `list_dir` | List directory contents |
| `search_files` | Search file contents with a pattern |
| `run_shell` | Execute a shell command and capture output |
| `git_status` | Show `git status` |
| `git_diff` | Show `git diff` (staged or unstaged) |
| `git_log` | Show recent commit log |
| `git_add` | Stage files |
| `git_commit` | Commit staged changes |
| `grep_code` | Regex search across the codebase |
| `find_files` | Glob-pattern file search |
| `read_lines` | Read specific line ranges from a file |
| `write_lines` | Replace specific line ranges in a file |

#### Context Injection

Prefix a message with `@filename` to inject that file's content into the prompt:

```
> @src/main.rs refactor the argument parsing to use clap
```

---

## Build from Source

### Prerequisites

- Rust stable toolchain: `rustup update stable`
- Linux only — system libraries:

```bash
sudo apt-get install -y \
  libgtk-3-dev libxcb-render0-dev libxcb-shape0-dev \
  libxcb-xfixes0-dev libxkbcommon-dev libssl-dev
```

### Build

```bash
# Clone
git clone https://github.com/0xnullsect0r/Quark.git
cd Quark

# CPU-only (all platforms)
cargo build --release --package quark-gui --features backend-cpu

# GPU — WGPU/Metal (macOS & Linux)
cargo build --release --package quark-gui --features "backend-cpu backend-wgpu"

# GPU — CUDA (NVIDIA)
cargo build --release --package quark-gui --features "backend-cpu backend-cuda"

# Build companion CLIs
cargo build --release --package quark-chat --features backend-cpu
cargo build --release --package quark-code --features backend-cpu
```

Binaries are written to `target/release/`:
- `quark` — GUI application
- `quark-chat` — terminal REPL
- `quark-code` — AI coding agent

### Build Platform Installers

```bash
# macOS DMG (run on macOS)
bash build/macos/bundle.sh

# Linux .deb (run on Debian/Ubuntu)
bash build/linux/deb.sh

# Linux .rpm (run on Fedora/RHEL)
bash build/linux/rpm.sh

# Linux AppImage
bash build/linux/appimage.sh

# Windows NSIS installer (run on Windows with NSIS installed)
cd installer/windows && makensis quark.nsi
```

---

## Contributing

Pull requests welcome! Please:

1. Fork the repo and create a feature branch.
2. Run `cargo clippy --workspace --features backend-cpu -- -D warnings` and fix all warnings before submitting.
3. Add tests for new public API surface where practical.
4. Keep PRs focused — one logical change per PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) if present, or open an issue to discuss larger changes first.

---

## License

[MIT](LICENSE) © 2025 Quark Contributors

