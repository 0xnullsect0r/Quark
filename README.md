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
- [Offloading: training models bigger than memory](#offloading-training-models-bigger-than-memory)
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

`makepkg` clones the latest Quark source from GitHub and compiles it locally
with `cargo build --release --features backend-cpu`. The package is named
`quark-git` (AUR convention for packages built from git).

> Once Quark is on the AUR, you can also use an AUR helper: `yay -S quark-git`

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

All presets use the same GQA + MoE architecture. The GUI's preset picker shows the exact parameter count and a training-memory estimate for each. The **Training** tab shows the plan for your machine: in memory or offloaded, how much device memory and disk it needs.

| Preset | Params (active/token) | Layers | Hidden | Context | Experts (top-k) | How it trains |
|--------|------:|-------:|-------:|--------:|----------------:|---------------|
| **Quark Tiny**    | 12 M | 6 | 256 | 512 | 4 (2) | in memory, ≈ 0.4 GB — laptop CPU is fine |
| **Quark Small**   | 224 M | 12 | 768 | 1024 | 8 (2) | in memory, ≈ 8 GB — a GPU with 8 GB+ |
| **Quark 10B-A2B** | 10.0 B (2.1 B) | 30 | 3072 | 2048 | 16 (2), every layer | **offloaded**: ≈ 6 GB on the GPU at a time, ≈ 75 GB of SSD (+ checkpoints) |
| Quark 1B … 400B   | see picker | | | | | larger ones need offloading or big hardware |

The 10B-A2B preset is the largest meant for a single machine. It is a mixture of experts: 10 B parameters, but each token only uses about 2 B of them, which keeps compute per token near that of a 2 B model. The older 1B … 400B names are nominal: the picker shows the real counts.

On CUDA builds, **bf16** precision (Training tab) halves memory. Master weights and checkpoints stay f32.

---

## Backends

The backend is chosen at build time: CUDA if `backend-cuda` is enabled, otherwise WGPU if `backend-wgpu` is enabled, otherwise the CPU.

| Backend | Hardware | Feature flag | Notes |
|---------|----------|--------------|-------|
| **CPU** (Burn Flex) | Any x86-64 / ARM64 | `backend-cpu` | Default; SIMD + multithreaded |
| **WGPU / Metal** | Apple Silicon, AMD/Intel GPU | `backend-wgpu` | Recommended for macOS |
| **CUDA** | NVIDIA GPU (sm_70+) | `backend-cuda` | Best performance on NVIDIA |

Pre-built releases ship the `backend-cpu` binary. Build from source with `backend-wgpu` or `backend-cuda` for GPU acceleration.

---

## Offloading: training models bigger than memory

With **Offloading** set to *Auto* (Training tab), a model that doesn't fit in RAM/VRAM is trained **layer by layer**:

1. Weights (f32 master copies) and optimizer state live in an offload folder on disk, with the most recently used layers cached in RAM (the RAM limit is in **Settings**).
2. **Forward:** each layer is loaded onto the GPU (or CPU) in turn, the batch is run through it, and only the layer's *input* activations are kept (spilling to disk if needed).
3. **Backward**, last layer first: each layer is reloaded and recomputed from its saved input to get gradients. That layer is then **updated immediately** and written back, so the full model's gradients never exist at once.

Only one layer's weights, gradients and optimizer state are on the device at a time: ≈ 6 GB for the 10B preset. A test checks that one offloaded step matches the in-memory trainer to within float tolerance.

- **Optimizers:** *AdamW* (8 bytes of state per parameter), *AdamW compact* (≈ 3 bytes: int8 first moment + bf16 second moment, tracks AdamW closely) or *Adafactor* (almost no state). For 10B, *AdamW compact* is the sensible default.
- **Checkpoints** are `checkpoint-N/` folders (one file per layer, optimizer state included; the last 2 are kept). Resuming restores the optimizer state.
- **Disk:** for 10B-A2B with AdamW compact, ≈ 40 GB of weights + ≈ 30 GB of optimizer state + activations, plus ≈ 70 GB per kept checkpoint. **Use an SSD**: every step reads and writes all of it.
- **Gradient clipping** is per layer when offloading (each layer's gradient is capped so the total can't exceed *Max grad norm*).

### Honest expectations for 10B

- **It fits and it trains:** one 10B layer at full size (2048 tokens) was measured at 4.3 GB of memory for forward + backward.
- **It is slow:** on a 4-core CPU one layer's forward + backward takes ≈ 32 s, so ≈ 20 minutes per 2048-token step for the whole model. A modern GPU is 50–100× faster at the compute. Disk I/O (~150 GB per step at 10B) then becomes the limit.
- **Pretraining a 10B model to a *useful* quality from scratch needs on the order of 10²² operations**, which is years on any single machine. Expect to use the 10B preset for experiments, continued training, or fine-tuning, and the Tiny/Small presets for complete from-scratch runs.

Run `quark-cli bench 10b --layer-only` (or `quark-cli bench small`) to measure your own machine.

---

## GUI Panels

| Panel | What it does |
|-------|-------------|
| **Config** | Select a model preset or fully customize every architecture parameter (layers, hidden size, heads, experts, context length, dtype). |
| **Dataset** | Add files or folders; preview tokenized samples; set train/validation split; optionally download and build **The Pile** (see below). |
| **Training** | Start / pause / stop training. Live loss and learning-rate charts. Tokens/sec throughput, ETA to completion, gradient norm, eval loss, process RAM (and VRAM on CUDA), and a memory estimate before you start. |
| **Checkpoints** | Browse all saved checkpoints with timestamps and loss values. Load any checkpoint to resume training or run inference. Export weights as `.safetensors`. |
| **Chat** | Stream tokens from your trained model. Adjust temperature, top-p, top-k, and max tokens. Edit the system prompt. |
| **Settings** | Configure resource limits, hardware backend, disk offload path, theme, and log level. |
| **Export** | Package your model into a standalone **quark-chat** REPL or **quark-code** AI coding agent. |

---

## Training Guide

1. **Configure** — open the **Config** panel and pick a preset. Start with **Quark Tiny** to check that everything works, then move to **Small** on a GPU.
2. **Load data** — open **Dataset**, click **Add Files/Folder** and point Quark at your code corpus. Quark will tokenize and pack sequences automatically.
3. **Set resource limits** — open **Settings** and drag the VRAM / RAM / CPU sliders to leave headroom for other applications.
4. **Start training** — open **Training** and click **▶ Start**. Quark runs the training loop on a background thread; the UI stays responsive.
5. **Monitor** — watch the loss curve converge. The ETA and tokens/sec update in real time. Checkpoints auto-save every N steps (configurable).
6. **Fine-tune for chat and tools** — a pretrained model only continues text. To make it answer questions and use `quark-code`'s tools, fine-tune it on conversations (next section).
7. **Chat** — load a checkpoint in **Checkpoints**, then open **Chat**.

### Chat & tool-use fine-tuning

In the **Training** tab, choose **Fine-tune on chat data**, pick a base `checkpoint-N.bin`, and add one or more `.jsonl` files with one conversation per line:

```json
{"messages": [
  {"role": "system", "content": "You are Quark Code…"},
  {"role": "user", "content": "Where is parse_config defined?"},
  {"role": "assistant", "content": "<tool_call>{\"tool\": \"grep_code\", \"pattern\": \"fn parse_config\"}</tool_call>"},
  {"role": "tool", "content": "grep_code (ok):\nsrc/config.rs:12:pub fn parse_config(…)"},
  {"role": "assistant", "content": "It's in src/config.rs at line 12."}
]}
```

- **Roles:** `system`, `user`, `assistant` and `tool` (a tool's output, written as `name (ok|error):` followed by the output).
- **What's trained:** only the assistant turns, so the model learns to answer, to emit `<tool_call>` blocks, and to end its turn.
- **Template:** conversations use the same chat template (`quark-core/src/chat.rs`) that `quark-chat`, `quark-code` and the Chat panel use at inference time.
- **Where it writes:** the architecture and tokenizer come from the base checkpoint's folder. The output goes to `<base>/finetune/`, so the base checkpoint is never overwritten.
- **Learning rate:** the default drops to 5e-5 when you switch to fine-tuning.
- **Sample data:** [`examples/chat-sft-sample.jsonl`](examples/chat-sft-sample.jsonl) has 30 example conversations (Q&A plus multi-step tool use) showing the format. It is far too small to teach a model on its own. Real instruction-following needs thousands to hundreds of thousands of conversations. You can convert public chat datasets (for example OpenAssistant or Dolly) into this format, and add tool-use examples from your own projects.

> **Tokenizer note:** tokenizers now always include all 256 byte values, so any text can be encoded. Tokenizers trained with older Quark versions only knew the characters in their corpus and silently drop anything else, including the `<`/`>` in chat tags. Retrain the tokenizer (and pretrain again) before fine-tuning. Quark reports conversations that can't be encoded.

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

The RAM % limit sizes the in-RAM cache used when offloading, and the disk offload path is where offloaded weights, optimizer state and activations go (relative paths are inside the training output folder). Put it on an SSD.

---

## Exporting Your Model

Once training is complete, open the **Export** panel to package your model as a standalone application. Choose the weight format:

| Weights | Size vs f32 | 10B-A2B size | Notes |
|---------|------------:|-------------:|-------|
| f32 | 1× | ≈ 40 GB | exact |
| 8-bit | ≈ ¼ | ≈ 11 GB | near-identical output |
| 4-bit | ≈ ⅐ | ≈ 6 GB | small quality loss; the one to use for 10B on a laptop |

Quantized models load almost instantly: weights are memory-mapped. On the CPU build a hand-written 4-bit kernel does the decoding. It measured ≈ 9.5 billion weights/s on 4 cores, so a 10B-A2B model (≈ 2 B weights per token) generates a few tokens per second. It also runs when the model is larger than free RAM, because the OS pages weights in from disk. On GPU builds, the quantized weights stay on the GPU and are unpacked just before each layer's matmul.

The same export is available headless:

```bash
quark-cli export ~/.quark/checkpoints/checkpoint-2000 ./my-model --q4   # or --q8, or neither for f32
quark-cli infer  ./my-model "Write a Rust function that reverses a string"
```

### quark-chat

A lightweight terminal REPL that runs your model locally. It loads the model from the `model/` folder next to the executable (`model/checkpoint/`, written by the Export panel), streams replies, and can call the enabled MCP tools. Commands: `/clear`, `/mcp`, `/help`, `/exit`.

The exported bundle includes the weights, tokenizer and config. No Python, no internet access, no dependencies — just run the binary.

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
| `run_shell` | Execute a shell command and capture output (off by default; pass `--allow-shell` or enable it in `model/mcp.json`) |
| `git_status` | Show `git status` |
| `git_diff` | Show `git diff` (staged or unstaged) |
| `git_log` | Show recent commit log |
| `git_add` | Stage files |
| `git_commit` | Commit staged changes |
| `grep_code` | Regex search across the codebase |
| `find_files` | Glob-pattern file search |
| `read_lines` | Read specific line ranges from a file |
| `write_lines` | Replace specific line ranges in a file |

In **Plan** mode, tools that change the project (`write_file`, `write_lines`, `apply_diff`, `git_add`, `git_commit`, `run_shell`) are blocked. Tool results go back to the model, which can make further tool calls (up to 8 rounds per turn) before it gives its answer.

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

