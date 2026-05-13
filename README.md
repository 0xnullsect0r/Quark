# Quark

**Quark** is a downloadable desktop app that lets you train and run your own Llama 4-style Mixture-of-Experts (MoE) coding LLM — entirely on your own hardware, no cloud required.

---

## What is Quark?

Quark packages a full training + inference loop into a native GUI app. You bring your code, pick a model preset, and Quark handles the rest: data ingestion, training, checkpointing, and an interactive chat session with your finished model.

---

## Architecture

Quark implements a modern transformer architecture mirroring the design of Llama 4:

| Component | Details |
|-----------|---------|
| Attention | Grouped-Query Attention (GQA) |
| FFN | Mixture-of-Experts (MoE) with top-k routing |
| Activation | SwiGLU |
| Positional encoding | Rotary Position Embeddings (RoPE) |
| Normalization | RMSNorm (pre-norm) |

---

## Model Presets

| Preset | Layers | Hidden dim | Experts | Top-k | Target use |
|--------|--------|------------|---------|-------|-----------|
| **Quark-1B** | 16 | 2048 | 8 | 2 | 8–16 GB RAM, fast iteration |
| **Quark-3B** | 28 | 3072 | 8 | 2 | 16–32 GB RAM, higher quality |

Both presets are tuned for coding tasks. Custom architectures can be configured in the **Config** panel.

---

## Backends

| Backend | Platforms | Feature flag |
|---------|-----------|--------------|
| CPU (ndarray) | Windows, macOS, Linux | `backend-cpu` |
| WGPU / Metal | macOS, Linux | `backend-wgpu` |
| CUDA | NVIDIA GPUs | `backend-cuda` |

Multiple backends can be compiled in simultaneously. Quark selects the fastest available backend at runtime.

---

## Memory Tiering

Quark implements a three-tier memory system so you can train models that are larger than your VRAM:

```
VRAM  →  RAM  →  Disk
```

Active layers live in VRAM. Layers that don't fit spill to RAM, then to a memory-mapped disk buffer. Offloading is transparent — no manual configuration beyond the resource limits in Settings.

---

## Resource Limits

All resource usage is controlled from the **Settings** panel:

| Setting | Description |
|---------|-------------|
| VRAM % | Maximum fraction of GPU memory Quark may use |
| RAM % | Maximum fraction of system RAM Quark may use |
| CPU thread % | How many logical cores to hand to the training workers |
| GPU compute % | Fraction of GPU compute budget reserved for Quark |

Reduce these to keep your machine responsive while training runs in the background.

---

## Quick Start

1. Download the latest binary for your platform from the [Releases](../../releases) page.
2. Launch `quark-gui` (or `quark-windows-amd64.exe` on Windows).
3. Open the **Config** panel, pick a preset (**Quark-1B** or **Quark-3B**), and review the architecture settings.
4. Open the **Dataset** panel, click **Add Files / Folder**, and point Quark at your code.
5. Open the **Training** panel and click **Start Training**. The loss and learning-rate charts update live.
6. Checkpoints are saved automatically — manage them in the **Checkpoints** panel.
7. Once training converges, open the **Chat** panel to run inference against your model.

---

## Build from Source

### Prerequisites

- Rust stable toolchain (`rustup update stable`)
- On Ubuntu/Debian: system libraries for the GUI

```bash
sudo apt-get install -y \
  libgtk-3-dev libxcb-render0-dev libxcb-shape0-dev \
  libxcb-xfixes0-dev libxkbcommon-dev libssl-dev
```

### Compile

```bash
# CPU only (all platforms)
cargo build --release --package quark-gui --features backend-cpu

# GPU (WGPU / Metal — macOS and Linux)
cargo build --release --package quark-gui --features "backend-cpu backend-wgpu"

# CUDA (NVIDIA)
cargo build --release --package quark-gui --features "backend-cpu backend-cuda"
```

The binary is written to `target/release/quark-gui` (`quark-gui.exe` on Windows).

---

## GUI Panels

| Panel | Purpose |
|-------|---------|
| **Config** | Model architecture (preset or custom), context length, dtype |
| **Dataset** | Add/remove training files, preview tokenised samples, set split ratio |
| **Training** | Start/stop training, live loss chart, live LR chart, epoch progress |
| **Checkpoints** | Browse saved checkpoints, load a checkpoint, export weights |
| **Chat** | Interactive inference with your trained model, temperature/top-p sliders |
| **Settings** | Resource limits (VRAM %, RAM %, CPU %, GPU %), theme, log level |

---

## License

[MIT](LICENSE) © 2025 Quark Contributors
