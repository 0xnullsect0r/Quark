#![allow(dead_code, unused_imports, unused_variables)]

use std::path::Path;

use anyhow::Result;

use super::safetensors::{load_checkpoint, TensorData};

/// Map a HuggingFace Llama/Mistral weight key to the Quark naming convention.
///
/// HF:    `model.layers.0.self_attn.q_proj.weight`
/// Quark: `layers.0.attn.q_proj.weight`
///
/// Returns `None` only for keys that cannot be meaningfully mapped and should
/// be silently dropped (there are none in the current mapping — unknown keys
/// are kept as-is with their original name).
fn hf_key_to_quark(hf_key: &str) -> String {
    // Strip the leading "model." prefix when present.
    let key = hf_key.strip_prefix("model.").unwrap_or(hf_key);

    key.replace("self_attn.", "attn.")
        .replace("mlp.", "ffn.")
        .replace("input_layernorm.", "input_norm.")
        .replace("post_attention_layernorm.", "post_attn_norm.")
}

/// Load weights from a HuggingFace model directory and remap key names to the
/// Quark convention.
///
/// Supports:
/// - a single `model.safetensors` file, and
/// - sharded `model-00001-of-NNNNN.safetensors` files.
pub fn import_hf_weights(model_dir: &Path) -> Result<Vec<TensorData>> {
    // Collect all .safetensors files in the directory.
    let mut paths: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();

    if paths.is_empty() {
        anyhow::bail!(
            "No .safetensors files found in {}",
            model_dir.display()
        );
    }

    // Sort for deterministic shard ordering.
    paths.sort();

    let mut all_tensors: Vec<TensorData> = Vec::new();
    for path in &paths {
        let tensors = load_checkpoint(path)?;
        for mut t in tensors {
            t.name = hf_key_to_quark(&t.name);
            all_tensors.push(t);
        }
    }

    tracing::info!(
        "Imported {} tensors from HuggingFace model dir {}",
        all_tensors.len(),
        model_dir.display()
    );
    Ok(all_tensors)
}
