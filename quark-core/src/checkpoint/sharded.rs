//! Sharded checkpoints: a directory with one safetensors file per model stage,
//! so models far larger than RAM can be saved and loaded a stage at a time.
//!
//! ```text
//! checkpoint-1200/
//!   meta.json            {"format": "quark-sharded-v1", "step": 1200}
//!   config.json          QuarkConfig
//!   tokenizer.json
//!   embed.safetensors
//!   layer.0000.safetensors … layer.NNNN.safetensors
//!   head.safetensors
//!   optim/…              optimizer state (streamed training only)
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use crate::memory::stage::{load_stage, module_to_stage};
use crate::memory::store::{deserialize, serialize, StageTensors};
use crate::model::{
    block::DecoderBlock,
    config::QuarkConfig,
    stages::{layer_stage, EmbedStage, HeadStage, EMBED_STAGE, HEAD_STAGE},
    QuarkModel,
};

pub const SHARDED_FORMAT: &str = "quark-sharded-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardedMeta {
    pub format: String,
    pub step: u64,
}

/// Whether `path` is a sharded checkpoint directory.
pub fn is_sharded(path: &Path) -> bool {
    path.is_dir() && path.join("meta.json").exists()
}

pub fn stage_path(dir: &Path, stage: &str) -> PathBuf {
    dir.join(format!("{stage}.safetensors"))
}

pub fn write_stage(dir: &Path, stage: &str, data: &StageTensors) -> Result<()> {
    let path = stage_path(dir, stage);
    let tmp = path.with_extension("safetensors.tmp");
    std::fs::write(&tmp, serialize(data)?).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

pub fn read_stage(dir: &Path, stage: &str) -> Result<StageTensors> {
    let path = stage_path(dir, stage);
    deserialize(&std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?)
}

pub fn read_meta(dir: &Path) -> Result<ShardedMeta> {
    let meta: ShardedMeta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
    anyhow::ensure!(meta.format == SHARDED_FORMAT, "unknown checkpoint format {}", meta.format);
    Ok(meta)
}

/// Write `meta.json` and `config.json` (the stage files are written
/// separately, by [`save_sharded`] or the streamed trainer).
pub fn write_meta(dir: &Path, cfg: &QuarkConfig, step: u64) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let meta = ShardedMeta { format: SHARDED_FORMAT.to_owned(), step };
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(cfg)?)?;
    std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}

/// Save an in-memory model as a sharded checkpoint.
pub fn save_sharded<B: Backend>(
    model: &QuarkModel<B>,
    cfg: &QuarkConfig,
    dir: &Path,
    step: u64,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    // Collect from the model itself: cloning a model whose parameters are
    // still lazily initialised would draw fresh random weights.
    for (stage, data) in split_into_stages(module_to_stage(model)?, cfg)? {
        write_stage(dir, &stage, &data)?;
    }
    // meta last: a directory without meta.json is an incomplete checkpoint
    write_meta(dir, cfg, step)
}

/// Split a whole model's tensors (paths as in [`QuarkModel`]) into per-stage
/// tensors with stage-relative paths.
pub fn split_into_stages(
    model: StageTensors,
    cfg: &QuarkConfig,
) -> Result<Vec<(String, StageTensors)>> {
    let mut stages: Vec<(String, StageTensors)> = crate::model::stages::stage_names(cfg)
        .into_iter()
        .map(|name| (name, StageTensors::default()))
        .collect();
    let n_layers = cfg.num_hidden_layers;
    for (path, data) in model.tensors {
        let (index, name) = if path.starts_with("embed_tokens.") {
            (0, path)
        } else if let Some(rest) = path.strip_prefix("layers.") {
            let (i, name) = rest.split_once('.').context("bad layer path")?;
            let i: usize = i.parse()?;
            anyhow::ensure!(i < n_layers, "layer {i} out of range");
            (1 + i, name.to_owned())
        } else if path.starts_with("norm.") || path.starts_with("lm_head.") {
            (n_layers + 1, path)
        } else {
            anyhow::bail!("unexpected parameter {path}");
        };
        stages[index].1.tensors.push((name, data));
    }
    Ok(stages)
}

/// Load a whole sharded checkpoint into memory on `device`.
pub fn load_sharded<B: Backend>(dir: &Path, device: &B::Device) -> Result<(QuarkConfig, QuarkModel<B>)> {
    read_meta(dir)?;
    let cfg: QuarkConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;

    let mut embed = EmbedStage::new(&cfg, device);
    load_stage(&mut embed, &read_stage(dir, EMBED_STAGE)?)?;
    let layers = (0..cfg.num_hidden_layers)
        .map(|i| {
            let mut layer = DecoderBlock::new(&cfg, cfg.is_moe_layer(i), device);
            load_stage(&mut layer, &read_stage(dir, &layer_stage(i))?)
                .with_context(|| format!("layer {i}"))?;
            Ok(layer)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut head = HeadStage::new(&cfg, device);
    load_stage(&mut head, &read_stage(dir, HEAD_STAGE)?)?;

    Ok((cfg, QuarkModel::from_stages(embed, layers, head)))
}

/// Latest complete `checkpoint-N/` directory under `dir`.
pub fn latest_sharded(dir: &Path) -> Option<(PathBuf, u64)> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let step = path.file_name()?.to_str()?.strip_prefix("checkpoint-")?.parse().ok()?;
            is_sharded(&path).then_some((path, step))
        })
        .max_by_key(|(_, step)| *step)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Int, Tensor, TensorData};

    use super::*;
    use crate::backend::InferBackend as B;

    #[test]
    fn sharded_roundtrip_matches_model() {
        let cfg = QuarkConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 64,
            max_position_embeddings: 16,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_layer_freq: 2,
            ..QuarkConfig::quark_tiny()
        };
        let device = Default::default();
        let model = QuarkModel::<B>::new(&cfg, &device);
        let dir = std::env::temp_dir().join(format!("quark-sharded-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let ckpt = dir.join("checkpoint-7");
        save_sharded(&model, &cfg, &ckpt, 7).unwrap();

        assert!(is_sharded(&ckpt));
        assert_eq!(latest_sharded(&dir).unwrap().1, 7);
        let (loaded_cfg, loaded) = load_sharded::<B>(&ckpt, &device).unwrap();
        assert_eq!(loaded_cfg.num_hidden_layers, 3);

        let ids = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![1, 5, 9, 3], [1, 4]), &device);
        let a: Vec<f32> = model.forward(ids.clone()).into_data().to_vec().unwrap();
        let b: Vec<f32> = loaded.forward(ids).into_data().to_vec().unwrap();
        assert_eq!(a, b);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
