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

use crate::memory::stage::{load_stage, load_stage_skipping, module_to_stage};
use crate::memory::store::{deserialize, serialize, StageTensors};
use crate::model::proj::QuantFormat;
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
    /// Set for inference exports whose projection weights are quantized
    /// (see `checkpoint::quantize`).
    #[serde(default)]
    pub quantization: Option<QuantFormat>,
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
    let meta = ShardedMeta { format: SHARDED_FORMAT.to_owned(), step, quantization: None };
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

/// Load a whole sharded checkpoint (plain or quantized) into memory on
/// `device`.
pub fn load_sharded<B: Backend>(dir: &Path, device: &B::Device) -> Result<(QuarkConfig, QuarkModel<B>)> {
    let meta = read_meta(dir)?;
    let cfg: QuarkConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let q = meta.quantization;

    let mut embed = EmbedStage::new(&cfg, device);
    load_stage(&mut embed, &read_stage(dir, EMBED_STAGE)?)?;
    let layers = (0..cfg.num_hidden_layers)
        .map(|i| {
            let mut layer = DecoderBlock::new(&cfg, cfg.is_moe_layer(i), device);
            match q {
                None => load_stage(&mut layer, &read_stage(dir, &layer_stage(i))?),
                Some(format) => {
                    let stage = MappedStage::open(dir, &layer_stage(i))?;
                    load_quantized_stage(&mut layer, &stage, format, device)
                }
            }
            .with_context(|| format!("layer {i}"))?;
            Ok(layer)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut head = HeadStage::new(&cfg, device);
    match q {
        None => load_stage(&mut head, &read_stage(dir, HEAD_STAGE)?)?,
        Some(format) => {
            let stage = MappedStage::open(dir, HEAD_STAGE)?;
            load_quantized_stage(&mut head, &stage, format, device)?
        }
    }

    Ok((cfg, QuarkModel::from_stages(embed, layers, head)))
}

/// A quantized stage file, memory-mapped: packed weight words are used in
/// place (zero-copy), other tensors are copied out.
struct MappedStage {
    map: std::sync::Arc<memmap2::Mmap>,
    /// Everything except `*.quant.packed`.
    tensors: StageTensors,
    /// `*.quant.packed` → (byte offset in `map`, words, `[out, words]` shape).
    packed: std::collections::HashMap<String, (usize, usize, [usize; 2])>,
}

impl MappedStage {
    fn open(dir: &Path, stage: &str) -> Result<Self> {
        let path = stage_path(dir, stage);
        let file = std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: checkpoint files are written once (by rename) and not modified in place.
        let map = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file)? });
        let st = safetensors::SafeTensors::deserialize(&map)?;
        let base = map.as_ptr() as usize;
        let mut tensors = Vec::new();
        let mut packed = std::collections::HashMap::new();
        for (name, view) in st.tensors() {
            if name.ends_with(".quant.packed") {
                let shape = view.shape();
                let offset = view.data().as_ptr() as usize - base;
                packed.insert(name, (offset, view.data().len() / 4, [shape[0], shape[1]]));
            } else {
                let dtype = match view.dtype() {
                    safetensors::Dtype::F32 => burn::tensor::DType::F32,
                    safetensors::Dtype::F16 => burn::tensor::DType::F16,
                    safetensors::Dtype::BF16 => burn::tensor::DType::BF16,
                    other => anyhow::bail!("unexpected dtype {other:?} for {name}"),
                };
                let data = burn::tensor::TensorData::from_bytes_vec(view.data().to_vec(), view.shape().to_vec(), dtype);
                tensors.push((name, data));
            }
        }
        drop(st);
        Ok(Self { map, tensors: StageTensors::new(tensors), packed })
    }

    fn host_quant(&self, format: QuantFormat, path: &str) -> Result<crate::model::proj::HostQuant> {
        use crate::model::proj::{HostQuant, Words};
        let key = format!("{path}.quant.packed");
        let &(offset, len, [out_features, words]) = self.packed.get(&key).with_context(|| format!("{key} missing"))?;
        let scales = self.tensors.get(&format!("{path}.quant.scales")).with_context(|| format!("{path}: scales missing"))?;
        let packed = match Words::mapped(std::sync::Arc::clone(&self.map), offset, len) {
            Some(words) => words,
            None => Words::Owned(bytemuck::pod_collect_to_vec(&self.map[offset..offset + len * 4])),
        };
        let groups = scales.shape.dims::<2>()[1];
        let in_features = words * format.per_word();
        let scales = scales.clone().convert::<f32>().to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(HostQuant { format, in_features, out_features, group: in_features / groups, packed, scales })
    }
}

/// Load a quantized stage: dense tensors through burn-store, then the packed
/// weights attached to each projection.
fn load_quantized_stage<B: Backend, M: burn::module::Module<B> + HasProjs<B>>(
    module: &mut M,
    stage: &MappedStage,
    format: QuantFormat,
    device: &B::Device,
) -> Result<()> {
    let proj_paths: Vec<String> = module.projs().into_iter().map(|(p, _)| p).collect();
    let dense = StageTensors::new(
        stage.tensors.tensors.iter().filter(|(n, _)| !n.contains(".quant.")).cloned().collect(),
    );
    load_stage_skipping(module, &dense, &proj_paths.iter().map(|p| format!("{p}.weight")).collect::<Vec<_>>())?;
    for (path, proj) in module.projs() {
        proj.set_quantized(stage.host_quant(format, &path)?, device);
    }
    Ok(())
}

/// Modules whose projections can be quantized.
pub trait HasProjs<B: Backend> {
    fn projs(&mut self) -> Vec<(String, &mut crate::model::proj::Proj<B>)>;
}

impl<B: Backend> HasProjs<B> for DecoderBlock<B> {
    fn projs(&mut self) -> Vec<(String, &mut crate::model::proj::Proj<B>)> {
        self.projs_mut()
    }
}

impl<B: Backend> HasProjs<B> for HeadStage<B> {
    fn projs(&mut self) -> Vec<(String, &mut crate::model::proj::Proj<B>)> {
        self.projs_mut()
    }
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

    #[test]
    fn quantized_stages_are_memory_mapped() {
        use crate::model::proj::{QuantFormat, Words};
        let cfg = QuarkConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 64,
            num_experts: 2,
            num_experts_per_tok: 1,
            moe_layer_freq: 1,
            ..QuarkConfig::quark_tiny()
        };
        let device = Default::default();
        let dir = std::env::temp_dir().join(format!("quark-mmap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        save_sharded(&QuarkModel::<B>::new(&cfg, &device), &cfg, &dir.join("f32"), 1).unwrap();
        crate::checkpoint::quantize::quantize_checkpoint(&dir.join("f32"), &dir.join("q4"), QuantFormat::Q4).unwrap();
        let stage = MappedStage::open(&dir.join("q4"), &layer_stage(0)).unwrap();
        for path in ["attn.q_proj", "moe.experts.1.down_proj"] {
            let q = stage.host_quant(QuantFormat::Q4, path).unwrap();
            assert!(matches!(q.packed, Words::Mapped { .. }), "{path}: {:?}", q.packed);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
