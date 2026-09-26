//! Exporting a trained checkpoint for inference (`quark-chat`, `quark-code`,
//! the GUI chat), optionally quantized.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::{module::Module, record::Recorder};

use crate::backend::InferBackend;
use crate::checkpoint::{
    quantize::quantize_checkpoint,
    sharded::{is_sharded, read_meta, save_sharded, stage_path},
    CheckpointRecorder,
};
use crate::model::{config::QuarkConfig, proj::QuantFormat, stages::stage_names, QuarkModel};

/// Name of the model directory inside an exported app bundle's `model/`.
pub const BUNDLE_CHECKPOINT: &str = "checkpoint";

/// The `tokenizer.json` belonging to a checkpoint (inside a sharded
/// checkpoint, or next to a `.bin`).
pub fn tokenizer_for(checkpoint: &Path) -> Option<PathBuf> {
    let own = checkpoint.join("tokenizer.json");
    if own.exists() {
        return Some(own);
    }
    let sibling = checkpoint.parent()?.join("tokenizer.json");
    sibling.exists().then_some(sibling)
}

/// Write `src` (a `.bin` checkpoint or a sharded directory) to `dst` as a
/// sharded checkpoint ready for inference: no optimizer state, tokenizer
/// included, projection weights quantized if `quant` is set.
pub fn export_for_inference(src: &Path, dst: &Path, quant: Option<QuantFormat>) -> Result<()> {
    let _ = std::fs::remove_dir_all(dst);
    let tokenizer = tokenizer_for(src);

    // Get a plain sharded copy to work from.
    let (sharded, temp) = if is_sharded(src) {
        (src.to_path_buf(), None)
    } else {
        let cfg = QuarkConfig::for_checkpoint(src)
            .with_context(|| format!("no config.json next to {}", src.display()))?;
        let device = Default::default();
        let record = CheckpointRecorder::new()
            .load(src.with_extension(""), &device)
            .with_context(|| format!("loading {}", src.display()))?;
        let model = QuarkModel::<InferBackend>::new(&cfg, &device).load_record(record);
        let tmp = dst.with_extension("f32-tmp");
        let _ = std::fs::remove_dir_all(&tmp);
        save_sharded(&model, &cfg, &tmp, 0)?;
        (tmp.clone(), Some(tmp))
    };

    let meta = read_meta(&sharded)?;
    match (quant, meta.quantization) {
        (Some(format), None) => {
            quantize_checkpoint(&sharded, dst, format)?;
        }
        (Some(want), Some(have)) if want != have => {
            anyhow::bail!("{} is already {have:?}-quantized", src.display())
        }
        _ => {
            // Plain copy of the model files (no optimizer state).
            std::fs::create_dir_all(dst)?;
            let cfg: QuarkConfig =
                serde_json::from_str(&std::fs::read_to_string(sharded.join("config.json"))?)?;
            for stage in stage_names(&cfg) {
                std::fs::copy(stage_path(&sharded, &stage), stage_path(dst, &stage))?;
            }
            for file in ["config.json", "meta.json"] {
                std::fs::copy(sharded.join(file), dst.join(file))?;
            }
        }
    }
    if let Some(tok) = tokenizer {
        std::fs::copy(tok, dst.join("tokenizer.json"))?;
    }
    if let Some(tmp) = temp {
        let _ = std::fs::remove_dir_all(tmp);
    }
    Ok(())
}

/// The model inside an app bundle's `model/` directory: the sharded
/// `checkpoint/` (possibly quantized), or a legacy `checkpoint.bin`.
pub fn find_bundled_model(model_dir: &Path) -> Option<PathBuf> {
    let sharded = model_dir.join(BUNDLE_CHECKPOINT);
    if is_sharded(&sharded) {
        return Some(sharded);
    }
    let bin = model_dir.join("checkpoint.bin");
    bin.exists().then_some(bin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::sharded::load_sharded;

    #[test]
    fn exports_bin_and_sharded_with_and_without_quantization() {
        let cfg = QuarkConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 64,
            num_experts: 2,
            num_experts_per_tok: 1,
            ..QuarkConfig::quark_tiny()
        };
        let device = Default::default();
        let dir = std::env::temp_dir().join(format!("quark-export-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("train")).unwrap();
        let model = QuarkModel::<InferBackend>::new(&cfg, &device);
        // materialise the lazy parameters before cloning
        crate::memory::stage::module_to_stage(&model).unwrap();
        CheckpointRecorder::new().record(model.clone().into_record(), dir.join("train/checkpoint-5")).unwrap();
        std::fs::write(dir.join("train/config.json"), serde_json::to_string(&cfg).unwrap()).unwrap();
        std::fs::write(dir.join("train/tokenizer.json"), "{}").unwrap();

        let bin = dir.join("train/checkpoint-5.bin");
        for (src, quant) in [(bin.clone(), None), (bin, Some(QuantFormat::Q4))] {
            let dst = dir.join(format!("bundle-{quant:?}/model/{BUNDLE_CHECKPOINT}"));
            export_for_inference(&src, &dst, quant).unwrap();
            assert_eq!(find_bundled_model(dst.parent().unwrap()).unwrap(), dst);
            assert!(dst.join("tokenizer.json").exists());
            assert_eq!(read_meta(&dst).unwrap().quantization, quant);
            let (_, loaded) = load_sharded::<InferBackend>(&dst, &device).unwrap();
            let _ = loaded;
        }
        // A sharded source re-exported with quantization.
        let q8 = dir.join("q8");
        export_for_inference(&dir.join(format!("bundle-None/model/{BUNDLE_CHECKPOINT}")), &q8, Some(QuantFormat::Q8)).unwrap();
        assert_eq!(read_meta(&q8).unwrap().quantization, Some(QuantFormat::Q8));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
