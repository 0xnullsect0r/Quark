//! Export a sharded checkpoint with quantized projection weights (Q4 or Q8)
//! for inference: a 10B-parameter model is ≈ 6 GB at Q4.
//!
//! Every attention / FFN / expert projection and the LM head is quantized
//! ([`HostQuant`]); the embedding, norms and MoE routers stay f32.

use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{DType, TensorData};

use crate::checkpoint::sharded::{read_meta, read_stage, write_stage, ShardedMeta, SHARDED_FORMAT};
use crate::memory::store::StageTensors;
use crate::model::{
    config::QuarkConfig,
    proj::{group_size, HostQuant, QuantFormat},
    stages::stage_names,
};

/// Whether a stage tensor is a projection weight that gets quantized.
pub fn is_quantizable(name: &str) -> bool {
    name.ends_with("_proj.weight") || name == "lm_head.weight"
}

/// Rebuild a [`HostQuant`] from its stored `packed` (u32 `[out, words]`) and
/// `scales` (`[out, groups]`) tensors.
pub fn host_quant_from(format: QuantFormat, packed: &TensorData, scales: &TensorData) -> Result<HostQuant> {
    let [out_features, words] = packed.shape.dims::<2>();
    let groups = scales.shape.dims::<2>()[1];
    let in_features = words * format.per_word();
    let packed = packed.clone().convert::<u32>().to_vec::<u32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let scales = scales.clone().convert::<f32>().to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(HostQuant {
        format,
        in_features,
        out_features,
        group: in_features / groups,
        packed: crate::model::proj::Words::Owned(packed),
        scales,
    })
}

/// Sizes of a quantization export.
#[derive(Debug, Clone, Copy)]
pub struct QuantReport {
    pub source_bytes: u64,
    pub output_bytes: u64,
    pub quantized_params: u64,
}

/// Quantize the sharded checkpoint `src` into `dst` (a new sharded checkpoint
/// directory). One stage is processed at a time, so this works for models far
/// larger than RAM.
pub fn quantize_checkpoint(src: &Path, dst: &Path, format: QuantFormat) -> Result<QuantReport> {
    let meta = read_meta(src)?;
    anyhow::ensure!(meta.quantization.is_none(), "{} is already quantized", src.display());
    let cfg: QuarkConfig = serde_json::from_str(&std::fs::read_to_string(src.join("config.json"))?)?;
    std::fs::create_dir_all(dst)?;

    let mut report = QuantReport { source_bytes: 0, output_bytes: 0, quantized_params: 0 };
    for stage in stage_names(&cfg) {
        let data = read_stage(src, &stage).with_context(|| format!("stage {stage}"))?;
        report.source_bytes += data.bytes();
        let mut out = Vec::with_capacity(data.tensors.len());
        for (name, tensor) in data.tensors {
            let shape = tensor.shape.to_vec();
            if !is_quantizable(&name) || shape.len() != 2 || !shape[0].is_multiple_of(format.per_word()) {
                out.push((name, tensor.convert_dtype(DType::F32)));
                continue;
            }
            let (in_f, out_f) = (shape[0], shape[1]);
            let weights = tensor.convert::<f32>().to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let q = HostQuant::quantize(format, &weights, in_f, out_f);
            report.quantized_params += (in_f * out_f) as u64;
            let prefix = name.trim_end_matches(".weight");
            let words = in_f / format.per_word();
            let groups = in_f / group_size(in_f);
            let packed = q.packed.as_slice().to_vec();
            out.push((format!("{prefix}.quant.packed"), TensorData::new(packed, [out_f, words])));
            let scales: Vec<half::f16> = q.scales.iter().map(|&s| half::f16::from_f32(s)).collect();
            out.push((format!("{prefix}.quant.scales"), TensorData::new(scales, [out_f, groups])));
        }
        let out = StageTensors::new(out);
        report.output_bytes += out.bytes();
        write_stage(dst, &stage, &out)?;
    }

    std::fs::write(dst.join("config.json"), serde_json::to_string_pretty(&cfg)?)?;
    let tok = src.join("tokenizer.json");
    if tok.exists() {
        std::fs::copy(&tok, dst.join("tokenizer.json"))?;
    }
    let meta = ShardedMeta { format: SHARDED_FORMAT.to_owned(), step: meta.step, quantization: Some(format) };
    std::fs::write(dst.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Int, Tensor};

    use super::*;
    use crate::backend::InferBackend as B;
    use crate::checkpoint::sharded::{load_sharded, save_sharded};
    use crate::model::QuarkModel;

    #[test]
    fn quantized_checkpoint_loads_and_tracks_the_original() {
        let cfg = QuarkConfig {
            vocab_size: 64,
            hidden_size: 64,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            max_position_embeddings: 32,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_layer_freq: 2,
            ..QuarkConfig::quark_tiny()
        };
        let device = Default::default();
        let dir = std::env::temp_dir().join(format!("quark-quant-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let model = QuarkModel::<B>::new(&cfg, &device);
        save_sharded(&model, &cfg, &dir.join("f32"), 3).unwrap();

        let ids = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![1, 7, 3, 9, 12, 5], [1, 6]), &device);
        let reference: Vec<f32> = model.forward(ids.clone()).into_data().to_vec().unwrap();
        let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));

        for (format, tol) in [(QuantFormat::Q8, 0.02), (QuantFormat::Q4, 0.25)] {
            let out = dir.join(format!("{format:?}"));
            let report = quantize_checkpoint(&dir.join("f32"), &out, format).unwrap();
            assert!(report.output_bytes * 2 < report.source_bytes, "{report:?}");
            assert_eq!(read_meta(&out).unwrap().quantization, Some(format));

            let (_, q) = load_sharded::<B>(&out, &device).unwrap();
            let got: Vec<f32> = q.forward(ids.clone()).into_data().to_vec().unwrap();
            let err = got.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(err < tol * scale, "{format:?}: max logit error {err} (scale {scale})");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
