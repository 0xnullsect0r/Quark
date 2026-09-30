//! Moving a Burn module's parameters in and out of [`StageTensors`].
//!
//! A stage's tensors are named by their path in the module (for example
//! `attn.q_proj.weight`), using `burn-store`'s snapshot machinery. Loading a
//! stage overwrites the parameters of an existing module of the same shape
//! (a "skeleton"), so one skeleton per layer type is reused for every layer.

use anyhow::Result;
use burn::{
    module::Module,
    tensor::{backend::Backend, DType, Element},
};
use burn::store::{ModuleSnapshot, TensorSnapshot};

use super::store::StageTensors;

/// Copy every parameter of `module` into host memory (float parameters as
/// f32, whatever precision the module computes in).
pub fn module_to_stage<B: Backend, M: Module<B>>(module: &M) -> Result<StageTensors> {
    let tensors = module
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| {
            let data = snapshot
                .to_data()
                .map_err(|e| anyhow::anyhow!("reading {}: {e:?}", snapshot.full_path()))?;
            let data = if data.dtype.is_float() { data.convert_dtype(DType::F32) } else { data };
            Ok((snapshot.full_path(), data))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StageTensors::new(tensors))
}

/// Overwrite `module`'s parameters with the tensors in `stage`, which must
/// cover every parameter with matching shapes. Float data is cast to the
/// backend's float type (e.g. bf16).
pub fn load_stage<B: Backend, M: Module<B>>(module: &mut M, stage: &StageTensors) -> Result<()> {
    load_stage_skipping(module, stage, &[])
}

/// Like [`load_stage`], but the parameters at `skip` may be absent (they stay
/// uninitialised, e.g. the dense weights of quantized projections).
pub fn load_stage_skipping<B: Backend, M: Module<B>>(
    module: &mut M,
    stage: &StageTensors,
    skip: &[String],
) -> Result<()> {
    let float = <B::FloatElem as Element>::dtype();
    let snapshots = stage
        .tensors
        .iter()
        .map(|(path, data)| {
            let path_stack = path.split('.').map(str::to_owned).collect();
            let data = if data.dtype.is_float() && data.dtype != float {
                data.clone().convert_dtype(float)
            } else {
                data.clone()
            };
            TensorSnapshot::from_data(data, path_stack, vec![], Default::default())
        })
        .collect();
    apply_checked(module, snapshots, skip)
}

fn apply_checked<B: Backend, M: Module<B>>(
    module: &mut M,
    snapshots: Vec<TensorSnapshot>,
    skip: &[String],
) -> Result<()> {
    let mut result = module.apply(snapshots, None, None, false);
    result.missing.retain(|(path, _)| !skip.contains(path));
    if !result.errors.is_empty() || !result.missing.is_empty() || !result.unused.is_empty() {
        anyhow::bail!(
            "stage does not match module: errors={:?} missing={:?} unused={:?}",
            result.errors,
            result.missing,
            result.unused
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, Tensor};

    use super::*;
    use crate::backend::InferBackend as B;
    use crate::model::{block::DecoderBlock, config::QuarkConfig};

    fn cfg() -> QuarkConfig {
        QuarkConfig {
            hidden_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 64,
            num_experts: 4,
            num_experts_per_tok: 2,
            ..QuarkConfig::quark_tiny()
        }
    }

    #[test]
    fn stage_roundtrip_reproduces_block() {
        let device = Default::default();
        for is_moe in [false, true] {
            let source = DecoderBlock::<B>::new(&cfg(), is_moe, &device);
            let stage = module_to_stage(&source).unwrap();
            assert!(stage.tensors.iter().any(|(n, _)| n == "attn.q_proj.weight"), "{:?}", stage.tensors.iter().map(|(n, _)| n).collect::<Vec<_>>());

            // A differently initialised skeleton becomes identical after loading.
            let mut skeleton = DecoderBlock::<B>::new(&cfg(), is_moe, &device);
            load_stage(&mut skeleton, &stage).unwrap();
            let x = Tensor::<B, 3>::random([1, 5, 32], Distribution::Normal(0.0, 1.0), &device);
            let a: Vec<f32> = source.forward(x.clone(), true).into_data().to_vec().unwrap();
            let b: Vec<f32> = skeleton.forward(x, true).into_data().to_vec().unwrap();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn mismatched_stage_is_rejected() {
        let device = Default::default();
        let dense = DecoderBlock::<B>::new(&cfg(), false, &device);
        let mut moe = DecoderBlock::<B>::new(&cfg(), true, &device);
        assert!(load_stage(&mut moe, &module_to_stage(&dense).unwrap()).is_err());
    }
}
