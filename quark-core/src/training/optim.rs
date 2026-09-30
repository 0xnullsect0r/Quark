//! Per-parameter optimizers for the streamed trainer.
//!
//! Burn's optimizers need the whole module (and keep all state in memory), so
//! the streamed trainer uses these instead. They work on one flattened
//! parameter at a time, on the compute device, with state kept as host tensors
//! (in a [`StageTensors`] from the tensor store) between steps.

use burn::tensor::{backend::Backend, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use crate::memory::store::StageTensors;
use crate::training::adamw::AdamWConfig;

/// Which optimizer the streamed trainer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OptimizerKind {
    /// AdamW with f32 moments: 8 bytes of state per parameter.
    #[default]
    AdamW,
    /// AdamW with an int8 block-quantized first moment and a bf16 second
    /// moment: ~3 bytes of state per parameter.
    AdamWCompact,
    /// Adafactor (no momentum, factored second moment for matrices): close to
    /// zero state.
    Adafactor,
}

impl OptimizerKind {
    /// Approximate optimizer-state bytes per parameter.
    pub fn state_bytes_per_param(self) -> f64 {
        match self {
            OptimizerKind::AdamW => 8.0,
            OptimizerKind::AdamWCompact => 3.0 + 4.0 / QUANT_BLOCK as f64,
            OptimizerKind::Adafactor => 0.05,
        }
    }
}

/// Block size for int8 quantization of the first moment.
pub const QUANT_BLOCK: usize = 64;

/// One parameter's update inputs.
pub struct ParamUpdate<'a, B: Backend> {
    /// Parameter name (state tensors are stored as `{name}.{suffix}`).
    pub name: &'a str,
    pub shape: &'a [usize],
    pub param: Tensor<B, 1>,
    pub grad: Tensor<B, 1>,
}

/// Apply one optimizer step to a parameter and return its new value.
///
/// `step` is the 1-based number of this optimizer step. `state` holds the
/// previous state (empty on the first step); new state tensors are appended to
/// `new_state`.
pub fn update<B: Backend>(
    kind: OptimizerKind,
    hyper: &AdamWConfig,
    lr: f64,
    step: u64,
    p: ParamUpdate<'_, B>,
    state: &StageTensors,
    new_state: &mut Vec<(String, TensorData)>,
) -> Tensor<B, 1> {
    match kind {
        OptimizerKind::AdamW | OptimizerKind::AdamWCompact => {
            adamw(kind, hyper, lr, step, p, state, new_state)
        }
        OptimizerKind::Adafactor => adafactor(hyper, lr, step, p, state, new_state),
    }
}

fn load<B: Backend>(state: &StageTensors, key: &str, device: &B::Device) -> Option<Tensor<B, 1>> {
    let data = state.get(key)?.clone().convert::<f32>();
    let n = data.num_elements();
    Some(Tensor::from_data(TensorData::new(data.to_vec::<f32>().ok()?, [n]), device))
}

fn host(t: Tensor<impl Backend, 1>) -> TensorData {
    t.into_data().convert::<f32>()
}

/// AdamW, following Burn's `AdamW` exactly (bias-corrected moments, decoupled
/// weight decay applied to the parameter before the update).
fn adamw<B: Backend>(
    kind: OptimizerKind,
    hyper: &AdamWConfig,
    lr: f64,
    step: u64,
    p: ParamUpdate<'_, B>,
    state: &StageTensors,
    new_state: &mut Vec<(String, TensorData)>,
) -> Tensor<B, 1> {
    let device = p.param.device();
    let (b1, b2) = (hyper.beta1 as f32, hyper.beta2 as f32);
    let g = p.grad;
    let n = g.dims()[0];

    let (m_prev, v_prev) = match kind {
        OptimizerKind::AdamWCompact => (
            dequantize_i8(state, &format!("{}.m", p.name), n, &device),
            load::<B>(state, &format!("{}.v", p.name), &device),
        ),
        _ => (
            load::<B>(state, &format!("{}.m", p.name), &device),
            load::<B>(state, &format!("{}.v", p.name), &device),
        ),
    };
    let m = match m_prev {
        Some(m) => m.mul_scalar(b1) + g.clone().mul_scalar(1.0 - b1),
        None => g.clone().mul_scalar(1.0 - b1),
    };
    let v = match v_prev {
        Some(v) => v.mul_scalar(b2) + g.clone().square().mul_scalar(1.0 - b2),
        None => g.square().mul_scalar(1.0 - b2),
    };

    let t = step as i32;
    let m_hat = m.clone().div_scalar(1.0 - b1.powi(t));
    let v_hat = v.clone().div_scalar(1.0 - b2.powi(t));
    let delta = m_hat / v_hat.sqrt().add_scalar(hyper.eps as f32);

    let decay = lr * hyper.weight_decay;
    let param = if decay == 0.0 { p.param } else { p.param.mul_scalar(1.0 - decay) };
    let updated = param - delta.mul_scalar(lr);

    match kind {
        OptimizerKind::AdamWCompact => {
            quantize_i8(&format!("{}.m", p.name), m, new_state);
            let v_bf16 = host(v).convert::<half::bf16>();
            new_state.push((format!("{}.v", p.name), v_bf16));
        }
        _ => {
            new_state.push((format!("{}.m", p.name), host(m)));
            new_state.push((format!("{}.v", p.name), host(v)));
        }
    }
    updated
}

/// Adafactor without momentum (Shazeer & Stern, 2018): factored second moment
/// for matrices, update clipping at RMS 1, decay `1 - t^-0.8`.
fn adafactor<B: Backend>(
    hyper: &AdamWConfig,
    lr: f64,
    step: u64,
    p: ParamUpdate<'_, B>,
    state: &StageTensors,
    new_state: &mut Vec<(String, TensorData)>,
) -> Tensor<B, 1> {
    const EPS1: f32 = 1e-30;
    let device = p.param.device();
    let beta = 1.0 - (step as f32).powf(-0.8);
    let g = p.grad;
    let n = g.dims()[0];
    let g2 = g.clone().square().add_scalar(EPS1);

    let u = if let [rows, cols] = *p.shape {
        let g2 = g2.reshape([rows, cols]);
        let row_mean = g2.clone().mean_dim(1).reshape([rows]);
        let col_mean = g2.mean_dim(0).reshape([cols]);
        let key_r = format!("{}.vr", p.name);
        let key_c = format!("{}.vc", p.name);
        let vr = match load::<B>(state, &key_r, &device) {
            Some(prev) => prev.mul_scalar(beta) + row_mean.mul_scalar(1.0 - beta),
            None => row_mean,
        };
        let vc = match load::<B>(state, &key_c, &device) {
            Some(prev) => prev.mul_scalar(beta) + col_mean.mul_scalar(1.0 - beta),
            None => col_mean,
        };
        let v_hat = vr.clone().reshape([rows, 1]).matmul(vc.clone().reshape([1, cols]))
            / vr.clone().mean().reshape([1, 1]);
        new_state.push((key_r, host(vr)));
        new_state.push((key_c, host(vc)));
        g.reshape([rows, cols]).div(v_hat.sqrt()).reshape([n])
    } else {
        let key = format!("{}.v", p.name);
        let v = match load::<B>(state, &key, &device) {
            Some(prev) => prev.mul_scalar(beta) + g2.mul_scalar(1.0 - beta),
            None => g2,
        };
        let u = g.div(v.clone().sqrt());
        new_state.push((key, host(v)));
        u
    };

    // Clip the update to RMS <= 1.
    let rms = u.clone().square().mean().sqrt().reshape([1]);
    let u = u / rms.clamp_min(1.0);

    let decay = lr * hyper.weight_decay;
    let param = if decay == 0.0 { p.param } else { p.param.mul_scalar(1.0 - decay) };
    param - u.mul_scalar(lr)
}

/// Store `x` as int8 in blocks of [`QUANT_BLOCK`] with one f32 absmax scale
/// per block (`{key}.q`, `{key}.s`). Values are square-root companded
/// (`q = 127·sign(x)·sqrt(|x|/absmax)`), so small moments next to a large one
/// keep ~1/16000 of the block maximum as resolution instead of rounding to 0.
fn quantize_i8<B: Backend>(key: &str, x: Tensor<B, 1>, out: &mut Vec<(String, TensorData)>) {
    let n = x.dims()[0];
    let blocks = n.div_ceil(QUANT_BLOCK);
    let padded = if blocks * QUANT_BLOCK == n {
        x
    } else {
        let pad = Tensor::zeros([blocks * QUANT_BLOCK - n], &x.device());
        Tensor::cat(vec![x, pad], 0)
    };
    let x = padded.reshape([blocks, QUANT_BLOCK]);
    let scale = x.clone().abs().max_dim(1).clamp_min(1e-30); // [blocks, 1]
    let unit = x / scale.clone(); // in [-1, 1]
    let q = (unit.clone().sign() * unit.abs().sqrt()).mul_scalar(127.0).round().int();
    let q = q.into_data().convert::<i8>();
    out.push((format!("{key}.q"), q));
    out.push((format!("{key}.s"), host(scale.reshape([blocks]))));
}

fn dequantize_i8<B: Backend>(
    state: &StageTensors,
    key: &str,
    n: usize,
    device: &B::Device,
) -> Option<Tensor<B, 1>> {
    let q = state.get(&format!("{key}.q"))?.clone().convert::<f32>();
    let scale = load::<B>(state, &format!("{key}.s"), device)?;
    let blocks = scale.dims()[0];
    let q = Tensor::<B, 1>::from_data(TensorData::new(q.to_vec::<f32>().ok()?, [blocks * QUANT_BLOCK]), device);
    let unit = q.reshape([blocks, QUANT_BLOCK]).div_scalar(127.0);
    let x = unit.clone().sign() * unit.square() * scale.reshape([blocks, 1]);
    Some(x.reshape([blocks * QUANT_BLOCK]).narrow(0, 0, n))
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Module,
        nn::{Linear, LinearConfig},
        optim::{GradientsParams, Optimizer},
        tensor::{Distribution, Tensor},
    };

    use super::*;
    use crate::backend::{ComputeBackend, TrainBackend};

    type AB = TrainBackend;
    type IB = ComputeBackend;

    fn weights(l: &Linear<AB>) -> Vec<f32> {
        l.weight.val().into_data().to_vec().unwrap()
    }

    fn loss(l: &Linear<AB>, x: &Tensor<AB, 2>) -> Tensor<AB, 1> {
        l.forward(x.clone()).square().mean()
    }

    /// Run `steps` of our optimizer on a Linear layer's weight (no bias) and
    /// return the final weights.
    fn ours(kind: OptimizerKind, hyper: &AdamWConfig, init: &Linear<AB>, x: &Tensor<AB, 2>, steps: u64) -> Vec<f32> {
        let mut layer = init.clone();
        let mut state = StageTensors::default();
        for step in 1..=steps {
            let grads = loss(&layer, x).backward();
            let g = layer.weight.grad(&grads).unwrap();
            let shape = g.dims();
            let n = shape[0] * shape[1];
            let mut new_state = Vec::new();
            let p = ParamUpdate {
                name: "weight",
                shape: &shape,
                param: layer.weight.val().inner().reshape([n]),
                grad: g.reshape([n]),
            };
            let updated = update::<IB>(kind, hyper, 0.01, step, p, &state, &mut new_state);
            state = StageTensors::new(new_state);
            let w = Tensor::<AB, 2>::from_inner(updated.reshape(shape));
            layer.weight = layer.weight.map(|_| w.require_grad());
        }
        weights(&layer)
    }

    fn setup() -> (AdamWConfig, Linear<AB>, Tensor<AB, 2>) {
        let device = Default::default();
        <AB as Backend>::seed(&device, 7);
        let hyper = AdamWConfig { weight_decay: 0.1, ..AdamWConfig::default() };
        let layer = LinearConfig::new(16, 8).with_bias(false).init::<AB>(&device);
        let _ = weights(&layer); // materialise lazy init before cloning
        let x = Tensor::<AB, 2>::random([4, 16], Distribution::Normal(0.0, 1.0), &device);
        (hyper, layer, x)
    }

    #[test]
    fn adamw_matches_burn() {
        let (hyper, layer, x) = setup();
        let mut optim = hyper.to_burn_config().init::<AB, Linear<AB>>();
        let mut reference = layer.clone();
        for _ in 0..3 {
            let grads = GradientsParams::from_grads(loss(&reference, &x).backward(), &reference);
            reference = optim.step(0.01, reference, grads);
        }
        let ours = ours(OptimizerKind::AdamW, &hyper, &layer, &x, 3);
        for (a, b) in ours.iter().zip(weights(&reference)) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    fn loss_with(layer: &Linear<AB>, w: &[f32], x: &Tensor<AB, 2>) -> f32 {
        let mut l = layer.clone();
        let t = Tensor::<AB, 1>::from_data(TensorData::new(w.to_vec(), [w.len()]), &Default::default());
        l.weight = l.weight.map(|_| t.reshape([16, 8]));
        loss(&l, x).into_scalar()
    }

    #[test]
    fn compact_and_adafactor_reduce_loss() {
        let (hyper, layer, x) = setup();
        let before: f32 = loss(&layer, &x).into_scalar();
        let adamw_loss = loss_with(&layer, &ours(OptimizerKind::AdamW, &hyper, &layer, &x, 20), &x);
        for kind in [OptimizerKind::AdamWCompact, OptimizerKind::Adafactor] {
            let after = loss_with(&layer, &ours(kind, &hyper, &layer, &x, 20), &x);
            assert!(after < before * 0.5, "{kind:?}: {before} → {after}");
        }
        // Compact states follow full AdamW: same loss, and near-identical
        // weights over a short horizon (long horizons amplify tiny differences).
        let compact_loss = loss_with(&layer, &ours(OptimizerKind::AdamWCompact, &hyper, &layer, &x, 20), &x);
        assert!((compact_loss - adamw_loss).abs() <= 0.1 * adamw_loss + 1e-4, "{compact_loss} vs {adamw_loss}");
        let a = ours(OptimizerKind::AdamW, &hyper, &layer, &x, 2);
        let c = ours(OptimizerKind::AdamWCompact, &hyper, &layer, &x, 2);
        let drift = a.iter().zip(&c).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(drift < 5e-3, "2-step drift {drift}");
    }

    #[test]
    fn int8_roundtrip() {
        let device = Default::default();
        let x = Tensor::<IB, 1>::random([300], Distribution::Normal(0.0, 1.0), &device);
        let mut out = Vec::new();
        quantize_i8("m", x.clone(), &mut out);
        let state = StageTensors::new(out);
        let back = dequantize_i8::<IB>(&state, "m", 300, &device).unwrap();
        let err: f32 = (back - x.clone()).abs().max().into_scalar();
        let max: f32 = x.abs().max().into_scalar();
        // worst case at the top of the range: 2/127 of the block max
        assert!(err <= 2.0 * max / 127.0 + 1e-6, "{err}");
    }
}
