//! Streamed (layer-by-layer) training for models larger than memory.
//!
//! Weights and optimizer state live in a [`TensorStore`] (RAM, spilling to
//! disk). Each optimizer step:
//!
//! 1. **Forward** (no autodiff): load each stage in turn, run every
//!    micro-batch through it, and save each stage's *input* activations
//!    (to a second store, which also spills to disk).
//! 2. **Backward**, last stage first: load the stage with autodiff, recompute
//!    it from its saved input for each micro-batch, backpropagate the gradient
//!    coming from the stage above, and accumulate the parameter gradients.
//! 3. **Update** that stage right away (clip → optimizer → write back to the
//!    store), then drop its gradients.
//!
//! So only one stage's weights, gradients and optimizer state are ever on the
//! compute device, and the full-model gradient never exists. This is
//! per-layer activation checkpointing plus a fused backward/update.
//!
//! Gradient clipping is per stage: each stage's gradient is clipped to
//! `max_grad_norm / sqrt(num_stages)`, which bounds the global norm by
//! `max_grad_norm` (the in-memory trainer clips the global norm exactly).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use burn::{
    module::Module,
    optim::GradientsParams,
    store::ModuleSnapshot,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Device, ElementConversion, Int, Tensor, TensorData,
    },
};

use crate::backend::ComputeBackend;
use crate::checkpoint::sharded::{self, write_meta};
use crate::data::batch::DataBatch;
use crate::memory::stage::{load_stage, module_to_stage};
use crate::memory::store::{StageTensors, TensorStore};
use crate::model::{
    block::DecoderBlock,
    config::QuarkConfig,
    stages::{layer_stage, stage_names, EmbedStage, HeadStage, EMBED_STAGE, HEAD_STAGE},
};
use crate::tokenizer::bpe::PAD_ID;
use crate::training::adamw::AdamWConfig;
use crate::training::loss::masked_cross_entropy;
use crate::training::optim::{self, OptimizerKind, ParamUpdate};
use crate::training::trainer::{batch_tensors, AUX_LOSS_COEF};

type Inner<AB> = <AB as AutodiffBackend>::InnerBackend;

fn optim_stage(stage: &str) -> String {
    format!("optim.{stage}")
}

fn act_key(layer: usize, micro: usize) -> String {
    format!("act.{layer:04}.{micro:04}")
}

/// Default logit elements per head chunk (~256 MB in f32).
const DEFAULT_HEAD_CHUNK_ELEMS: usize = 64 << 20;

/// `(start, len)` ranges covering `0..total` in steps of `size`.
fn chunks(total: usize, size: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..total).step_by(size.max(1)).map(move |start| (start, size.min(total - start)))
}

/// Non-padding labels at positions `start..start + len` of `batch`.
fn target_count(batch: &DataBatch, start: usize, len: usize) -> u64 {
    batch
        .labels
        .iter()
        .map(|row| row.iter().skip(start).take(len).filter(|&&t| t != PAD_ID).count() as u64)
        .sum()
}

/// Result of one optimizer step.
#[derive(Debug, Clone, Copy)]
pub struct StepStats {
    /// Mean cross-entropy over the micro-batches.
    pub loss: f32,
    /// Global L2 gradient norm before clipping.
    pub grad_norm: f32,
    /// Non-padding target tokens processed.
    pub tokens: u64,
}

/// Optimizer settings for [`StreamedTrainer`].
#[derive(Debug, Clone)]
pub struct StreamedOptim {
    pub kind: OptimizerKind,
    pub hyper: AdamWConfig,
    /// Global gradient-norm bound (see the module docs); 0 disables clipping.
    pub max_grad_norm: f32,
}

/// Skeleton modules whose parameters are overwritten with each stage's data.
struct Skeletons<B: Backend> {
    embed: EmbedStage<B>,
    dense: Option<DecoderBlock<B>>,
    moe: Option<DecoderBlock<B>>,
    head: HeadStage<B>,
}

impl<B: Backend> Skeletons<B> {
    fn new(cfg: &QuarkConfig, device: &B::Device) -> Self {
        let any = |moe: bool| (0..cfg.num_hidden_layers).any(|i| cfg.is_moe_layer(i) == moe);
        Self {
            embed: EmbedStage::new(cfg, device),
            dense: any(false).then(|| DecoderBlock::new(cfg, false, device)),
            moe: any(true).then(|| DecoderBlock::new(cfg, true, device)),
            head: HeadStage::new(cfg, device),
        }
    }

    fn layer(&mut self, cfg: &QuarkConfig, i: usize) -> &mut DecoderBlock<B> {
        let slot = if cfg.is_moe_layer(i) { &mut self.moe } else { &mut self.dense };
        slot.as_mut().expect("skeleton exists for every layer kind in the config")
    }
}

/// See the module docs.
pub struct StreamedTrainer<AB: AutodiffBackend> {
    cfg: QuarkConfig,
    store: Arc<TensorStore>,
    acts: Arc<TensorStore>,
    device: Device<AB>,
    optim: StreamedOptim,
    /// Optimizer steps taken so far.
    pub step: u64,
    /// Logit elements (positions × batch × vocab) per head chunk.
    head_chunk_elems: usize,
    infer: Skeletons<Inner<AB>>,
    train: Skeletons<AB>,
}

impl<AB: AutodiffBackend> StreamedTrainer<AB> {
    /// A trainer over `store` (weights + optimizer state) and `acts`
    /// (activations). Stages missing from `store` are randomly initialised,
    /// one at a time.
    pub fn new(
        cfg: QuarkConfig,
        store: Arc<TensorStore>,
        acts: Arc<TensorStore>,
        device: Device<AB>,
        optim: StreamedOptim,
        step: u64,
    ) -> Result<Self> {
        let infer = Skeletons::new(&cfg, &device);
        let train = Skeletons::new(&cfg, &device);
        let head_chunk_elems = DEFAULT_HEAD_CHUNK_ELEMS;
        let trainer = Self { cfg, store, acts, device, optim, step, head_chunk_elems, infer, train };
        trainer.init_missing_stages()?;
        Ok(trainer)
    }

    /// Limit the head's logits to about `elems` values per chunk (mainly for
    /// tests; the default keeps a chunk around 256 MB in f32).
    pub fn set_head_chunk_elems(&mut self, elems: usize) {
        self.head_chunk_elems = elems.max(1);
    }

    /// Sequence positions per head chunk for `batch`.
    fn head_chunk(&self, batch: &DataBatch) -> usize {
        let rows = batch.input_ids.len().max(1);
        (self.head_chunk_elems / (rows * self.cfg.vocab_size).max(1)).max(1)
    }

    pub fn config(&self) -> &QuarkConfig {
        &self.cfg
    }

    pub fn store(&self) -> &Arc<TensorStore> {
        &self.store
    }

    fn init_missing_stages(&self) -> Result<()> {
        let cfg = &self.cfg;
        let device: Device<Inner<AB>> = self.device.clone();
        if !self.store.contains(EMBED_STAGE) {
            self.store.put(EMBED_STAGE, module_to_stage(&EmbedStage::<Inner<AB>>::new(cfg, &device))?)?;
        }
        for i in 0..cfg.num_hidden_layers {
            let name = layer_stage(i);
            if !self.store.contains(&name) {
                let layer = DecoderBlock::<Inner<AB>>::new(cfg, cfg.is_moe_layer(i), &device);
                self.store.put(&name, module_to_stage(&layer)?)?;
            }
        }
        if !self.store.contains(HEAD_STAGE) {
            self.store.put(HEAD_STAGE, module_to_stage(&HeadStage::<Inner<AB>>::new(cfg, &device))?)?;
        }
        Ok(())
    }

    fn put_act(&self, key: &str, x: Tensor<Inner<AB>, 3>) -> Result<()> {
        self.acts.put(key, StageTensors::new(vec![("x".into(), x.into_data())]))
    }

    fn get_act(&self, key: &str) -> Result<Tensor<Inner<AB>, 3>> {
        let stage = self.acts.get(key)?;
        let data = stage.get("x").context("activation missing")?.clone();
        Ok(Tensor::from_data(data, &self.device))
    }

    /// Forward-only pass: returns the mean cross-entropy over `batches`.
    pub fn eval_loss(&mut self, batches: &[DataBatch]) -> Result<f32> {
        if batches.is_empty() {
            return Ok(f32::NAN);
        }
        self.forward_to_acts(batches)?;
        load_stage(&mut self.infer.head, &*self.store.get(HEAD_STAGE)?)?;
        let mut total = 0.0;
        for (mb, batch) in batches.iter().enumerate() {
            let (_, labels, n_tok) = batch_tensors::<Inner<AB>>(batch, &self.device);
            let x_all = self.get_act(&act_key(self.cfg.num_hidden_layers, mb))?;
            for (start, len) in chunks(x_all.dims()[1], self.head_chunk(batch)) {
                let n_chunk = target_count(batch, start, len);
                if n_chunk == 0 {
                    continue;
                }
                let logits = self.infer.head.forward(x_all.clone().narrow(1, start, len));
                let [b, s, v] = logits.dims();
                let chunk_labels = labels.clone().narrow(1, start, len).reshape([b * s]);
                let loss = masked_cross_entropy(logits.reshape([b * s, v]), chunk_labels, PAD_ID).into_scalar().elem::<f32>();
                total += loss * n_chunk as f32 / n_tok.max(1) as f32;
            }
        }
        self.clear_acts(batches.len())?;
        Ok(total / batches.len() as f32)
    }

    /// Run every micro-batch through every layer (not the head), saving each
    /// layer's input activations.
    fn forward_to_acts(&mut self, micro: &[DataBatch]) -> Result<()> {
        let cfg = self.cfg.clone();
        load_stage(&mut self.infer.embed, &*self.store.get(EMBED_STAGE)?)?;
        self.store.prefetch(&layer_stage(0));
        for (mb, batch) in micro.iter().enumerate() {
            let (ids, _, _) = batch_tensors::<Inner<AB>>(batch, &self.device);
            let x = self.infer.embed.forward(ids);
            self.put_act(&act_key(0, mb), x)?;
        }
        for i in 0..cfg.num_hidden_layers {
            let next = if i + 1 < cfg.num_hidden_layers { layer_stage(i + 1) } else { HEAD_STAGE.to_owned() };
            let data = self.store.get(&layer_stage(i))?;
            self.store.prefetch(&next);
            let layer = self.infer.layer(&cfg, i);
            load_stage(layer, &data)?;
            for (mb, _) in micro.iter().enumerate() {
                let x = self.get_act(&act_key(i, mb))?;
                let y = self.infer.layer(&cfg, i).forward(x, true);
                self.put_act(&act_key(i + 1, mb), y)?;
            }
        }
        Ok(())
    }

    fn clear_acts(&self, micro: usize) -> Result<()> {
        for i in 0..=self.cfg.num_hidden_layers {
            for mb in 0..micro {
                self.acts.remove(&act_key(i, mb))?;
            }
        }
        Ok(())
    }

    /// One optimizer step over `micro` (gradient accumulation across the
    /// micro-batches) at learning rate `lr`.
    pub fn train_step(&mut self, micro: &[DataBatch], lr: f64) -> Result<StepStats> {
        anyhow::ensure!(!micro.is_empty(), "no micro-batches");
        let cfg = self.cfg.clone();
        let n_micro = micro.len();
        let n_layers = cfg.num_hidden_layers;
        let n_moe = (0..n_layers).filter(|&i| cfg.is_moe_layer(i)).count().max(1);
        let n_stages = n_layers + 2;
        let stage_clip = if self.optim.max_grad_norm > 0.0 {
            self.optim.max_grad_norm / (n_stages as f32).sqrt()
        } else {
            0.0
        };
        self.step += 1;

        // ── 1. forward ───────────────────────────────────────────────────────
        self.forward_to_acts(micro)?;

        let mut sq_norm = 0.0f32;
        let mut loss_sum = 0.0f32;
        let mut tokens = 0u64;

        // ── 2+3. head: loss, backward, update ────────────────────────────────
        load_stage(&mut self.train.head, &*self.store.get(HEAD_STAGE)?)?;
        self.store.prefetch(&optim_stage(HEAD_STAGE));
        if n_layers > 0 {
            self.store.prefetch(&layer_stage(n_layers - 1));
        }
        let mut acc = GradAccumulator::default();
        let mut upstream: Vec<Tensor<Inner<AB>, 3>> = Vec::with_capacity(n_micro);
        for (mb, batch) in micro.iter().enumerate() {
            let (_, labels, n_tok) = batch_tensors::<AB>(batch, &self.device);
            tokens += n_tok;
            let x_all = self.get_act(&act_key(n_layers, mb))?;
            let [_, seq, _] = x_all.dims();
            // The head runs on chunks of positions so the full [tokens × vocab]
            // logits never exist at once. Each chunk's mean loss is weighted
            // by its share of the micro-batch's real tokens.
            let mut grads_in = Vec::new();
            for (start, len) in chunks(seq, self.head_chunk(batch)) {
                let n_chunk = target_count(batch, start, len);
                if n_chunk == 0 {
                    grads_in.push(Tensor::zeros([x_all.dims()[0], len, self.cfg.hidden_size], &self.device));
                    continue;
                }
                let x = Tensor::<AB, 3>::from_inner(x_all.clone().narrow(1, start, len)).require_grad();
                let logits = self.train.head.forward(x.clone());
                let [b, s, v] = logits.dims();
                let chunk_labels = labels.clone().narrow(1, start, len).reshape([b * s]);
                let weight = n_chunk as f32 / n_tok.max(1) as f32;
                let loss = masked_cross_entropy(logits.reshape([b * s, v]), chunk_labels, PAD_ID).mul_scalar(weight);
                loss_sum += loss.clone().into_scalar().elem::<f32>();
                let grads = loss.div_scalar(n_micro as f32).backward();
                grads_in.push(x.grad(&grads).context("no gradient for head input")?);
                acc.add(&self.train.head, grads);
            }
            upstream.push(Tensor::cat(grads_in, 1));
        }
        sq_norm += self.apply_update(HEAD_STAGE, acc, stage_clip, lr)?;

        // ── layers, last to first ────────────────────────────────────────────
        for i in (0..n_layers).rev() {
            let name = layer_stage(i);
            let data = self.store.get(&name)?;
            self.store.prefetch(&optim_stage(&name));
            let prev = if i > 0 { layer_stage(i - 1) } else { EMBED_STAGE.to_owned() };
            self.store.prefetch(&prev);
            load_stage(self.train.layer(&cfg, i), &data)?;

            let mut acc = GradAccumulator::default();
            let mut next_upstream = Vec::with_capacity(n_micro);
            for (mb, g_out) in upstream.into_iter().enumerate() {
                let x = Tensor::<AB, 3>::from_inner(self.get_act(&act_key(i, mb))?).require_grad();
                let layer = self.train.layer(&cfg, i);
                let (y, aux) = layer.forward_with_aux(x.clone(), true);
                let mut objective = (y * Tensor::from_inner(g_out)).sum();
                if let Some(aux) = aux {
                    objective = objective + aux.mul_scalar(AUX_LOSS_COEF / (n_moe * n_micro) as f32);
                }
                let grads = objective.backward();
                next_upstream.push(x.grad(&grads).context("no gradient for layer input")?);
                acc.add(self.train.layer(&cfg, i), grads);
                self.acts.remove(&act_key(i + 1, mb))?;
            }
            upstream = next_upstream;
            sq_norm += self.apply_update(&name, acc, stage_clip, lr)?;
        }

        // ── embedding ────────────────────────────────────────────────────────
        load_stage(&mut self.train.embed, &*self.store.get(EMBED_STAGE)?)?;
        let mut acc = GradAccumulator::default();
        for (mb, (batch, g_out)) in micro.iter().zip(upstream).enumerate() {
            let (ids, _, _) = batch_tensors::<AB>(batch, &self.device);
            let e = self.train.embed.forward(ids);
            let grads = (e * Tensor::from_inner(g_out)).sum().backward();
            acc.add(&self.train.embed, grads);
            self.acts.remove(&act_key(0, mb))?;
        }
        sq_norm += self.apply_update(EMBED_STAGE, acc, stage_clip, lr)?;

        Ok(StepStats { loss: loss_sum / n_micro as f32, grad_norm: sq_norm.sqrt(), tokens })
    }

    /// Clip, run the optimizer on one stage, and write weights + state back.
    /// Returns the stage's squared gradient norm before clipping.
    fn apply_update(&self, stage: &str, acc: GradAccumulator<Inner<AB>>, clip: f32, lr: f64) -> Result<f32> {
        let weights = self.store.get(stage)?;
        let opt_key = optim_stage(stage);
        let state = if self.store.contains(&opt_key) { self.store.get(&opt_key)? } else { Arc::default() };

        let sq_norm: f32 = acc
            .grads
            .values()
            .map(|g| g.clone().square().sum().into_scalar().elem::<f32>())
            .sum();
        let norm = sq_norm.sqrt();
        let scale = if clip > 0.0 && norm > clip { clip / (norm + 1e-6) } else { 1.0 };

        let mut new_weights = Vec::with_capacity(weights.tensors.len());
        let mut new_state = Vec::new();
        for (name, data) in &weights.tensors {
            let Some(grad) = acc.grads.get(name) else {
                // No gradient (e.g. an expert no token was routed to): keep the
                // weights and carry the optimizer state over unchanged.
                new_weights.push((name.clone(), data.clone()));
                let prefix = format!("{name}.");
                new_state.extend(state.tensors.iter().filter(|(k, _)| k.starts_with(&prefix)).cloned());
                continue;
            };
            let shape = data.shape.to_vec();
            let n: usize = shape.iter().product();
            // The optimizer always runs in f32 (on the f32 compute backend), even
            // when the model computes in bf16.
            let f32_device: Device<ComputeBackend> = Default::default();
            let to_f32 = |d: TensorData| -> Result<Tensor<ComputeBackend, 1>> {
                let v = d.convert::<f32>().to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
                Ok(Tensor::from_data(TensorData::new(v, [n]), &f32_device))
            };
            let param = to_f32(data.clone())?;
            let grad = to_f32(grad.clone().into_data())?;
            let grad = if scale < 1.0 { grad.mul_scalar(scale) } else { grad };
            let update = ParamUpdate { name, shape: &shape, param, grad };
            let updated = optim::update(self.optim.kind, &self.optim.hyper, lr, self.step, update, &state, &mut new_state);
            let host = updated.into_data().convert::<f32>();
            new_weights.push((name.clone(), TensorData::new(host.to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?, shape)));
        }
        self.store.put(stage, StageTensors::new(new_weights))?;
        self.store.put(&opt_key, StageTensors::new(new_state))?;
        Ok(sq_norm)
    }

    /// Save a sharded checkpoint (weights, optimizer state, meta) to `dir`.
    pub fn save_checkpoint(&self, dir: &Path) -> Result<()> {
        self.store.flush()?;
        std::fs::create_dir_all(dir.join("optim"))?;
        for stage in stage_names(&self.cfg) {
            link_or_copy(&self.store.dir().join(format!("{stage}.safetensors")), &sharded::stage_path(dir, &stage))?;
            let opt = self.store.dir().join(format!("{}.safetensors", optim_stage(&stage)));
            if opt.exists() {
                link_or_copy(&opt, &dir.join("optim").join(format!("{stage}.safetensors")))?;
            }
        }
        write_meta(dir, &self.cfg, self.step)
    }

    /// Put a sharded checkpoint's weights (and optimizer state, if present)
    /// into `store`, returning its step.
    pub fn restore_into(store: &TensorStore, dir: &Path, cfg: &QuarkConfig, with_optim: bool) -> Result<u64> {
        let meta = sharded::read_meta(dir)?;
        store.flush()?;
        for stage in stage_names(cfg) {
            link_or_copy(&sharded::stage_path(dir, &stage), &store.dir().join(format!("{stage}.safetensors")))?;
            let opt = dir.join("optim").join(format!("{stage}.safetensors"));
            if with_optim && opt.exists() {
                link_or_copy(&opt, &store.dir().join(format!("{}.safetensors", optim_stage(&stage))))?;
            }
        }
        Ok(meta.step)
    }
}

/// Hard-link (instant, and safe because the store replaces files by rename)
/// or copy `from` to `to`.
fn link_or_copy(from: &Path, to: &Path) -> Result<()> {
    let _ = std::fs::remove_file(to);
    if std::fs::hard_link(from, to).is_err() {
        std::fs::copy(from, to).with_context(|| format!("copying {} → {}", from.display(), to.display()))?;
    }
    Ok(())
}

/// Per-parameter gradient sums (flattened), keyed by parameter path.
struct GradAccumulator<B: Backend> {
    grads: HashMap<String, Tensor<B, 1>>,
}

impl<B: Backend> Default for GradAccumulator<B> {
    fn default() -> Self {
        Self { grads: HashMap::new() }
    }
}

impl<IB: Backend> GradAccumulator<IB> {
    fn add<AB, M>(&mut self, module: &M, grads: AB::Gradients)
    where
        AB: AutodiffBackend<InnerBackend = IB>,
        M: burn::module::AutodiffModule<AB>,
    {
        let params = module.collect(None, None, false);
        let mut grads = GradientsParams::from_grads(grads, module);
        for snapshot in params {
            let Some(id) = snapshot.tensor_id else { continue };
            let n: usize = snapshot.shape.iter().product();
            let grad: Option<Tensor<IB, 1>> = match snapshot.shape.len() {
                1 => grads.remove::<IB, 1>(id),
                2 => grads.remove::<IB, 2>(id).map(|g| g.reshape([n])),
                3 => grads.remove::<IB, 3>(id).map(|g| g.reshape([n])),
                _ => None,
            };
            if let Some(grad) = grad {
                let path = snapshot.full_path();
                let sum = match self.grads.remove(&path) {
                    Some(prev) => prev + grad,
                    None => grad,
                };
                self.grads.insert(path, sum);
            }
        }
    }
}

// ── Training run (called from `trainer::run_training_loop`) ─────────────────

/// Checkpoints kept by a streamed run (they can be tens of GB each).
const KEEP_CHECKPOINTS: usize = 2;

/// Offload directory: `TrainerConfig::tier.disk_offload_path`, relative to the
/// output directory unless absolute.
pub fn offload_dir(config: &crate::training::trainer::TrainerConfig) -> std::path::PathBuf {
    let p = &config.tier.disk_offload_path;
    if p.is_absolute() { p.clone() } else { config.output_dir.join(p) }
}

/// The streamed counterpart of the in-memory loop: same events, eval,
/// checkpointing, resume and fine-tuning, with sharded checkpoints.
pub(crate) fn run_streamed<AB: AutodiffBackend>(
    model_config: QuarkConfig,
    config: &crate::training::trainer::TrainerConfig,
    batches: Vec<DataBatch>,
    eval_batches: Vec<DataBatch>,
    tx: &crate::training::metrics::MetricsSender,
    stop: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    use crate::memory::budget::HardwareBudget;
    use crate::training::metrics::{TrainingEvent, TrainingMetrics};
    use crate::training::trainer::{fmt_gb, process_rss, TrainingMode, MAX_EVAL_BATCHES};

    let log = |m: String| {
        let _ = tx.send(TrainingEvent::Log(m));
    };
    let phase = |m: &str| {
        let _ = tx.send(TrainingEvent::Phase(m.to_owned()));
    };

    // ── stores ───────────────────────────────────────────────────────────────
    let budget = HardwareBudget::detect();
    let ram = config.tier.ram_limit_bytes(&budget).max(1 << 30);
    let root = offload_dir(config);
    let acts_dir = root.join("activations");
    let _ = std::fs::remove_dir_all(&acts_dir); // stale activations from a crashed run
    let store = TensorStore::new(root.join("weights"), ram / 2)?;
    let acts = TensorStore::new(&acts_dir, ram / 4)?;
    let per_layer = model_config.param_count() / (model_config.num_hidden_layers as u64 + 2).max(1);
    let state_bytes = (config.optimizer.state_bytes_per_param() * model_config.param_count() as f64) as u64;
    log(format!(
        "   Offload: {} (weights {} f32 + optimizer state ≈{}); RAM cache {}; ≈{} per layer on the device",
        root.display(),
        fmt_gb(model_config.param_count() * 4),
        fmt_gb(state_bytes),
        fmt_gb(ram / 2),
        fmt_gb(per_layer * 16),
    ));

    // ── starting point: resume, fine-tune base, or fresh ────────────────────
    let mut step = 0;
    let resumable = config
        .resume
        .then(|| sharded::latest_sharded(&config.output_dir))
        .flatten()
        .filter(|(dir, _)| QuarkConfig::for_checkpoint(dir).is_some_and(|c| {
            serde_json::to_value(&c).ok() == serde_json::to_value(&model_config).ok()
        }));
    if let Some((dir, _)) = resumable {
        step = StreamedTrainer::<AB>::restore_into(&store, &dir, &model_config, true)?;
        log(format!("↻  Resumed from {} (step {step}, optimizer state restored)", dir.display()));
    } else if let TrainingMode::FineTune { base_checkpoint } = &config.mode {
        if sharded::is_sharded(base_checkpoint) {
            StreamedTrainer::<AB>::restore_into(&store, base_checkpoint, &model_config, false)?;
        } else {
            use burn::record::Recorder;
            let device: Device<Inner<AB>> = Default::default();
            let record = crate::checkpoint::CheckpointRecorder::new()
                .load(base_checkpoint.with_extension(""), &device)
                .with_context(|| format!("loading {}", base_checkpoint.display()))?;
            let model = crate::model::QuarkModel::<Inner<AB>>::new(&model_config, &device).load_record(record);
            for (stage, data) in sharded::split_into_stages(module_to_stage(&model)?, &model_config)? {
                store.put(&stage, data)?;
            }
        }
        log(format!("   Loaded base weights from {}", base_checkpoint.display()));
    }
    if step >= config.max_steps {
        log(format!("✅  Already trained for {step} steps (max_steps={})", config.max_steps));
        phase("Complete!");
        return Ok(());
    }

    phase("Initialising model…");
    let device: Device<AB> = Default::default();
    <AB as Backend>::seed(&device, config.seed);
    let optim = StreamedOptim {
        kind: config.optimizer,
        hyper: config.adamw.clone(),
        max_grad_norm: config.max_grad_norm,
    };
    let mut trainer = StreamedTrainer::<AB>::new(model_config.clone(), store, acts, device, optim, step)?;
    log(format!("   Optimizer: {:?}", config.optimizer));

    // Synthetic data when no corpus was given (matches the in-memory demo loop).
    let batches = if batches.is_empty() {
        let seq = model_config.max_position_embeddings.min(64);
        (0..8)
            .map(|k| {
                let seqs = (0..config.batch_size.max(1))
                    .map(|r| (0..seq).map(|t| ((k * 131 + r * 17 + t) % model_config.vocab_size) as u32).collect())
                    .collect();
                crate::data::batch::collate_batch(seqs, PAD_ID)
            })
            .collect()
    } else {
        batches
    };

    // ── loop ─────────────────────────────────────────────────────────────────
    phase("Training (offloaded)…");
    let accum = config.grad_accum_steps.max(1);
    let start_step = step;
    let start = Instant::now();
    let mut cursor = (step as usize * accum) % batches.len();
    let mut epoch = (step as usize * accum / batches.len()) as u32;
    let mut tokens_seen = 0u64;
    let mut sys = sysinfo::System::new();
    let io = |s: &TensorStore| {
        s.stats().read_bytes.load(std::sync::atomic::Ordering::Relaxed)
            + s.stats().write_bytes.load(std::sync::atomic::Ordering::Relaxed)
    };
    // Existing sharded checkpoints count towards KEEP_CHECKPOINTS.
    let mut saved: Vec<std::path::PathBuf> = {
        let mut existing: Vec<(u64, std::path::PathBuf)> = std::fs::read_dir(&config.output_dir)
            .map(|rd| {
                rd.filter_map(|e| {
                    let path = e.ok()?.path();
                    let step = path.file_name()?.to_str()?.strip_prefix("checkpoint-")?.parse().ok()?;
                    sharded::is_sharded(&path).then_some((step, path))
                })
                .collect()
            })
            .unwrap_or_default();
        existing.sort();
        existing.into_iter().map(|(_, p)| p).collect()
    };

    let save = |trainer: &StreamedTrainer<AB>, saved: &mut Vec<std::path::PathBuf>| -> Result<()> {
        let dir = config.output_dir.join(format!("checkpoint-{}", trainer.step));
        trainer.save_checkpoint(&dir)?;
        let tok = config.output_dir.join("tokenizer.json");
        if tok.exists() {
            std::fs::copy(&tok, dir.join("tokenizer.json"))?;
        }
        log(format!("💾  Checkpoint saved → {}", dir.display()));
        saved.push(dir);
        while saved.len() > KEEP_CHECKPOINTS {
            let old = saved.remove(0);
            let _ = std::fs::remove_dir_all(&old);
            log(format!("   Removed old checkpoint {} (keeping the last {KEEP_CHECKPOINTS})", old.display()));
        }
        Ok(())
    };

    while trainer.step < config.max_steps && !stop.load(Ordering::SeqCst) {
        let micro: Vec<DataBatch> = (0..accum)
            .map(|_| {
                if cursor >= batches.len() {
                    cursor = 0;
                    epoch += 1;
                    log(format!("━━  Epoch {} started", epoch + 1));
                }
                cursor += 1;
                batches[cursor - 1].clone()
            })
            .collect();
        let lr = config.schedule.get_lr(trainer.step);
        let io_before = io(trainer.store()) + io(&trainer.acts);
        let t0 = Instant::now();
        let stats = trainer.train_step(&micro, lr)?;
        let step_secs = t0.elapsed().as_secs_f32();
        tokens_seen += stats.tokens;
        let io_bytes = io(trainer.store()) + io(&trainer.acts) - io_before;

        let elapsed = start.elapsed().as_secs_f32();
        let done = trainer.step - start_step;
        let _ = tx.send(TrainingEvent::Metrics(TrainingMetrics {
            step: trainer.step,
            loss: stats.loss,
            learning_rate: lr as f32,
            tokens_per_sec: tokens_seen as f32 / elapsed.max(1e-3),
            grad_norm: stats.grad_norm,
            vram_used_bytes: 0,
            ram_used_bytes: process_rss(&mut sys).unwrap_or(0),
            disk_used_bytes: 0,
            disk_io_bytes: io_bytes,
            epoch,
            eta_secs: (elapsed / done as f32 * (config.max_steps - trainer.step) as f32) as u64,
        }));
        if done == 1 || trainer.step.is_multiple_of(10) {
            log(format!(
                "   step={:>6}  loss={:.4}  lr={lr:.2e}  |g|={:.3}  {step_secs:.1}s/step  disk {}/step",
                trainer.step, stats.loss, stats.grad_norm, fmt_gb(io_bytes)
            ));
        }

        if !eval_batches.is_empty()
            && config.eval_every_steps > 0
            && trainer.step.is_multiple_of(config.eval_every_steps)
        {
            let n = eval_batches.len().min(MAX_EVAL_BATCHES);
            let loss = trainer.eval_loss(&eval_batches[..n])?;
            log(format!("   eval  step={:>6}  loss={loss:.4}  ppl={:.1}", trainer.step, loss.exp()));
            let _ = tx.send(TrainingEvent::Eval { step: trainer.step, loss });
        }
        if trainer.step < config.max_steps
            && config.save_every_steps > 0
            && trainer.step.is_multiple_of(config.save_every_steps)
        {
            save(&trainer, &mut saved)?;
        }
    }

    if stop.load(Ordering::SeqCst) {
        log(format!("⏹  Training stopped at step {}", trainer.step));
        phase("Stopped");
    } else {
        log(format!("✅  Training complete — {} steps in {:.1}s", trainer.step, start.elapsed().as_secs_f32()));
        phase("Complete!");
    }
    save(&trainer, &mut saved)?;
    let _ = std::fs::remove_dir_all(&acts_dir);
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::optim::{GradientsAccumulator, Optimizer};

    use super::*;
    use crate::backend::TrainBackend as AB;
    use crate::checkpoint::sharded::{load_sharded, save_sharded};
    use crate::data::batch::collate_batch;
    use crate::model::QuarkModel;

    fn cfg() -> QuarkConfig {
        QuarkConfig {
            vocab_size: 50,
            hidden_size: 32,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 48,
            max_position_embeddings: 16,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_layer_freq: 2, // layers 0 and 2 are MoE
            ..QuarkConfig::quark_tiny()
        }
    }

    fn micro_batches() -> Vec<DataBatch> {
        let seqs = |seed: u32| -> Vec<Vec<u32>> {
            (0..2).map(|r| (0..9).map(|t| (seed * 7 + r * 13 + t * 5) % 49 + 1).collect()).collect()
        };
        vec![collate_batch(seqs(1), PAD_ID), collate_batch(seqs(2), PAD_ID), collate_batch(seqs(3), PAD_ID)]
    }

    fn temp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("quark-streamed-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// One in-memory step (full autodiff + Burn AdamW, no clipping) for reference.
    fn reference_step(model: QuarkModel<AB>, batches: &[DataBatch], hyper: &AdamWConfig, lr: f64) -> (QuarkModel<AB>, f32) {
        let device = Default::default();
        let mut accumulator = GradientsAccumulator::new();
        let mut loss_sum = 0.0;
        for batch in batches {
            let (ids, labels, _) = batch_tensors::<AB>(batch, &device);
            let (logits, aux) = model.forward_with_aux(ids);
            let [b, s, v] = logits.dims();
            let loss = masked_cross_entropy(logits.reshape([b * s, v]), labels.reshape([b * s]), PAD_ID);
            loss_sum += loss.clone().into_scalar().elem::<f32>();
            let total = loss + aux.unwrap().mul_scalar(AUX_LOSS_COEF);
            let grads = GradientsParams::from_grads(total.div_scalar(batches.len() as f32).backward(), &model);
            accumulator.accumulate(&model, grads);
        }
        let mut optim = hyper.to_burn_config().init::<AB, QuarkModel<AB>>();
        (optim.step(lr, model, accumulator.grads()), loss_sum / batches.len() as f32)
    }

    fn all_params(stages: &TensorStore, cfg: &QuarkConfig) -> Vec<(String, Vec<f32>)> {
        stage_names(cfg)
            .into_iter()
            .flat_map(|s| {
                let data = stages.get(&s).unwrap();
                data.tensors
                    .iter()
                    .map(|(n, t)| (format!("{s}/{n}"), t.to_vec::<f32>().unwrap()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn streamed_step_matches_in_memory_step() {
        let cfg = cfg();
        let device = Default::default();
        <AB as Backend>::seed(&device, 3);
        let dir = temp("equiv");
        let hyper = AdamWConfig { weight_decay: 0.1, ..AdamWConfig::default() };
        let lr = 1e-3;
        let batches = micro_batches();

        // Shared initial weights.
        let model = QuarkModel::<AB>::new(&cfg, &device);
        let init = dir.join("init");
        save_sharded(&model, &cfg, &init, 0).unwrap();

        // Reference: in-memory trainer maths.
        let (reference, ref_loss) = reference_step(model, &batches, &hyper, lr);
        let ref_dir = dir.join("ref");
        save_sharded(&reference, &cfg, &ref_dir, 1).unwrap();

        // Streamed, with a RAM limit so small every stage goes through disk.
        let store = TensorStore::new(dir.join("store"), 1).unwrap();
        let acts = TensorStore::new(dir.join("acts"), 1).unwrap();
        StreamedTrainer::<AB>::restore_into(&store, &init, &cfg, true).unwrap();
        let optim = StreamedOptim { kind: OptimizerKind::AdamW, hyper, max_grad_norm: 0.0 };
        let mut trainer = StreamedTrainer::<AB>::new(cfg.clone(), store, acts, device, optim, 0).unwrap();
        trainer.set_head_chunk_elems(2 * 3 * cfg.vocab_size); // 3 positions per chunk
        let stats = trainer.train_step(&batches, lr).unwrap();
        assert!((stats.loss - ref_loss).abs() < 1e-5, "{} vs {ref_loss}", stats.loss);

        let ref_store = TensorStore::new(dir.join("ref-store"), u64::MAX).unwrap();
        StreamedTrainer::<AB>::restore_into(&ref_store, &ref_dir, &cfg, true).unwrap();
        let ours = all_params(trainer.store(), &cfg);
        let theirs = all_params(&ref_store, &cfg);
        assert_eq!(ours.len(), theirs.len());
        let mut worst = (0.0f32, String::new());
        for ((name, a), (_, b)) in ours.iter().zip(&theirs) {
            for (x, y) in a.iter().zip(b) {
                if (x - y).abs() > worst.0 {
                    worst = ((x - y).abs(), name.clone());
                }
            }
        }
        // Adam's first step is ~g/(|g|+eps), which amplifies summation-order
        // noise in near-zero gradients; a real bug would show up at ~lr.
        assert!(worst.0 < 0.05 * lr as f32, "max weight difference {} in {}", worst.0, worst.1);

        // The saved checkpoint loads as a normal model.
        let ckpt = dir.join("checkpoint-1");
        trainer.save_checkpoint(&ckpt).unwrap();
        let (_, loaded) = load_sharded::<crate::backend::InferBackend>(&ckpt, &Default::default()).unwrap();
        assert_eq!(loaded.num_params(), cfg.param_count() as usize);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn streamed_training_reduces_loss_with_each_optimizer() {
        let cfg = cfg();
        let batches = micro_batches();
        for kind in [OptimizerKind::AdamW, OptimizerKind::AdamWCompact, OptimizerKind::Adafactor] {
            let device = Default::default();
            <AB as Backend>::seed(&device, 5);
            let dir = temp(&format!("{kind:?}"));
            let store = TensorStore::new(dir.join("store"), 64 << 10).unwrap();
            let acts = TensorStore::new(dir.join("acts"), 64 << 10).unwrap();
            let optim = StreamedOptim { kind, hyper: AdamWConfig::default(), max_grad_norm: 1.0 };
            let mut trainer = StreamedTrainer::<AB>::new(cfg.clone(), store, acts, device, optim, 0).unwrap();
            let first = trainer.train_step(&batches, 3e-3).unwrap().loss;
            let mut last = first;
            for _ in 0..15 {
                last = trainer.train_step(&batches, 3e-3).unwrap().loss;
            }
            assert!(last < first * 0.8, "{kind:?}: {first} → {last}");
            let eval = trainer.eval_loss(&batches).unwrap();
            assert!(eval.is_finite() && eval < first, "{kind:?} eval {eval}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
