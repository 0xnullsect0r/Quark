//! High-level inference engine: load a checkpoint + tokenizer, generate text.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{Context, Result};
use burn::module::Module;
use burn::record::Recorder;

use crate::checkpoint::CheckpointRecorder;

use crate::backend::InferBackend;
use crate::chat::{find_stop, partial_stop_len};
use crate::inference::generate::{GenerateConfig, generate_with};
use crate::inference::sampling::SamplingParams;
use crate::model::QuarkModel;
use crate::model::config::QuarkConfig;
use crate::tokenizer::bpe::{BOS_ID, QuarkTokenizer};

type Device = burn::tensor::Device<InferBackend>;

/// Loaded model + tokenizer ready for text generation.
pub struct InferenceEngine {
    model: QuarkModel<InferBackend>,
    tokenizer: QuarkTokenizer,
    device: Device,
    max_context: usize,
}

// Burn's NdArray Param types use OnceCell which is not Sync, but InferenceEngine
// is read-only after construction so cross-thread sharing is safe in practice.
unsafe impl Send for InferenceEngine {}
unsafe impl Sync for InferenceEngine {}

impl InferenceEngine {
    /// Load a checkpoint and tokenizer from disk.
    ///
    /// `checkpoint` is a `.bin` file or a sharded `checkpoint-N/` directory
    /// produced by the Quark training loop.
    /// `tokenizer` must be a `tokenizer.json` produced by Quark tokenizer training.
    /// `config` is the model architecture; must match the checkpoint (a sharded
    /// checkpoint's own `config.json` takes precedence).
    pub fn load(checkpoint: &Path, config: &QuarkConfig, tokenizer: &Path) -> Result<Self> {
        let device = Device::default();

        let (config, model) = if crate::checkpoint::sharded::is_sharded(checkpoint) {
            crate::checkpoint::sharded::load_sharded::<InferBackend>(checkpoint, &device)
                .with_context(|| format!("Failed to load checkpoint: {}", checkpoint.display()))?
        } else {
            // Init model skeleton, then load — the recorder adds the extension
            let model = QuarkModel::<InferBackend>::new(config, &device);
            let stem = checkpoint.with_extension("");
            let record = CheckpointRecorder::new()
                .load(stem, &device)
                .with_context(|| format!("Failed to load checkpoint: {}", checkpoint.display()))?;
            (config.clone(), model.load_record(record))
        };

        let tokenizer = QuarkTokenizer::load(tokenizer)
            .with_context(|| format!("Failed to load tokenizer: {}", tokenizer.display()))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            max_context: config.max_position_embeddings,
        })
    }

    /// Generate a response for `prompt`, returning the full decoded string.
    pub fn generate(&self, prompt: &str, params: SamplingParams) -> Result<String> {
        let (tx, _rx) = mpsc::channel();
        self.generate_streaming(prompt, params, tx)
    }

    /// Stream decoded text pieces through `token_tx` as tokens are generated,
    /// then return the full decoded response. The response ends before the
    /// first of `params.stop_strings`. Stops early if the receiver is dropped.
    pub fn generate_streaming(
        &self,
        prompt: &str,
        params: SamplingParams,
        token_tx: mpsc::Sender<String>,
    ) -> Result<String> {
        let ids = self.encode_prompt(prompt)?;
        let stops = params.stop_strings.clone();

        // Decode the whole response so far and emit only the new suffix, so
        // multi-token characters and merged whitespace come out correctly.
        let mut new_ids: Vec<u32> = Vec::new();
        let mut text = String::new();
        let mut emitted = 0usize; // bytes of `text` already sent
        let mut stopped = false;
        generate_with(&self.model, self.generate_config(ids, params), &self.device, |tok| {
            new_ids.push(tok);
            let Ok(decoded) = self.tokenizer.decode(&new_ids) else {
                return true;
            };
            // Hold back incomplete UTF-8 sequences until the next token.
            if decoded.ends_with('\u{FFFD}') {
                return true;
            }
            text = decoded;
            if let Some(at) = find_stop(&text, &stops) {
                text.truncate(at);
                stopped = true;
            }
            // Don't send what might be the start of a stop string.
            let hold = if stopped { 0 } else { partial_stop_len(&text, &stops) };
            let ready = text.len() - hold;
            let mut keep_going = !stopped;
            if ready > emitted && text.is_char_boundary(ready) && text.is_char_boundary(emitted) {
                keep_going &= token_tx.send(text[emitted..ready].to_owned()).is_ok();
                emitted = ready;
            }
            keep_going
        })?;

        if !stopped && emitted < text.len() && text.is_char_boundary(emitted) {
            let _ = token_tx.send(text[emitted..].to_owned());
        }
        Ok(text)
    }

    fn generate_config(&self, prompt_ids: Vec<u32>, sampling: SamplingParams) -> GenerateConfig {
        GenerateConfig {
            prompt_ids,
            sampling,
            seed: 42,
            max_context: self.max_context,
        }
    }

    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let mut ids = vec![BOS_ID];
        let encoded = self.tokenizer.encode(prompt)?;
        ids.extend(encoded);
        Ok(ids)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}
