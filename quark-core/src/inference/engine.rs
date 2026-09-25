//! High-level inference engine: load a checkpoint + tokenizer, generate text.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{Context, Result};
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};

use crate::backend::InferBackend;
use crate::inference::generate::{GenerateConfig, generate, generate_with};
use crate::inference::sampling::SamplingParams;
use crate::model::QuarkModel;
use crate::model::config::QuarkConfig;
use crate::tokenizer::bpe::{BOS_ID, QuarkTokenizer};

type Device = <InferBackend as burn::tensor::backend::Backend>::Device;

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
    /// `checkpoint` must be a `.bin` file produced by the Quark training loop.
    /// `tokenizer` must be a `tokenizer.json` produced by Quark tokenizer training.
    /// `config` is the model architecture; must match the checkpoint.
    pub fn load(checkpoint: &Path, config: &QuarkConfig, tokenizer: &Path) -> Result<Self> {
        let device = Device::default();

        // Init model skeleton
        let model = QuarkModel::<InferBackend>::new(config, &device);

        // Load checkpoint — CompactRecorder strips the extension to find the file
        let stem = checkpoint.with_extension("");
        let record = CompactRecorder::new()
            .load(stem, &device)
            .with_context(|| format!("Failed to load checkpoint: {}", checkpoint.display()))?;
        let model = model.load_record(record);

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
        let ids = self.encode_prompt(prompt)?;
        let prompt_len = ids.len();
        let output = generate(&self.model, self.generate_config(ids, params), &self.device)?;
        // Decode only the newly generated tokens (after the prompt)
        self.tokenizer.decode(&output[prompt_len..])
    }

    /// Stream decoded text pieces through `token_tx` as tokens are generated,
    /// then return the full decoded response. Stops early if the receiver is
    /// dropped.
    pub fn generate_streaming(
        &self,
        prompt: &str,
        params: SamplingParams,
        token_tx: mpsc::Sender<String>,
    ) -> Result<String> {
        let ids = self.encode_prompt(prompt)?;
        let prompt_len = ids.len();

        // Decode the whole response so far and emit only the new suffix, so
        // multi-token characters and merged whitespace come out correctly.
        let mut new_ids: Vec<u32> = Vec::new();
        let mut emitted = String::new();
        let output = generate_with(
            &self.model,
            self.generate_config(ids, params),
            &self.device,
            |tok| {
                new_ids.push(tok);
                let Ok(text) = self.tokenizer.decode(&new_ids) else {
                    return true;
                };
                // Hold back incomplete UTF-8 sequences until the next token.
                if text.ends_with('\u{FFFD}') || !text.starts_with(emitted.as_str()) {
                    return true;
                }
                let piece = &text[emitted.len()..];
                if piece.is_empty() {
                    return true;
                }
                let ok = token_tx.send(piece.to_owned()).is_ok();
                emitted = text;
                ok
            },
        )?;

        let text = self.tokenizer.decode(&output[prompt_len..])?;
        if let Some(rest) = text.strip_prefix(emitted.as_str()) {
            if !rest.is_empty() {
                let _ = token_tx.send(rest.to_owned());
            }
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
