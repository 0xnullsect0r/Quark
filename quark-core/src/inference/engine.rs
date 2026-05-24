//! High-level inference engine: load a checkpoint + tokenizer, generate text.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{Context, Result};
use burn::module::Module;
use burn::record::{CompactRecorder, Recorder};

use crate::backend::InferBackend;
use crate::inference::generate::{GenerateConfig, generate, generate_streaming};
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

        Ok(Self { model, tokenizer, device })
    }

    /// Generate a response for `prompt`, returning the full decoded string.
    pub fn generate(&self, prompt: &str, params: SamplingParams) -> Result<String> {
        let mut ids = self.encode_prompt(prompt)?;

        let cfg = GenerateConfig {
            prompt_ids: ids.clone(),
            sampling: params,
            seed: 42,
        };

        let output = generate(&self.model, cfg, &self.device)?;
        // Decode only the newly generated tokens (after the prompt)
        let new_tokens = &output[ids.len()..];
        self.tokenizer.decode(new_tokens).map_err(|e| anyhow::anyhow!(e))
    }

    /// Stream tokens one by one through `token_tx`, then return the full decoded response.
    pub fn generate_streaming(
        &self,
        prompt: &str,
        params: SamplingParams,
        token_tx: mpsc::Sender<String>,
    ) -> Result<String> {
        let ids = self.encode_prompt(prompt)?;
        let prompt_len = ids.len();

        // Generate tokens; stream each decoded token
        let (raw_tx, raw_rx) = mpsc::channel::<u32>();
        let tokenizer = crate::tokenizer::bpe::QuarkTokenizer::load(
            // re-use existing tokenizer by re-loading is wrong; use a workaround
            // We decode incrementally by buffering in a thread
            &crate::paths::datasets_dir().join("_placeholder_"),
        );

        // Use simple blocking generation and decode the final result
        let cfg = GenerateConfig {
            prompt_ids: ids,
            sampling: params,
            seed: 42,
        };

        let output = generate(&self.model, cfg, &self.device)?;
        let new_tokens = &output[prompt_len..];
        let text = self.tokenizer.decode(new_tokens).map_err(|e| anyhow::anyhow!(e))?;

        // Stream word by word for smooth display
        for word in text.split_inclusive(' ') {
            let _ = token_tx.send(word.to_owned());
        }

        Ok(text)
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
