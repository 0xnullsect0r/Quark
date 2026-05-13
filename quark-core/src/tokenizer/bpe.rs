#![allow(dead_code, unused_imports, unused_variables)]

use std::path::{Path, PathBuf};

use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{
    AddedToken, DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    TokenizerBuilder,
};

pub const BOS_TOKEN: &str = "<s>";
pub const EOS_TOKEN: &str = "</s>";
pub const PAD_TOKEN: &str = "<pad>";
pub const UNK_TOKEN: &str = "<unk>";

pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 3;

/// BPE tokenizer wrapper around the HuggingFace `tokenizers` crate.
pub struct QuarkTokenizer {
    inner: tokenizers::Tokenizer,
}

impl QuarkTokenizer {
    /// Train a new BPE tokenizer from corpus files and write it to `output_path`.
    pub fn train(
        corpus_files: &[PathBuf],
        vocab_size: usize,
        output_path: &Path,
    ) -> anyhow::Result<Self> {
        let mut trainer = BpeTrainerBuilder::new()
            .vocab_size(vocab_size)
            .min_frequency(2)
            .special_tokens(vec![
                AddedToken::from(PAD_TOKEN, true),
                AddedToken::from(BOS_TOKEN, true),
                AddedToken::from(EOS_TOKEN, true),
                AddedToken::from(UNK_TOKEN, true),
                AddedToken::from("<|code|>", true),
                AddedToken::from("<|endcode|>", true),
            ])
            .build();

        let mut tokenizer = TokenizerBuilder::<
            BPE,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        >::default()
        .with_model(BPE::default())
        .with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(ByteLevel::default())))
        .with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevel::default())))
        .build()
        .map_err(|e| anyhow::anyhow!("tokenizer build error: {e}"))?;

        let files: Vec<String> = corpus_files
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();

        tokenizer
            .train_from_files(&mut trainer, files)
            .map_err(|e| anyhow::anyhow!("train error: {e}"))?;

        tokenizer
            .save(output_path, false)
            .map_err(|e| anyhow::anyhow!("save error: {e}"))?;

        Ok(Self { inner: tokenizer.into() })
    }

    /// Load a previously trained tokenizer from disk.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("tokenizer load error: {e}"))?;
        Ok(Self { inner })
    }

    /// Encode a text string into token ids.
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode error: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token ids back to a string.
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode error: {e}"))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
