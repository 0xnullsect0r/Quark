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

/// `(start, end)` byte offsets of a token in the encoded text.
pub type ByteRange = (usize, usize);
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
            // Every byte must be encodable, not just the characters that
            // happen to appear in the corpus (unknown ones are dropped).
            .initial_alphabet(ByteLevel::alphabet())
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

    /// Encode `text`, also returning each token's byte range in `text`.
    ///
    /// With byte-level BPE every character of a token's string stands for one
    /// input byte, so ranges come from the token lengths (the encoding's own
    /// offsets refer to the byte-mapped string, not `text`). A space the
    /// pre-tokenizer adds in front of `text` is not counted.
    pub fn encode_with_offsets(
        &self,
        text: &str,
    ) -> anyhow::Result<(Vec<u32>, Vec<ByteRange>)> {
        let ids = self.encode(text)?;
        let lens: Vec<usize> = ids
            .iter()
            .map(|&id| self.inner.id_to_token(id).map_or(0, |t| t.chars().count()))
            .collect();
        let total: usize = lens.iter().sum();
        if total < text.len() {
            anyhow::bail!(
                "tokenizer cannot encode {} byte(s) of this text — it was trained without \
                 the full byte alphabet; retrain the tokenizer",
                text.len() - total
            );
        }
        let added = total - text.len();
        let mut pos = 0usize;
        let mut skip = added; // leading bytes that aren't in `text`
        let offsets = lens
            .iter()
            .map(|&len| {
                let dropped = skip.min(len);
                skip -= dropped;
                let start = pos;
                pos = (pos + len - dropped).min(text.len());
                (start, pos)
            })
            .collect();
        Ok((ids, offsets))
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

// ─── Background training entry point ─────────────────────────────────────────

/// Messages emitted by the background tokenizer training thread.
#[derive(Debug)]
pub enum TokenizerMessage {
    Log(String),
    Progress(f32),
    Done(PathBuf),
    Error(String),
}

/// Spawn BPE tokenizer training in a background thread.
///
/// `max_bytes` caps how much plain text is sampled before training (0 = no
/// cap, dangerous with large corpora).  JSONL files have their `"text"` field
/// extracted so JSON syntax doesn't pollute the vocabulary.  The sampled text
/// is written to a temp file beside `output_path` and deleted after training.
pub fn start_tokenizer_training(
    corpus_files: Vec<PathBuf>,
    vocab_size: usize,
    output_path: PathBuf,
    max_bytes: u64,
) -> std::sync::mpsc::Receiver<TokenizerMessage> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use std::io::{BufRead, BufReader, Write as _};
        use std::time::Instant;
        let start = Instant::now();

        let n = corpus_files.len();
        let total_bytes: u64 = corpus_files
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum();
        let total_gib = total_bytes as f64 / (1u64 << 30) as f64;
        let cap_label = if max_bytes == 0 {
            "no cap".into()
        } else {
            format!("{:.1} GiB cap", max_bytes as f64 / (1u64 << 30) as f64)
        };

        let _ = tx.send(TokenizerMessage::Log(format!(
            "Found {n} corpus file(s) ({total_gib:.1} GiB on disk, {cap_label})"
        )));
        let _ = tx.send(TokenizerMessage::Log(
            "Sampling text from corpus…".into(),
        ));
        let _ = tx.send(TokenizerMessage::Progress(0.05));

        // ── Sample + extract plain text into a temp file ──────────────────
        let tmp_path = output_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("_tokenizer_corpus_tmp.txt");

        let tmp_file = match std::fs::File::create(&tmp_path) {
            Ok(f) => f,
            Err(e) => {
                let _ = tx.send(TokenizerMessage::Error(
                    format!("Cannot create temp file: {e}"),
                ));
                return;
            }
        };
        let mut writer = std::io::BufWriter::new(tmp_file);
        let mut written_bytes: u64 = 0;
        let mut docs_written: u64 = 0;
        let mut hit_cap = false;

        'files: for path in &corpus_files {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let file = match std::fs::File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    let _ = tx.send(TokenizerMessage::Log(format!(
                        "⚠  Skipping {}: {e}", path.display()
                    )));
                    continue;
                }
            };
            for raw in BufReader::new(file).lines() {
                let Ok(raw) = raw else { continue };
                let text: &str = &if ext == "jsonl" {
                    serde_json::from_str::<serde_json::Value>(&raw)
                        .ok()
                        .and_then(|v| {
                            v.get("text").and_then(|t| t.as_str()).map(str::to_owned)
                        })
                        .unwrap_or(raw)
                } else {
                    raw
                };
                let text = text.trim();
                if text.is_empty() {
                    continue;
                }
                if let Err(e) = writeln!(writer, "{text}") {
                    let _ = tx.send(TokenizerMessage::Error(format!("Write error: {e}")));
                    let _ = std::fs::remove_file(&tmp_path);
                    return;
                }
                written_bytes += text.len() as u64 + 1;
                docs_written += 1;
                if max_bytes > 0 && written_bytes >= max_bytes {
                    hit_cap = true;
                    break 'files;
                }
            }
        }
        drop(writer); // flush

        let written_gib = written_bytes as f64 / (1u64 << 30) as f64;
        if hit_cap {
            let _ = tx.send(TokenizerMessage::Log(format!(
                "Sampled {written_gib:.2} GiB ({docs_written} docs) — cap reached. \
                 Raise the cap to use more data."
            )));
        } else {
            let _ = tx.send(TokenizerMessage::Log(format!(
                "Sampled {written_gib:.2} GiB ({docs_written} docs)"
            )));
        }
        let _ = tx.send(TokenizerMessage::Progress(0.25));

        let _ = tx.send(TokenizerMessage::Log(format!(
            "Training BPE (vocab_size={vocab_size})… this takes ~5–20 min per GiB"
        )));
        let _ = tx.send(TokenizerMessage::Progress(0.30));

        match QuarkTokenizer::train(std::slice::from_ref(&tmp_path), vocab_size, &output_path) {
            Ok(_) => {
                let elapsed = start.elapsed().as_secs_f32();
                let _ = std::fs::remove_file(&tmp_path);
                let _ = tx.send(TokenizerMessage::Progress(1.0));
                let _ = tx.send(TokenizerMessage::Log(format!(
                    "✅  Tokenizer trained in {elapsed:.0}s — saved to {}",
                    output_path.display()
                )));
                let _ = tx.send(TokenizerMessage::Done(output_path));
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp_path);
                let _ = tx.send(TokenizerMessage::Error(format!("{e}")));
            }
        }
    });
    rx
}
