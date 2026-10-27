//! Pretraining data as a flat token stream on disk.
//!
//! The corpus is tokenized once, document by document in parallel, into
//! `tokens-<key>.bin`: every document's ids followed by `EOS_ID`, as native
//! `u32`. Training then slices fixed-length windows straight out of the
//! memory-mapped file, so RAM use doesn't grow with the corpus. The key is a
//! hash of the corpus files (path, size, mtime) and the tokenizer file, so a
//! cache is reused by later runs on the same data and rebuilt when either
//! changes.

use std::{
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::{
    data::{batch::DataBatch, loader},
    tokenizer::bpe::{QuarkTokenizer, EOS_ID},
};

/// Bump when the on-disk format changes.
const FORMAT_VERSION: u32 = 1;
/// Text encoded per parallel batch.
const CHUNK_BYTES: usize = 16 << 20;

/// What the cache was built from; saved next to the token file.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Meta {
    version: u32,
    tokens: u64,
    documents: u64,
    files: Vec<PathBuf>,
}

/// A memory-mapped token stream.
pub struct TokenCache {
    mmap: memmap2::Mmap,
    path: PathBuf,
    documents: u64,
}

impl TokenCache {
    /// Open the cache for `corpus` + `tokenizer_path` under `dir`, building it
    /// first if it doesn't exist yet. `log` receives progress lines.
    pub fn open_or_build(
        corpus: &[PathBuf],
        tokenizer_path: &Path,
        tokenizer: &QuarkTokenizer,
        dir: &Path,
        log: &dyn Fn(String),
    ) -> Result<Self> {
        anyhow::ensure!(!corpus.is_empty(), "no corpus files");
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let key = cache_key(corpus, tokenizer_path)?;
        let bin = dir.join(format!("tokens-{key:016x}.bin"));
        let meta_path = bin.with_extension("json");

        if let Some(meta) = read_meta(&meta_path, &bin) {
            log(format!(
                "   Reusing tokenized corpus {} ({} tokens from {} documents)",
                bin.display(),
                meta.tokens,
                meta.documents
            ));
            return Self::open(&bin, meta.documents);
        }

        let meta = build(corpus, tokenizer, &bin, log)?;
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)
            .with_context(|| format!("writing {}", meta_path.display()))?;
        Self::open(&bin, meta.documents)
    }

    fn open(path: &Path, documents: u64) -> Result<Self> {
        let file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the file is only ever replaced through a rename of a complete
        // temporary, never written in place while mapped.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mapping {}", path.display()))?;
        anyhow::ensure!(
            mmap.len() % 4 == 0,
            "{} is not a u32 stream",
            path.display()
        );
        Ok(Self {
            mmap,
            path: path.to_path_buf(),
            documents,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn documents(&self) -> u64 {
        self.documents
    }

    /// Every token, in corpus order.
    pub fn tokens(&self) -> &[u32] {
        bytemuck::cast_slice(&self.mmap[..])
    }

    pub fn len(&self) -> usize {
        self.mmap.len() / 4
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Number of `seq_len` windows whose next-token targets exist.
    pub fn num_windows(&self, seq_len: usize) -> usize {
        if seq_len == 0 {
            return 0;
        }
        self.len().saturating_sub(1) / seq_len
    }

    /// Input ids and their next-token targets for window `idx`.
    pub fn window(&self, idx: usize, seq_len: usize) -> (&[u32], &[u32]) {
        let start = idx * seq_len;
        let tokens = self.tokens();
        (
            &tokens[start..start + seq_len],
            &tokens[start + 1..start + seq_len + 1],
        )
    }

    /// Collate the given windows into one batch (no padding: every window is
    /// exactly `seq_len` long).
    pub fn batch(&self, windows: &[u32], seq_len: usize) -> DataBatch {
        let mut input_ids = Vec::with_capacity(windows.len());
        let mut labels = Vec::with_capacity(windows.len());
        for &w in windows {
            let (x, y) = self.window(w as usize, seq_len);
            input_ids.push(x.to_vec());
            labels.push(y.to_vec());
        }
        let attention_mask = vec![vec![1u8; seq_len]; windows.len()];
        DataBatch {
            input_ids,
            labels,
            attention_mask,
        }
    }
}

/// The batches of one training run: fine-tuning examples held in memory, or
/// windows of a [`TokenCache`] visited in a fresh order every epoch.
pub enum BatchSet {
    InMemory(Vec<DataBatch>),
    Windows {
        cache: Arc<TokenCache>,
        /// The windows this set draws from (training or held-out).
        windows: Vec<u32>,
        seq_len: usize,
        batch_size: usize,
        seed: u64,
        /// The shuffled window order of the epoch it was built for.
        order: Mutex<(u32, Vec<u32>)>,
    },
}

impl BatchSet {
    pub fn windows(
        cache: Arc<TokenCache>,
        windows: Vec<u32>,
        seq_len: usize,
        batch_size: usize,
        seed: u64,
    ) -> Self {
        Self::Windows {
            cache,
            windows,
            seq_len,
            batch_size: batch_size.max(1),
            seed,
            order: Mutex::new((u32::MAX, Vec::new())),
        }
    }

    /// Batches per epoch.
    pub fn len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.len(),
            Self::Windows {
                windows,
                batch_size,
                ..
            } => windows.len() / batch_size,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Batch `idx` (< `len()`) of `epoch`.
    pub fn get(&self, idx: usize, epoch: u32) -> DataBatch {
        match self {
            Self::InMemory(b) => b[idx].clone(),
            Self::Windows {
                cache,
                windows,
                seq_len,
                batch_size,
                seed,
                order,
            } => {
                let mut order = order.lock().unwrap_or_else(|e| e.into_inner());
                if order.0 != epoch {
                    use rand::{seq::SliceRandom, SeedableRng};
                    let mut shuffled = windows.clone();
                    let epoch_seed =
                        seed.wrapping_add((epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    shuffled.shuffle(&mut rand::rngs::StdRng::seed_from_u64(epoch_seed));
                    *order = (epoch, shuffled);
                }
                let start = idx * batch_size;
                cache.batch(&order.1[start..start + batch_size], *seq_len)
            }
        }
    }

    /// The first `n` batches of epoch 0, materialised (for evaluation).
    pub fn take(&self, n: usize) -> Vec<DataBatch> {
        (0..self.len().min(n)).map(|i| self.get(i, 0)).collect()
    }
}

// ── building ──────────────────────────────────────────────────────────────────

fn build(
    corpus: &[PathBuf],
    tokenizer: &QuarkTokenizer,
    bin: &Path,
    log: &dyn Fn(String),
) -> Result<Meta> {
    let total_bytes: u64 = corpus
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
    log(format!(
        "   Tokenizing {} file(s), {:.2} GB of text → {}",
        corpus.len(),
        total_bytes as f64 / 1e9,
        bin.display()
    ));

    let tmp = bin.with_extension("bin.tmp");
    let mut out = BufWriter::with_capacity(
        4 << 20,
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?,
    );
    let started = std::time::Instant::now();
    let mut tokens = 0u64;
    let mut documents = 0u64;
    let mut bytes_done = 0u64;
    let mut next_report = total_bytes / 20; // every 5%
    let mut chunk: Vec<String> = Vec::new();
    let mut chunk_bytes = 0usize;

    // Encode the chunk in parallel and append it; returns (tokens, documents) written.
    let flush =
        |chunk: &mut Vec<String>, out: &mut BufWriter<std::fs::File>| -> Result<(u64, u64)> {
            let encoded: Vec<Vec<u32>> = chunk
                .par_iter()
                .map(|doc| tokenizer.encode(doc).unwrap_or_default())
                .collect();
            let (mut tokens, mut documents) = (0u64, 0u64);
            for ids in encoded {
                if ids.is_empty() {
                    continue;
                }
                out.write_all(bytemuck::cast_slice(&ids))?;
                out.write_all(bytemuck::bytes_of(&EOS_ID))?;
                tokens += ids.len() as u64 + 1;
                documents += 1;
            }
            chunk.clear();
            Ok((tokens, documents))
        };

    for path in corpus {
        loader::for_each_document(path, &mut |doc| {
            chunk_bytes += doc.len();
            bytes_done += doc.len() as u64;
            chunk.push(doc);
            if chunk_bytes >= CHUNK_BYTES {
                let (t, d) = flush(&mut chunk, &mut out)?;
                tokens += t;
                documents += d;
                chunk_bytes = 0;
                if bytes_done >= next_report && total_bytes > 0 {
                    next_report = bytes_done + total_bytes / 20;
                    let secs = started.elapsed().as_secs_f32();
                    log(format!(
                        "   {:.0}%  {:.2} GB tokenized, {} tokens, {:.1} MB/s",
                        bytes_done as f64 * 100.0 / total_bytes as f64,
                        bytes_done as f64 / 1e9,
                        tokens,
                        bytes_done as f32 / 1e6 / secs.max(1e-3)
                    ));
                }
            }
            Ok(())
        })
        .with_context(|| format!("reading {}", path.display()))?;
    }
    let (t, d) = flush(&mut chunk, &mut out)?;
    tokens += t;
    documents += d;
    out.flush()?;
    drop(out);
    anyhow::ensure!(tokens > 0, "no training tokens: the corpus files are empty");
    std::fs::rename(&tmp, bin).with_context(|| format!("moving {} into place", tmp.display()))?;
    log(format!(
        "   Tokenized {documents} documents into {tokens} tokens in {:.0}s",
        started.elapsed().as_secs_f32()
    ));
    Ok(Meta {
        version: FORMAT_VERSION,
        tokens,
        documents,
        files: corpus.to_vec(),
    })
}

fn read_meta(meta_path: &Path, bin: &Path) -> Option<Meta> {
    let meta: Meta = serde_json::from_str(&std::fs::read_to_string(meta_path).ok()?).ok()?;
    let len = std::fs::metadata(bin).ok()?.len();
    (meta.version == FORMAT_VERSION && len == meta.tokens * 4).then_some(meta)
}

/// FNV-1a over the corpus files' identity (path, size, mtime) and the
/// tokenizer file's contents.
fn cache_key(corpus: &[PathBuf], tokenizer_path: &Path) -> Result<u64> {
    let mut h = Fnv::default();
    h.update(&FORMAT_VERSION.to_le_bytes());
    let mut files: Vec<&PathBuf> = corpus.iter().collect();
    files.sort();
    for path in files {
        let meta =
            std::fs::metadata(path).with_context(|| format!("corpus file {}", path.display()))?;
        h.update(path.to_string_lossy().as_bytes());
        h.update(&meta.len().to_le_bytes());
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        h.update(&mtime.to_le_bytes());
    }
    h.update(
        &std::fs::read(tokenizer_path)
            .with_context(|| format!("tokenizer {}", tokenizer_path.display()))?,
    );
    Ok(h.0)
}

struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Fnv {
    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer(dir: &Path) -> (PathBuf, QuarkTokenizer) {
        let corpus = dir.join("tok-corpus.txt");
        std::fs::write(&corpus, "fn main() { println!(\"hello\"); }\n".repeat(50)).unwrap();
        let path = dir.join("tok.json");
        QuarkTokenizer::train(std::slice::from_ref(&corpus), 300, &path).unwrap();
        let tok = QuarkTokenizer::load(&path).unwrap();
        (path, tok)
    }

    #[test]
    fn builds_reuses_and_batches() {
        let dir = std::env::temp_dir().join(format!("quark-token-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (tok_path, tok) = tokenizer(&dir);

        let jsonl = dir.join("docs.jsonl");
        let lines: String = (0..100)
            .map(|i| {
                format!(
                    "{}\n",
                    serde_json::json!({ "text": format!("fn f{i}() {{ {i} }}") })
                )
            })
            .collect();
        std::fs::write(&jsonl, lines).unwrap();
        let txt = dir.join("doc.txt");
        std::fs::write(&txt, "fn main() { println!(\"hello\"); }\n").unwrap();

        let logs = Mutex::new(Vec::new());
        let log = |m: String| logs.lock().unwrap().push(m);
        let corpus = vec![jsonl.clone(), txt.clone()];
        let cache =
            TokenCache::open_or_build(&corpus, &tok_path, &tok, &dir.join("cache"), &log).unwrap();
        assert_eq!(cache.documents(), 101);
        let expected: usize = (0..100)
            .map(|i| tok.encode(&format!("fn f{i}() {{ {i} }}")).unwrap().len() + 1)
            .sum::<usize>()
            + tok
                .encode("fn main() { println!(\"hello\"); }\n")
                .unwrap()
                .len()
            + 1;
        assert_eq!(cache.len(), expected);
        assert_eq!(cache.tokens().iter().filter(|&&t| t == EOS_ID).count(), 101);

        // Windows are consecutive and labels are shifted by one.
        let seq = 8;
        let (x, y) = cache.window(3, seq);
        assert_eq!(x, &cache.tokens()[24..32]);
        assert_eq!(y, &cache.tokens()[25..33]);
        assert_eq!(cache.num_windows(seq), (cache.len() - 1) / seq);

        // Same files → reused, not rebuilt.
        let n_logs = logs.lock().unwrap().len();
        let again =
            TokenCache::open_or_build(&corpus, &tok_path, &tok, &dir.join("cache"), &log).unwrap();
        assert_eq!(again.path(), cache.path());
        assert!(logs.lock().unwrap()[n_logs].contains("Reusing"));

        // Every epoch visits every window once, in a different order.
        let cache = Arc::new(cache);
        let windows: Vec<u32> = (0..cache.num_windows(seq) as u32).collect();
        let set = BatchSet::windows(cache.clone(), windows.clone(), seq, 2, 7);
        let visited = |epoch| -> Vec<u32> {
            let mut seen = Vec::new();
            for i in 0..set.len() {
                let b = set.get(i, epoch);
                assert_eq!(b.input_ids.len(), 2);
                for row in &b.input_ids {
                    let start = cache
                        .tokens()
                        .windows(seq)
                        .position(|w| w == row.as_slice())
                        .unwrap();
                    seen.push((start / seq) as u32);
                }
            }
            seen
        };
        let e0 = visited(0);
        let e1 = visited(1);
        assert_ne!(e0, e1);
        let mut sorted = e0.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), set.len() * 2);
        assert_eq!(set.take(3).len(), 3);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
