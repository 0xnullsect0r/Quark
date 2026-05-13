#![allow(dead_code, unused_imports, unused_variables)]

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::Result;

/// Iterator over tokenised text sequences loaded from disk.
pub struct TextLoader {
    files: Vec<PathBuf>,
    max_seq_len: usize,
}

impl TextLoader {
    pub fn new(files: Vec<PathBuf>, max_seq_len: usize) -> Self {
        Self { files, max_seq_len }
    }

    pub fn num_files(&self) -> usize {
        self.files.len()
    }

    /// Walk a directory recursively and collect all `.txt` and `.jsonl` files.
    pub fn from_dir(dir: &Path, max_seq_len: usize) -> Result<Self> {
        let mut files = Vec::new();
        collect_files(dir, &mut files)?;
        Ok(Self { files, max_seq_len })
    }

    /// Load all text from files, returning one `String` per document.
    /// For `.jsonl` files the `"text"` field is extracted from each line when
    /// present; otherwise the raw line is used.  For `.txt` files the whole
    /// file is returned as a single document.
    pub fn load_texts(&self) -> Result<Vec<String>> {
        let mut texts = Vec::new();
        for path in &self.files {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            match ext {
                "jsonl" => {
                    let file = File::open(path)?;
                    for line in BufReader::new(file).lines() {
                        let line = line?;
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let text = extract_text_field(trimmed);
                        texts.push(text);
                    }
                }
                _ => {
                    // .txt or anything else: whole file as one document
                    let content = std::fs::read_to_string(path)?;
                    if !content.trim().is_empty() {
                        texts.push(content);
                    }
                }
            }
        }
        Ok(texts)
    }

    /// Estimate total file size in bytes.
    pub fn total_bytes(&self) -> u64 {
        self.files
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext, "txt" | "jsonl") {
                out.push(path);
            }
        }
    }
    Ok(())
}

/// Try to extract the `"text"` field from a JSON object line; fall back to the
/// raw line if parsing fails or the field is absent.
fn extract_text_field(line: &str) -> String {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
        if let Some(text) = val.get("text").and_then(|v| v.as_str()) {
            return text.to_owned();
        }
    }
    line.to_owned()
}
