#![allow(dead_code, unused_imports, unused_variables)]

/// Aggregate statistics about the training dataset.
#[derive(Debug, Clone, Default)]
pub struct DatasetStats {
    pub num_files: usize,
    pub num_documents: usize,
    pub total_tokens: u64,
    pub avg_doc_len: f64,
    pub total_bytes: u64,
}

impl DatasetStats {
    /// Compute statistics by tokenising every document in `loader`.
    pub fn compute(
        loader: &crate::data::loader::TextLoader,
        tokenizer: &crate::tokenizer::QuarkTokenizer,
    ) -> anyhow::Result<Self> {
        let texts = loader.load_texts()?;
        let mut total_tokens: u64 = 0;
        let mut total_len: usize = 0;

        for text in &texts {
            if let Ok(ids) = tokenizer.encode(text) {
                total_tokens += ids.len() as u64;
                total_len += ids.len();
            }
        }

        let num_docs = texts.len();
        let avg_doc_len = if num_docs > 0 {
            total_len as f64 / num_docs as f64
        } else {
            0.0
        };

        Ok(Self {
            num_files: loader.num_files(),
            num_documents: num_docs,
            total_tokens,
            avg_doc_len,
            total_bytes: loader.total_bytes(),
        })
    }

    /// Estimate the number of optimiser steps for a given `batch_size` and
    /// `max_seq_len` (tokens consumed per step = `max_seq_len * batch_size`).
    pub fn estimate_steps(&self, max_seq_len: usize, batch_size: usize) -> u64 {
        let tokens_per_step = max_seq_len as u64 * batch_size as u64;
        if tokens_per_step == 0 {
            return 0;
        }
        self.total_tokens / tokens_per_step
    }
}
