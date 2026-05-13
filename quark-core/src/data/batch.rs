#![allow(dead_code, unused_imports, unused_variables)]

/// A single training mini-batch of token ids.
#[derive(Debug, Clone)]
pub struct DataBatch {
    /// Shape: [batch, seq]
    pub input_ids: Vec<Vec<u32>>,
    /// Shape: [batch, seq] — shifted by one for next-token prediction.
    pub labels: Vec<Vec<u32>>,
    /// Shape: [batch, seq] — 1 for real tokens, 0 for padding.
    pub attention_mask: Vec<Vec<u8>>,
}

/// Collate a list of token sequences into a `DataBatch`, padding to the longest
/// sequence in the list.
///
/// `labels` are the input ids shifted left by one position (next-token
/// prediction).  The last position of each label sequence is set to `pad_id`
/// (ignored during loss computation).
pub fn collate_batch(sequences: Vec<Vec<u32>>, pad_id: u32) -> DataBatch {
    if sequences.is_empty() {
        return DataBatch {
            input_ids: Vec::new(),
            labels: Vec::new(),
            attention_mask: Vec::new(),
        };
    }

    let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);

    let mut input_ids = Vec::with_capacity(sequences.len());
    let mut labels = Vec::with_capacity(sequences.len());
    let mut attention_mask = Vec::with_capacity(sequences.len());

    for seq in sequences {
        let seq_len = seq.len();

        // Pad input to max_len
        let mut padded = seq.clone();
        padded.resize(max_len, pad_id);

        // Attention mask: 1 for real tokens, 0 for padding
        let mut mask = vec![0u8; max_len];
        for i in 0..seq_len {
            mask[i] = 1;
        }

        // Labels: shift left by 1; last position → pad_id (ignored)
        let mut lbl = Vec::with_capacity(max_len);
        for i in 1..seq_len {
            lbl.push(seq[i]);
        }
        lbl.push(pad_id); // last real position has no target
        lbl.resize(max_len, pad_id); // pad positions

        input_ids.push(padded);
        labels.push(lbl);
        attention_mask.push(mask);
    }

    DataBatch {
        input_ids,
        labels,
        attention_mask,
    }
}
