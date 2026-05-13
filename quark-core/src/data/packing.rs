#![allow(dead_code, unused_imports, unused_variables)]

use crate::tokenizer::bpe::EOS_ID;

/// Pack variable-length sequences into fixed-length chunks to maximise GPU
/// utilisation.  Sequences are concatenated with an EOS token between them
/// and then split into chunks of exactly `max_seq_len` tokens.  The final
/// partial chunk (if any) is kept as-is.
pub fn pack_sequences(sequences: Vec<Vec<u32>>, max_seq_len: usize) -> Vec<Vec<u32>> {
    if max_seq_len == 0 {
        return Vec::new();
    }

    let mut packed: Vec<Vec<u32>> = Vec::new();
    let mut current: Vec<u32> = Vec::new();

    for seq in sequences {
        for token in seq {
            current.push(token);
            if current.len() >= max_seq_len {
                packed.push(current[..max_seq_len].to_vec());
                current = current[max_seq_len..].to_vec();
            }
        }
        // EOS separator between documents
        current.push(EOS_ID);
        if current.len() >= max_seq_len {
            packed.push(current[..max_seq_len].to_vec());
            current = current[max_seq_len..].to_vec();
        }
    }

    if !current.is_empty() {
        packed.push(current);
    }

    packed
}
