#![allow(dead_code, unused_imports)]

pub mod hf_import;
pub mod safetensors;

pub use safetensors::{load_checkpoint, save_checkpoint, TensorData};

/// Recorder used for training checkpoints: full-precision weights in a
/// `.bin` file (the extension the GUI, quark-chat and quark-code look for).
pub type CheckpointRecorder = burn::record::BinFileRecorder<burn::record::FullPrecisionSettings>;
