#![allow(dead_code, unused_imports)]

/// Snapshot of training state emitted after each optimiser step.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub step: u64,
    pub loss: f32,
    pub learning_rate: f32,
    pub tokens_per_sec: f32,
    pub grad_norm: f32,
    pub vram_used_bytes: u64,
    pub ram_used_bytes: u64,
    pub disk_used_bytes: u64,
    pub epoch: u32,
    pub eta_secs: u64,
}

/// Events emitted by the training loop to the GUI over an mpsc channel.
#[derive(Debug)]
pub enum TrainingEvent {
    /// Numeric metrics snapshot after one optimiser step.
    Metrics(TrainingMetrics),
    /// Human-readable log line.
    Log(String),
    /// Short phase description for the status bar.
    Phase(String),
    /// Training completed successfully.
    Done,
    /// Training aborted; contains an error description.
    Error(String),
}

pub type MetricsSender = std::sync::mpsc::Sender<TrainingEvent>;
pub type MetricsReceiver = std::sync::mpsc::Receiver<TrainingEvent>;
