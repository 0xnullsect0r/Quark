#![allow(dead_code, unused_imports)]

use tokio::sync::mpsc;

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

pub type MetricsSender = mpsc::UnboundedSender<TrainingMetrics>;
pub type MetricsReceiver = mpsc::UnboundedReceiver<TrainingMetrics>;
