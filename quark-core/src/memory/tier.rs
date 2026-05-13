#![allow(dead_code, unused_imports, unused_variables)]

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::budget::HardwareBudget;

/// Fractions of each resource tier that Quark is allowed to use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    /// Fraction of detected VRAM to allow (0.0–1.0, default 0.60).
    pub vram_limit_frac: f32,
    /// Fraction of detected RAM to allow (0.0–1.0, default 0.75).
    pub ram_limit_frac: f32,
    /// Fraction of CPU threads to use (0.0–1.0, default 0.80).
    pub cpu_thread_frac: f32,
    /// Fraction of GPU compute to use (0.0–1.0, default 0.90).
    pub gpu_compute_frac: f32,
    /// Path for disk-offloaded tensors.
    pub disk_offload_path: PathBuf,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            vram_limit_frac: 0.60,
            ram_limit_frac: 0.75,
            cpu_thread_frac: 0.80,
            gpu_compute_frac: 0.90,
            disk_offload_path: PathBuf::from("offload"),
        }
    }
}

impl TierConfig {
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let s = toml::to_string_pretty(self)?;
        std::fs::write(path, s)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&s)?)
    }

    pub fn vram_limit_bytes(&self, budget: &HardwareBudget) -> u64 {
        (budget.vram_total_bytes as f64 * self.vram_limit_frac as f64) as u64
    }

    pub fn ram_limit_bytes(&self, budget: &HardwareBudget) -> u64 {
        (budget.ram_total_bytes as f64 * self.ram_limit_frac as f64) as u64
    }

    pub fn cpu_thread_count(&self, budget: &HardwareBudget) -> u32 {
        ((budget.cpu_logical_cores as f64 * self.cpu_thread_frac as f64) as u32).max(1)
    }
}
