#![allow(dead_code, unused_imports, unused_variables)]

use serde::{Deserialize, Serialize};
use sysinfo::System;

/// Detected hardware resource limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareBudget {
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
    pub ram_total_bytes: u64,
    pub ram_free_bytes: u64,
    pub cpu_logical_cores: u32,
    pub disk_free_bytes: u64,
}

impl HardwareBudget {
    /// Probe the current system and return the detected resource budget.
    pub fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let ram_total_bytes = sys.total_memory();
        let ram_free_bytes = sys.available_memory();
        let cpu_logical_cores = sys.cpus().len() as u32;

        // Disk: use the root/current directory mount as a proxy.
        let disk_free_bytes = {
            use sysinfo::Disks;
            let disks = Disks::new_with_refreshed_list();
            disks
                .iter()
                .map(|d| d.available_space())
                .max()
                .unwrap_or(0)
        };

        // VRAM requires a CUDA/Vulkan query; report 0 if unavailable.
        Self {
            vram_total_bytes: 0,
            vram_free_bytes: 0,
            ram_total_bytes,
            ram_free_bytes,
            cpu_logical_cores,
            disk_free_bytes,
        }
    }
}
