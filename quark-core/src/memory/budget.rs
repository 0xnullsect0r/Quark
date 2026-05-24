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

        let disk_free_bytes = {
            use sysinfo::Disks;
            let disks = Disks::new_with_refreshed_list();
            disks.iter().map(|d| d.available_space()).max().unwrap_or(0)
        };

        let (vram_total_bytes, vram_free_bytes) = detect_vram();

        Self {
            vram_total_bytes,
            vram_free_bytes,
            ram_total_bytes,
            ram_free_bytes,
            cpu_logical_cores,
            disk_free_bytes,
        }
    }
}

fn detect_vram() -> (u64, u64) {
    // ── NVIDIA via NVML ───────────────────────────────────────────────────────
    #[cfg(feature = "backend-cuda")]
    {
        if let Ok(nvml) = nvml_wrapper::Nvml::init() {
            if let Ok(device) = nvml.device_by_index(0) {
                if let Ok(mem) = device.memory_info() {
                    return (mem.total, mem.free);
                }
            }
        }
    }

    // ── AMD / Intel via wgpu adapter ─────────────────────────────────────────
    #[cfg(all(feature = "backend-wgpu", not(feature = "backend-cuda")))]
    {
        use wgpu::{Instance, InstanceDescriptor, PowerPreference, RequestAdapterOptions};
        let instance = Instance::new(InstanceDescriptor::default());
        if let Some(adapter) = pollster::block_on(instance.request_adapter(
            &RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        )) {
            // wgpu exposes max_buffer_size as the largest allocation the driver
            // will allow — a conservative proxy for available device memory.
            let limits = adapter.limits();
            let total = limits.max_buffer_size;
            return (total, total);
        }
    }

    (0, 0)
}
