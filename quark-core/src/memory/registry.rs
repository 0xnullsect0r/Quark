#![allow(dead_code, unused_imports, unused_variables)]

use std::collections::HashMap;
use std::path::PathBuf;

use crate::memory::budget::HardwareBudget;
use crate::memory::tier::TierConfig;

/// Where a layer's weights currently live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerLocation {
    Vram,
    Ram,
    Disk { path: PathBuf },
}

/// Per-layer metadata stored in the registry.
#[derive(Debug, Clone)]
struct LayerEntry {
    location: LayerLocation,
    bytes: u64,
}

/// Tracks which device each model layer is on and its byte size.
/// Maintains an LRU access-order list for eviction (front = oldest).
#[derive(Debug)]
pub struct LayerRegistry {
    layers: HashMap<String, LayerEntry>,
    /// LRU list: front = least-recently-used, back = most-recently-used.
    access_order: Vec<String>,
    vram_used: u64,
    ram_used: u64,
    disk_used: u64,
}

impl LayerRegistry {
    pub fn new() -> Self {
        Self {
            layers: HashMap::new(),
            access_order: Vec::new(),
            vram_used: 0,
            ram_used: 0,
            disk_used: 0,
        }
    }

    /// Register or update a layer's location and byte size.
    pub fn register(&mut self, name: impl Into<String>, location: LayerLocation, bytes: u64) {
        let name: String = name.into();

        // Remove old accounting if the layer already existed.
        if let Some(old) = self.layers.get(&name) {
            match &old.location {
                LayerLocation::Vram => self.vram_used = self.vram_used.saturating_sub(old.bytes),
                LayerLocation::Ram => self.ram_used = self.ram_used.saturating_sub(old.bytes),
                LayerLocation::Disk { .. } => {
                    self.disk_used = self.disk_used.saturating_sub(old.bytes)
                }
            }
        } else {
            // New layer: add to back of LRU list.
            self.access_order.push(name.clone());
        }

        // Add new accounting.
        match &location {
            LayerLocation::Vram => self.vram_used += bytes,
            LayerLocation::Ram => self.ram_used += bytes,
            LayerLocation::Disk { .. } => self.disk_used += bytes,
        }

        self.layers.insert(name, LayerEntry { location, bytes });
    }

    /// Mark a layer as recently accessed (move to back of LRU list).
    pub fn touch(&mut self, name: &str) {
        if let Some(pos) = self.access_order.iter().position(|n| n == name) {
            let n = self.access_order.remove(pos);
            self.access_order.push(n);
        }
    }

    /// Get the location of a layer.
    pub fn get(&self, name: &str) -> Option<&LayerLocation> {
        self.layers.get(name).map(|e| &e.location)
    }

    /// Returns the name of the LRU layer currently in VRAM (eviction candidate).
    pub fn lru_in_vram(&self) -> Option<&str> {
        self.access_order
            .iter()
            .find(|n| {
                self.layers
                    .get(n.as_str())
                    .map(|e| e.location == LayerLocation::Vram)
                    .unwrap_or(false)
            })
            .map(|s| s.as_str())
    }

    /// Returns the name of the LRU layer currently in RAM (eviction candidate).
    pub fn lru_in_ram(&self) -> Option<&str> {
        self.access_order
            .iter()
            .find(|n| {
                self.layers
                    .get(n.as_str())
                    .map(|e| e.location == LayerLocation::Ram)
                    .unwrap_or(false)
            })
            .map(|s| s.as_str())
    }

    /// Current VRAM usage in bytes.
    pub fn vram_used_bytes(&self) -> u64 {
        self.vram_used
    }

    /// Current RAM usage in bytes.
    pub fn ram_used_bytes(&self) -> u64 {
        self.ram_used
    }

    /// Current disk usage in bytes.
    pub fn disk_used_bytes(&self) -> u64 {
        self.disk_used
    }

    /// Returns true if adding `bytes` to VRAM would exceed the configured limit.
    pub fn would_exceed_vram(&self, bytes: u64, budget: &HardwareBudget, tier: &TierConfig) -> bool {
        self.vram_used + bytes > tier.vram_limit_bytes(budget)
    }

    /// Returns true if adding `bytes` to RAM would exceed the configured limit.
    pub fn would_exceed_ram(&self, bytes: u64, budget: &HardwareBudget, tier: &TierConfig) -> bool {
        self.ram_used + bytes > tier.ram_limit_bytes(budget)
    }
}

impl Default for LayerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
