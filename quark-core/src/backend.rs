//! Compile-time backend selection.
//!
//! Use `TrainBackend` (autodiff, needed for gradient computation) or
//! `InferBackend` (inference-only, no autodiff overhead) instead of
//! naming specific Burn backends directly.
//!
//! Priority: CUDA > WGPU > Flex (pure-Rust CPU fallback).
//!
//! Build combinations:
//!   NVIDIA  →  `--features "backend-cpu backend-cuda"`
//!   AMD / Intel / Apple  →  `--features "backend-cpu backend-wgpu"`
//!   CPU     →  `--features backend-cpu`

use burn::backend::Autodiff;

// ── Compute backend (f32) ─────────────────────────────────────────────────────

#[cfg(feature = "backend-cuda")]
pub type ComputeBackend = burn::backend::Cuda<f32>;

#[cfg(all(feature = "backend-wgpu", not(feature = "backend-cuda")))]
pub type ComputeBackend = burn::backend::Wgpu;

#[cfg(not(any(feature = "backend-cuda", feature = "backend-wgpu")))]
pub type ComputeBackend = burn::backend::Flex;

/// Training backend (autodiff over [`ComputeBackend`]).
pub type TrainBackend = Autodiff<ComputeBackend>;

/// Inference backend (no autodiff overhead).
pub type InferBackend = ComputeBackend;

/// bf16 compute backend, used for `Precision::Bf16` training (CUDA only).
#[cfg(feature = "backend-cuda")]
pub type ComputeBackendBf16 = burn::backend::Cuda<half::bf16>;
