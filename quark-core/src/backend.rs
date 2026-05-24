//! Compile-time backend selection.
//!
//! Use `TrainBackend` (autodiff, needed for gradient computation) or
//! `InferBackend` (inference-only, no autodiff overhead) instead of
//! naming specific Burn backends directly.
//!
//! Priority: CUDA > WGPU > NdArray (CPU-only fallback).
//!
//! Build combinations:
//!   NVIDIA  →  `--features "backend-cpu backend-cuda"`
//!   AMD     →  `--features "backend-cpu backend-wgpu"`
//!   CPU     →  `--features backend-cpu`

// ── Training backend (AutodiffBackend required for gradients) ─────────────────

#[cfg(all(feature = "backend-cuda", not(feature = "backend-wgpu")))]
pub type TrainBackend = burn_autodiff::Autodiff<burn_cuda::Cuda<f32>>;

#[cfg(feature = "backend-wgpu")]
pub type TrainBackend = burn_autodiff::Autodiff<burn_wgpu::Wgpu>;

#[cfg(not(any(feature = "backend-cuda", feature = "backend-wgpu")))]
pub type TrainBackend = burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>>;

// ── Inference backend (no autodiff, lower memory overhead) ───────────────────

#[cfg(all(feature = "backend-cuda", not(feature = "backend-wgpu")))]
pub type InferBackend = burn_cuda::Cuda<f32>;

#[cfg(feature = "backend-wgpu")]
pub type InferBackend = burn_wgpu::Wgpu;

#[cfg(not(any(feature = "backend-cuda", feature = "backend-wgpu")))]
pub type InferBackend = burn_ndarray::NdArray<f32>;
