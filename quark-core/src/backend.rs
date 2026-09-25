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

#[cfg(feature = "backend-cuda")]
pub type TrainBackend = burn_autodiff::Autodiff<burn_cuda::Cuda<f32>>;

#[cfg(all(feature = "backend-wgpu", not(feature = "backend-cuda")))]
pub type TrainBackend = burn_autodiff::Autodiff<burn_wgpu::Wgpu>;

#[cfg(not(any(feature = "backend-cuda", feature = "backend-wgpu")))]
pub type TrainBackend = burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>>;

// ── Inference backend (no autodiff, lower memory overhead) ───────────────────

#[cfg(feature = "backend-cuda")]
pub type InferBackend = burn_cuda::Cuda<f32>;

#[cfg(all(feature = "backend-wgpu", not(feature = "backend-cuda")))]
pub type InferBackend = burn_wgpu::Wgpu;

#[cfg(not(any(feature = "backend-cuda", feature = "backend-wgpu")))]
pub type InferBackend = burn_ndarray::NdArray<f32>;

// ── Compute backends (what autodiff wraps) ───────────────────────────────────

/// f32 compute backend underlying [`TrainBackend`].
pub type ComputeBackend = <TrainBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

/// bf16 compute backend, used for `Precision::Bf16` training (CUDA only).
#[cfg(feature = "backend-cuda")]
pub type ComputeBackendBf16 = burn_cuda::Cuda<half::bf16>;
