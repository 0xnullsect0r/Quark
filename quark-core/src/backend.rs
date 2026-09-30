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

/// Run one small kernel on the compute device and turn the usual GPU setup
/// failures into a readable error. Burn panics on these (often many times
/// over), so call this before loading a model or starting training.
/// The result is computed once per process.
pub fn check_device() -> anyhow::Result<()> {
    static RESULT: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    RESULT.get_or_init(probe_device).clone().map_err(anyhow::Error::msg)
}

fn probe_device() -> Result<(), String> {
    use std::sync::{Arc, Mutex};

    use burn::tensor::Tensor;

    // GPU runtimes often panic on their own worker thread, and the caller only
    // sees a closed channel, so collect panic messages from every thread
    // while the probe runs (and keep them off stderr).
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let previous = std::panic::take_hook();
    let sink = Arc::clone(&seen);
    std::panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        if let Ok(mut seen) = sink.lock() {
            seen.push(msg.to_string());
        }
    }));
    let run = || {
        let device = burn::tensor::Device::<ComputeBackend>::default();
        let x = Tensor::<ComputeBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        (x * 2.0).sum().into_data().to_vec::<f32>().map(|v| v.first().copied())
    };
    let result = std::panic::catch_unwind(run);
    std::panic::set_hook(previous);

    match result {
        Ok(Ok(Some(sum))) if (sum - 12.0).abs() < 1e-3 => Ok(()),
        Ok(other) => Err(format!("The compute device returned a wrong result ({other:?}).")),
        Err(_) => {
            let seen = seen.lock().map(|s| s.join("\n")).unwrap_or_default();
            Err(explain_device_error(&seen))
        }
    }
}

/// A short, actionable explanation for a GPU backend failure message.
pub fn explain_device_error(msg: &str) -> String {
    let first_line = msg.lines().next().unwrap_or(msg);
    if msg.contains("CUDA installation not found") {
        "CUDA toolkit not found. The CUDA build compiles GPU kernels at runtime with NVRTC, \
         which comes with the CUDA toolkit (the driver alone is not enough). Install the \
         toolkit and set CUDA_PATH to it (e.g. /usr/local/cuda or /opt/cuda), or use a \
         wgpu build (--features \"backend-cpu backend-wgpu\") instead."
            .into()
    } else if msg.contains("\"cuda\" shared library") {
        "NVIDIA driver not found (libcuda.so). Install the NVIDIA driver, or use a wgpu or \
         CPU build instead."
            .into()
    } else if msg.contains("UNSUPPORTED_PTX_VERSION") {
        "The CUDA toolkit is newer than your NVIDIA driver supports \
         (CUDA_ERROR_UNSUPPORTED_PTX_VERSION). Compare `nvcc --version` with the \"CUDA \
         Version\" shown by `nvidia-smi`: update the driver (and reboot), or point CUDA_PATH at \
         an older toolkit that matches it. Then delete the cubecl kernel cache \
         (~/.cache/cubecl) and try again, or use a wgpu build instead."
            .into()
    } else {
        format!("The compute device failed to run a test kernel: {first_line}")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn device_runs_a_kernel() {
        super::check_device().unwrap();
    }
}
