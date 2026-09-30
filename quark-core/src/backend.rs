//! Backend selection.
//!
//! Quark is built with every backend it supports (CUDA, wgpu and the Flex CPU
//! backend) and picks one at runtime: the first that runs a test kernel, in
//! the order CUDA → wgpu (discrete, then integrated GPU) → CPU. Set
//! `QUARK_BACKEND=cuda|wgpu|cpu` (or call [`set_preference`]) to force one.
//!
//! Code names backends only through these aliases. [`ComputeBackend`] is
//! Burn's `Dispatch` backend, whose device ([`device`]) says which real backend
//! runs the ops. Use [`default_device`] instead of `Default::default()` to get
//! the selected device for a backend type.
//!
//! Cargo features choose which backends are compiled in:
//! `backend-cpu` (always), `backend-wgpu`, `backend-cuda`.

use std::sync::{Mutex, OnceLock};

use burn::{
    backend::Autodiff,
    tensor::{backend::Backend, Tensor},
    DispatchDevice,
};

/// The runtime-dispatched compute backend (f32).
pub type ComputeBackend = burn::Dispatch;

/// Training backend (autodiff over [`ComputeBackend`]).
pub type TrainBackend = Autodiff<ComputeBackend>;

/// Inference backend (no autodiff overhead).
pub type InferBackend = ComputeBackend;

/// bf16 compute backend, used for `Precision::Bf16` training when the CUDA
/// backend is selected.
#[cfg(feature = "backend-cuda")]
pub type ComputeBackendBf16 = burn::backend::Cuda<half::bf16>;

/// A backend Quark can run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cuda,
    Wgpu,
    Cpu,
}

impl BackendKind {
    pub fn label(self) -> &'static str {
        match self {
            BackendKind::Cuda => "CUDA",
            BackendKind::Wgpu => "GPU (wgpu)",
            BackendKind::Cpu => "CPU",
        }
    }

    /// Whether this binary was built with the backend.
    pub fn compiled(self) -> bool {
        match self {
            BackendKind::Cuda => cfg!(feature = "backend-cuda"),
            BackendKind::Wgpu => cfg!(feature = "backend-wgpu"),
            BackendKind::Cpu => true,
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cuda" => Some(BackendKind::Cuda),
            "wgpu" | "vulkan" | "metal" | "gpu" => Some(BackendKind::Wgpu),
            "cpu" | "flex" => Some(BackendKind::Cpu),
            _ => None,
        }
    }
}

/// The backend that was selected, with its device and how it was chosen.
#[derive(Debug, Clone)]
pub struct Selection {
    pub kind: BackendKind,
    pub device: DispatchDevice,
    /// Set when a backend was forced but failed, so the CPU is used instead.
    pub error: Option<String>,
    /// One line per backend tried (for logs and the Settings tab).
    pub report: Vec<String>,
}

static PREFERENCE: Mutex<Option<BackendKind>> = Mutex::new(None);
static SELECTION: OnceLock<Selection> = OnceLock::new();

/// Force a backend (`None` = automatic). Only takes effect before the first
/// use of [`device`]; returns `false` if the backend was already chosen.
pub fn set_preference(kind: Option<BackendKind>) -> bool {
    if SELECTION.get().is_some() {
        return false;
    }
    if let Ok(mut p) = PREFERENCE.lock() {
        *p = kind;
    }
    true
}

/// The backend selected for this process (chosen on first call).
pub fn selection() -> &'static Selection {
    SELECTION.get_or_init(select)
}

/// The selected backend.
pub fn selected() -> BackendKind {
    selection().kind
}

/// The selected compute device.
pub fn device() -> DispatchDevice {
    selection().device.clone()
}

/// The selected device for backend `B`: the [`device`] for the dispatch
/// backends (with or without autodiff), `Default::default()` for any other.
pub fn default_device<B: Backend>() -> B::Device {
    let device: Box<dyn std::any::Any> = Box::new(device());
    match device.downcast::<B::Device>() {
        Ok(device) => *device,
        Err(_) => Default::default(),
    }
}

/// Whether `device` is the CPU backend's device. Quantized weights on it stay
/// in host memory and use the CPU decode kernel instead of a device unpack.
pub fn is_cpu_device<D: 'static>(device: &D) -> bool {
    let device: &dyn std::any::Any = device;
    matches!(device.downcast_ref::<DispatchDevice>(), Some(DispatchDevice::Flex(_)))
}

/// Select the backend and report if a forced one failed. Call before loading
/// a model or starting training.
pub fn check_device() -> anyhow::Result<()> {
    match &selection().error {
        Some(e) => Err(anyhow::anyhow!("{e}")),
        None => Ok(()),
    }
}

fn select() -> Selection {
    let forced = PREFERENCE.lock().ok().and_then(|p| *p).or_else(|| {
        std::env::var("QUARK_BACKEND").ok().and_then(|v| BackendKind::parse(&v))
    });
    let order: Vec<BackendKind> = match forced {
        Some(kind) => vec![kind],
        None => vec![BackendKind::Cuda, BackendKind::Wgpu],
    };
    let mut report = Vec::new();
    let mut error = None;
    for kind in order {
        if !kind.compiled() {
            let msg = format!("{}: not included in this build", kind.label());
            if forced.is_some() {
                error = Some(msg.clone());
            }
            report.push(msg);
            continue;
        }
        for device in candidates(kind) {
            match probe(&device) {
                Ok(()) => {
                    report.push(format!("{}: ok ({device:?})", kind.label()));
                    tracing::info!("Compute backend: {} ({device:?})", kind.label());
                    return Selection { kind, device, error: None, report };
                }
                Err(e) => {
                    let msg = format!("{}: {e}", kind.label());
                    tracing::info!("{msg}");
                    if forced.is_some() {
                        error = Some(msg.clone());
                    }
                    report.push(msg);
                }
            }
        }
    }
    report.push("CPU: ok".into());
    tracing::info!("Compute backend: CPU");
    Selection { kind: BackendKind::Cpu, device: cpu_device(), error, report }
}

fn cpu_device() -> DispatchDevice {
    DispatchDevice::Flex(Default::default())
}

/// The devices to try for a backend, best first.
#[allow(unused_mut)]
fn candidates(kind: BackendKind) -> Vec<DispatchDevice> {
    let mut out = Vec::new();
    match kind {
        BackendKind::Cuda => {
            #[cfg(feature = "backend-cuda")]
            out.push(DispatchDevice::Cuda(Default::default()));
        }
        BackendKind::Wgpu => {
            // Skip wgpu's CPU (software) adapter: Flex is faster.
            #[cfg(feature = "backend-wgpu")]
            {
                use burn::backend::wgpu::WgpuDevice;
                out.push(DispatchDevice::Wgpu(WgpuDevice::DiscreteGpu(0)));
                out.push(DispatchDevice::Wgpu(WgpuDevice::IntegratedGpu(0)));
            }
        }
        BackendKind::Cpu => out.push(cpu_device()),
    }
    out
}

/// Run one small kernel on `device`. GPU runtimes often panic on their own
/// worker thread (the caller only sees a closed channel), so panic messages
/// from every thread are collected while the probe runs and kept off stderr.
fn probe(device: &DispatchDevice) -> Result<(), String> {
    use std::sync::Arc;

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
        let x = Tensor::<ComputeBackend, 1>::from_floats([1.0, 2.0, 3.0], device);
        (x * 2.0).sum().into_data().to_vec::<f32>().map(|v| v.first().copied())
    };
    let result = std::panic::catch_unwind(run);
    std::panic::set_hook(previous);

    match result {
        Ok(Ok(Some(sum))) if (sum - 12.0).abs() < 1e-3 => Ok(()),
        Ok(other) => Err(format!("returned a wrong result ({other:?})")),
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
         toolkit and set CUDA_PATH to it (e.g. /usr/local/cuda or /opt/cuda). Quark uses \
         another backend until then."
            .into()
    } else if msg.contains("\"cuda\" shared library") {
        "no NVIDIA driver (libcuda.so)"
            .into()
    } else if msg.contains("UNSUPPORTED_PTX_VERSION") {
        "The CUDA toolkit is newer than your NVIDIA driver supports \
         (CUDA_ERROR_UNSUPPORTED_PTX_VERSION). Compare `nvcc --version` with the \"CUDA \
         Version\" shown by `nvidia-smi`: update the driver (and reboot), or point CUDA_PATH at \
         an older toolkit that matches it. Then delete the cubecl kernel cache \
         (~/.cache/cubecl) and try again."
            .into()
    } else {
        format!("failed to run a test kernel: {first_line}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_runs_a_kernel() {
        check_device().unwrap();
        probe(&device()).unwrap();
        assert!(selected().compiled());
    }

    #[test]
    fn cpu_always_works() {
        probe(&cpu_device()).unwrap();
    }
}
