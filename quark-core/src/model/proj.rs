//! Projection layer (`y = x · W`) that is either dense (training) or holds
//! block-quantized weights (inference of large models).
//!
//! Quantization is symmetric and block-wise along the input dimension: each
//! output column's weights are split into groups of `group` inputs with one f32
//! scale per group. Values are packed into `u32` words: 8 × 4-bit (`Q4`,
//! stored offset by +8) or 4 × 8-bit (`Q8`, offset by +128).
//!
//! * GPU backends keep the packed words on the device and unpack them with
//!   bitwise ops right before the matmul (one layer's weights at a time).
//! * The CPU backend keeps them in host memory and runs a multithreaded
//!   quantized matrix–vector kernel for decoding ([`HostQuant::matvec`]),
//!   which reads the 4-bit data directly (decode is memory-bound).

use std::sync::Arc;

use burn::{
    module::{Module, Param, ParamId},
    nn::LinearConfig,
    tensor::{backend::Backend, module::linear, ElementConversion, Int, Tensor, TensorData},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Quantization format of a model's projection weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantFormat {
    Q4,
    Q8,
}

impl QuantFormat {
    pub fn bits(self) -> usize {
        match self {
            QuantFormat::Q4 => 4,
            QuantFormat::Q8 => 8,
        }
    }

    /// Values per packed `u32` word.
    pub fn per_word(self) -> usize {
        32 / self.bits()
    }

    fn max_level(self) -> f32 {
        match self {
            QuantFormat::Q4 => 7.0,
            QuantFormat::Q8 => 127.0,
        }
    }

    fn offset(self) -> i32 {
        match self {
            QuantFormat::Q4 => 8,
            QuantFormat::Q8 => 128,
        }
    }

    fn mask(self) -> u32 {
        (1u32 << self.bits()) - 1
    }
}

/// Largest of 32/16/8 that divides `in_features`.
pub fn group_size(in_features: usize) -> usize {
    [32, 16, 8].into_iter().find(|g| in_features.is_multiple_of(*g)).unwrap_or(in_features)
}

/// Packed weight words, either owned or a view into a memory-mapped
/// checkpoint file (the OS then pages weights in from disk as needed, so a
/// model larger than RAM still runs, and loading is near-instant).
#[derive(Clone)]
pub enum Words {
    Owned(Vec<u32>),
    Mapped { map: Arc<memmap2::Mmap>, offset: usize, len: usize },
}

impl Words {
    pub fn as_slice(&self) -> &[u32] {
        match self {
            Words::Owned(v) => v,
            // Alignment is checked when the view is created.
            Words::Mapped { map, offset, len } => bytemuck::cast_slice(&map[*offset..*offset + len * 4]),
        }
    }

    /// A view of `len` words at byte `offset` of `map`, if 4-byte aligned.
    pub fn mapped(map: Arc<memmap2::Mmap>, offset: usize, len: usize) -> Option<Self> {
        let bytes = map.get(offset..offset + len * 4)?;
        bytemuck::try_cast_slice::<u8, u32>(bytes).ok()?;
        Some(Words::Mapped { map, offset, len })
    }
}

impl std::fmt::Debug for Words {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Words::Owned(v) => write!(f, "Owned({} words)", v.len()),
            Words::Mapped { len, .. } => write!(f, "Mapped({len} words)"),
        }
    }
}

/// Quantized weights in host memory, laid out `[out][in]` (transposed from
/// Burn's `[in, out]`), so each output reads one contiguous row.
#[derive(Debug, Clone)]
pub struct HostQuant {
    pub format: QuantFormat,
    pub in_features: usize,
    pub out_features: usize,
    pub group: usize,
    /// `[out, in / per_word]`
    pub packed: Words,
    /// `[out, in / group]`
    pub scales: Vec<f32>,
}

impl HostQuant {
    /// Quantize a dense `[in, out]` weight (row-major).
    pub fn quantize(format: QuantFormat, weight: &[f32], in_features: usize, out_features: usize) -> Self {
        assert_eq!(weight.len(), in_features * out_features);
        assert!(in_features.is_multiple_of(format.per_word()), "in_features must be a multiple of {}", format.per_word());
        let group = group_size(in_features);
        let groups = in_features / group;
        let words = in_features / format.per_word();
        let mut packed = vec![0u32; out_features * words];
        let mut scales = vec![0f32; out_features * groups];
        packed
            .par_chunks_mut(words)
            .zip(scales.par_chunks_mut(groups))
            .enumerate()
            .for_each(|(o, (row_words, row_scales))| {
                let col = |i: usize| weight[i * out_features + o];
                for (g, scale) in row_scales.iter_mut().enumerate() {
                    let absmax = (g * group..(g + 1) * group).map(|i| col(i).abs()).fold(0.0f32, f32::max);
                    *scale = if absmax > 0.0 { absmax / format.max_level() } else { 1.0 };
                }
                for (i, _) in (0..in_features).map(|i| (i, ())) {
                    let s = row_scales[i / group];
                    let q = (col(i) / s).round().clamp(-format.max_level(), format.max_level()) as i32 + format.offset();
                    let shift = (i % format.per_word()) * format.bits();
                    row_words[i / format.per_word()] |= (q as u32 & format.mask()) << shift;
                }
            });
        Self { format, in_features, out_features, group, packed: Words::Owned(packed), scales }
    }

    /// Dequantize to a dense `[in, out]` weight (row-major).
    pub fn dequantize(&self) -> Vec<f32> {
        let (inf, outf) = (self.in_features, self.out_features);
        let mut out = vec![0f32; inf * outf];
        let rows: Vec<Vec<f32>> = (0..outf).into_par_iter().map(|o| self.row(o)).collect();
        for (o, row) in rows.iter().enumerate() {
            for (i, v) in row.iter().enumerate() {
                out[i * outf + o] = *v;
            }
        }
        out
    }

    fn row(&self, o: usize) -> Vec<f32> {
        let f = self.format;
        let words = self.in_features / f.per_word();
        let groups = self.in_features / self.group;
        let row = &self.packed.as_slice()[o * words..(o + 1) * words];
        let scales = &self.scales[o * groups..(o + 1) * groups];
        (0..self.in_features)
            .map(|i| {
                let q = (row[i / f.per_word()] >> ((i % f.per_word()) * f.bits())) & f.mask();
                (q as i32 - f.offset()) as f32 * scales[i / self.group]
            })
            .collect()
    }

    /// `y = x · W` for each row of `x` (`[rows, in]` row-major), reading the
    /// packed weights directly. Returns `[rows, out]`.
    ///
    /// Like llama.cpp, each group of `x` is quantized to int8 (absmax), so a
    /// group's dot product is an integer sum straight over the packed bytes
    /// (which the compiler vectorises): with `q` the stored levels and `xq`
    /// the int8 inputs, `Σ (q - off)·x ≈ s_w · s_x · (Σ q·xq - off·Σ xq)`.
    pub fn matvec(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let f = self.format;
        let (inf, outf, group) = (self.in_features, self.out_features, self.group);
        let groups = inf / group;
        let row_bytes = inf * f.bits() / 8;
        let group_bytes = group * f.bits() / 8;
        let offset = f.offset();
        let bytes: &[u8] = bytemuck::cast_slice(self.packed.as_slice());

        // Quantize x per group: xq (int8), its scale and its sum.
        let mut xq = vec![0i8; rows * inf];
        let mut xs = vec![0f32; rows * groups];
        let mut xsum = vec![0i32; rows * groups];
        for r in 0..rows {
            for g in 0..groups {
                let src = &x[r * inf + g * group..r * inf + (g + 1) * group];
                let absmax = src.iter().fold(0f32, |m, v| m.max(v.abs()));
                let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
                let dst = &mut xq[r * inf + g * group..r * inf + (g + 1) * group];
                let mut sum = 0i32;
                for (d, v) in dst.iter_mut().zip(src) {
                    *d = (v / scale).round() as i8;
                    sum += *d as i32;
                }
                xs[r * groups + g] = scale;
                xsum[r * groups + g] = sum;
            }
        }

        let cols: Vec<Vec<f32>> = (0..outf)
            .into_par_iter()
            .map(|o| {
                let w = &bytes[o * row_bytes..(o + 1) * row_bytes];
                let s = &self.scales[o * groups..(o + 1) * groups];
                (0..rows)
                    .map(|r| {
                        let mut acc = 0f32;
                        for g in 0..groups {
                            let wb = &w[g * group_bytes..(g + 1) * group_bytes];
                            let xg = &xq[r * inf + g * group..r * inf + (g + 1) * group];
                            let dot = match f {
                                QuantFormat::Q4 => dot_q4(wb, xg),
                                QuantFormat::Q8 => dot_q8(wb, xg),
                            };
                            let corrected = dot - offset * xsum[r * groups + g];
                            acc += corrected as f32 * s[g] * xs[r * groups + g];
                        }
                        acc
                    })
                    .collect()
            })
            .collect();
        let mut y = vec![0f32; rows * outf];
        for (o, col) in cols.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                y[r * outf + o] = *v;
            }
        }
        y
    }

    pub fn bytes(&self) -> usize {
        self.packed.as_slice().len() * 4 + self.scales.len() * 4
    }
}

/// Σ level·x over one group of 4-bit levels (two per byte, low nibble first).
#[inline]
fn dot_q4(bytes: &[u8], x: &[i8]) -> i32 {
    bytes
        .iter()
        .zip(x.chunks_exact(2))
        .map(|(&b, xs)| (b & 15) as i32 * xs[0] as i32 + (b >> 4) as i32 * xs[1] as i32)
        .sum()
}

/// Σ level·x over one group of 8-bit levels.
#[inline]
fn dot_q8(bytes: &[u8], x: &[i8]) -> i32 {
    bytes.iter().zip(x).map(|(&b, &xv)| b as i32 * xv as i32).sum()
}

/// Device-side quantized weights (GPU backends) plus, on the CPU backend, the
/// host copy used by the decode kernel.
#[derive(Module, Debug)]
pub struct QuantParams<B: Backend> {
    /// `[out, in / per_word]` packed words (as i32), or `[1, 1]` on CPU.
    pub packed: Param<Tensor<B, 2, Int>>,
    /// `[out, in / group]`, or `[1, 1]` on CPU.
    pub scales: Param<Tensor<B, 2>>,
    #[module(skip)]
    pub host: Option<Arc<HostQuant>>,
    #[module(skip)]
    pub format: QuantFormat,
    #[module(skip)]
    pub in_features: usize,
    #[module(skip)]
    pub out_features: usize,
}

/// Whether quantized weights stay in host memory (CPU backend) rather than on
/// the compute device.
pub const HOST_QUANT: bool = !cfg!(any(feature = "backend-cuda", feature = "backend-wgpu"));

impl<B: Backend> QuantParams<B> {
    pub fn from_host(q: HostQuant, device: &B::Device) -> Self {
        Self::from_host_with(q, device, HOST_QUANT)
    }

    /// `keep_on_host` selects the CPU (host kernel) or device (bitwise unpack)
    /// representation.
    pub fn from_host_with(q: HostQuant, device: &B::Device, keep_on_host: bool) -> Self {
        let (format, in_features, out_features) = (q.format, q.in_features, q.out_features);
        if keep_on_host {
            return Self {
                packed: Param::initialized(ParamId::new(), Tensor::<B, 2, Int>::zeros([1, 1], device)),
                scales: Param::from_tensor(Tensor::zeros([1, 1], device)),
                host: Some(Arc::new(q)),
                format,
                in_features,
                out_features,
            };
        }
        let words = in_features / format.per_word();
        let groups = in_features / q.group;
        let packed: Vec<i32> = q.packed.as_slice().iter().map(|&w| w as i32).collect();
        Self {
            packed: Param::initialized(
                ParamId::new(),
                Tensor::<B, 2, Int>::from_data(TensorData::new(packed, [out_features, words]), device),
            ),
            scales: Param::from_tensor(Tensor::from_data(TensorData::new(q.scales, [out_features, groups]), device)),
            host: None,
            format,
            in_features,
            out_features,
        }
    }

    /// Dense `[in, out]` weight on the device.
    fn dequantize(&self) -> Tensor<B, 2> {
        let f = self.format;
        let (inf, outf) = (self.in_features, self.out_features);
        if let Some(host) = &self.host {
            let device = self.scales.val().device();
            return Tensor::from_data(TensorData::new(host.dequantize(), [inf, outf]), &device);
        }
        let packed = self.packed.val(); // [out, words]
        let words = inf / f.per_word();
        let parts: Vec<Tensor<B, 3, Int>> = (0..f.per_word())
            .map(|k| {
                packed
                    .clone()
                    .bitwise_right_shift_scalar(((k * f.bits()) as i32).elem())
                    .bitwise_and_scalar((f.mask() as i32).elem())
                    .reshape([outf, words, 1])
            })
            .collect();
        let q = Tensor::cat(parts, 2).reshape([outf, inf]).float().sub_scalar(f.offset() as f32);
        let groups = self.scales.val().dims()[1];
        let w = q.reshape([outf, groups, inf / groups]) * self.scales.val().reshape([outf, groups, 1]);
        w.reshape([outf, inf]).transpose()
    }

    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dims = x.dims();
        let rows = dims[..D - 1].iter().product::<usize>();
        match &self.host {
            // Decode (few rows): quantized kernel straight from packed data.
            Some(host) if rows <= 8 => {
                let device = x.device();
                let xs: Vec<f32> = x.into_data().convert::<f32>().to_vec().expect("f32 input");
                let y = host.matvec(&xs, rows);
                let mut out_dims = dims;
                out_dims[D - 1] = self.out_features;
                Tensor::<B, 1>::from_data(TensorData::new(y, [rows * self.out_features]), &device)
                    .reshape(out_dims)
            }
            _ => linear(x, self.dequantize(), None),
        }
    }
}

/// `y = x · W` with `W: [in, out]` (Burn's `Linear` layout, no bias).
#[derive(Module, Debug)]
pub struct Proj<B: Backend> {
    /// Dense weight `[in, out]`. Never materialised in a quantized model.
    pub weight: Param<Tensor<B, 2>>,
    pub quant: Option<QuantParams<B>>,
}

impl<B: Backend> Proj<B> {
    /// Dense projection with Burn's default `Linear` initialisation.
    pub fn new(in_features: usize, out_features: usize, device: &B::Device) -> Self {
        let linear = LinearConfig::new(in_features, out_features).with_bias(false).init(device);
        Self { weight: linear.weight, quant: None }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        match &self.quant {
            Some(q) => q.forward(x),
            None => linear(x, self.weight.val(), None),
        }
    }

    /// Replace the weights with quantized ones (the dense weight is left
    /// uninitialised, so it takes no memory if it was never loaded).
    pub fn set_quantized(&mut self, q: HostQuant, device: &B::Device) {
        self.quant = Some(QuantParams::from_host(q, device));
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;
    use crate::backend::InferBackend as B;

    fn dense(inf: usize, outf: usize) -> (Proj<B>, Vec<f32>) {
        let device = Default::default();
        let p = Proj::<B>::new(inf, outf, &device);
        let w: Vec<f32> = p.weight.val().into_data().to_vec().unwrap();
        (p, w)
    }

    #[test]
    fn quantize_roundtrip_error_is_bounded() {
        let (_, w) = dense(64, 24);
        for format in [QuantFormat::Q4, QuantFormat::Q8] {
            let q = HostQuant::quantize(format, &w, 64, 24);
            let back = q.dequantize();
            let absmax = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let err = w.iter().zip(&back).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            // half a quantization step of the largest group
            assert!(err <= absmax / format.max_level() * 0.5 + 1e-6, "{format:?}: {err}");
        }
    }

    #[test]
    fn quantized_forward_matches_dequantized_dense() {
        let device = Default::default();
        let (p, w) = dense(64, 24);
        for format in [QuantFormat::Q4, QuantFormat::Q8] {
            let q = HostQuant::quantize(format, &w, 64, 24);
            let deq = q.dequantize();
            let mut qp = Proj::<B>::new(64, 24, &device);
            qp.set_quantized(q.clone(), &device);
            let mut on_device = Proj::<B>::new(64, 24, &device);
            on_device.quant = Some(QuantParams::from_host_with(q, &device, false));
            let mut reference = Proj::<B>::new(64, 24, &device);
            reference.weight = Param::from_tensor(Tensor::from_data(TensorData::new(deq, [64, 24]), &device));
            for rows in [1, 3, 20] {
                // decode kernel for few rows, dequantize + matmul for many
                let x = Tensor::<B, 3>::random([1, rows, 64], Distribution::Normal(0.0, 1.0), &device);
                let a: Vec<f32> = qp.forward(x.clone()).into_data().to_vec().unwrap();
                let d: Vec<f32> = on_device.forward(x.clone()).into_data().to_vec().unwrap();
                let b: Vec<f32> = reference.forward(x).into_data().to_vec().unwrap();
                // The host decode kernel also quantizes x to int8 (as llama.cpp
                // does): allow ~1% of the output scale there.
                let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                for ((u, w), v) in a.iter().zip(&d).zip(&b) {
                    assert!((u - v).abs() < 0.01 * scale + 1e-4, "{format:?} rows={rows} host: {u} vs {v}");
                    assert!((w - v).abs() < 1e-3, "{format:?} rows={rows} device: {w} vs {v}");
                }
            }
        }
        let _ = p;
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// `cargo test --release -p quark-core --lib proj::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn decode_kernel_throughput() {
        let (inf, outf) = (3072, 8192);
        let w: Vec<f32> = (0..inf * outf).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
        let x: Vec<f32> = (0..inf).map(|i| (i % 13) as f32 / 13.0).collect();
        for format in [QuantFormat::Q4, QuantFormat::Q8] {
            let q = HostQuant::quantize(format, &w, inf, outf);
            let _ = q.matvec(&x, 1);
            let t = std::time::Instant::now();
            let reps = 20;
            for _ in 0..reps {
                std::hint::black_box(q.matvec(&x, 1));
            }
            let secs = t.elapsed().as_secs_f64() / reps as f64;
            println!(
                "{format:?} {inf}x{outf}: {:.2} ms/matvec, {:.1} GB/s of weights, {:.1} Gparam/s",
                secs * 1e3,
                q.bytes() as f64 / secs / 1e9,
                (inf * outf) as f64 / secs / 1e9
            );
        }
    }
}
