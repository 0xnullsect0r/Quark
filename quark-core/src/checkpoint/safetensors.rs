#![allow(dead_code, unused_imports, unused_variables)]

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

/// Tensor data with shape and dtype info.
#[derive(Debug, Clone)]
pub struct TensorData {
    pub name: String,
    /// Flat f32 array in row-major order.
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Save a collection of named f32 tensors to a `.safetensors` file.
pub fn save_checkpoint(path: &Path, tensors: &[TensorData]) -> Result<()> {
    // Build (name, raw-bytes, shape) triples so TensorView can borrow the bytes.
    let raw: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
        .iter()
        .map(|t| {
            let bytes: Vec<u8> = bytemuck::cast_slice(&t.data).to_vec();
            (t.name.clone(), bytes, t.shape.clone())
        })
        .collect();

    let views: Vec<(String, TensorView<'_>)> = raw
        .iter()
        .map(|(name, bytes, shape)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .expect("TensorView construction failed");
            (name.clone(), view)
        })
        .collect();

    safetensors::tensor::serialize_to_file(views, &None, path)?;
    Ok(())
}

/// Load named f32 tensors from a `.safetensors` file.
pub fn load_checkpoint(path: &Path) -> Result<Vec<TensorData>> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;

    let mut result = Vec::new();
    for (name, view) in st.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let raw: &[u8] = view.data();
        let data: Vec<f32> = bytemuck::cast_slice(raw).to_vec();
        result.push(TensorData {
            name: name.to_string(),
            data,
            shape,
        });
    }
    Ok(result)
}

/// Save model weights together with optimizer state (Adam m/v buffers).
/// Optimizer tensors are prefixed with `"optimizer."` in the file.
pub fn save_with_optimizer_state(
    path: &Path,
    model_tensors: &[TensorData],
    optimizer_tensors: &[TensorData],
) -> Result<()> {
    let mut all = model_tensors.to_vec();
    for t in optimizer_tensors {
        all.push(TensorData {
            name: format!("optimizer.{}", t.name),
            data: t.data.clone(),
            shape: t.shape.clone(),
        });
    }
    save_checkpoint(path, &all)
}
