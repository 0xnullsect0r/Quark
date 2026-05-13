/// Compute the global L2 norm of all gradients and scale them down if it
/// exceeds `max_norm` (global gradient norm clipping).
///
/// # Returns
/// The *pre-clipping* global L2 norm.
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    let total_norm = grads.iter().map(|g| g * g).sum::<f32>().sqrt();

    if total_norm > max_norm && total_norm > 0.0 {
        let scale = max_norm / total_norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
    total_norm
}
