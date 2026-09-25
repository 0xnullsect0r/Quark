//! Global L2 gradient-norm clipping for Burn gradients.

use burn::{
    module::{AutodiffModule, ModuleVisitor, ParamId},
    optim::GradientsParams,
    tensor::{ElementConversion, Tensor, backend::AutodiffBackend},
};

/// Sums the squared L2 norm of every parameter gradient.
struct GradSqNorm<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    sq_sum: Option<Tensor<B::InnerBackend, 1>>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradSqNorm<'_, B> {
    fn visit_float<const D: usize>(&mut self, id: ParamId, _tensor: &Tensor<B, D>) {
        if let Some(g) = self.grads.get::<B::InnerBackend, D>(id) {
            let sq = g.powf_scalar(2.0).sum();
            self.sq_sum = Some(match self.sq_sum.take() {
                Some(acc) => acc + sq,
                None => sq,
            });
        }
    }
}

/// Multiplies every parameter gradient by `scale`.
struct GradScale<'a> {
    grads: &'a mut GradientsParams,
    scale: f32,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradScale<'_> {
    fn visit_float<const D: usize>(&mut self, id: ParamId, _tensor: &Tensor<B, D>) {
        if let Some(g) = self.grads.remove::<B::InnerBackend, D>(id) {
            self.grads.register::<B::InnerBackend, D>(id, g * self.scale);
        }
    }
}

/// Clip gradients to a global L2 norm of `max_norm` (if `max_norm > 0`) and
/// return the norm before clipping.
pub fn clip_grad_norm<B: AutodiffBackend, M: AutodiffModule<B>>(
    model: &M,
    grads: &mut GradientsParams,
    max_norm: f32,
) -> f32 {
    let mut norm_visitor = GradSqNorm::<B> { grads, sq_sum: None };
    model.visit(&mut norm_visitor);
    let norm = norm_visitor
        .sq_sum
        .map(|sq| sq.into_scalar().elem::<f32>().sqrt())
        .unwrap_or(0.0);

    if max_norm > 0.0 && norm > max_norm {
        let mut scale_visitor = GradScale { grads, scale: max_norm / (norm + 1e-6) };
        model.visit(&mut scale_visitor);
    }
    norm
}
