use burn::tensor::{activation::log_softmax, backend::Backend, Int, Tensor};

/// Mean cross-entropy over the non-padding targets.
///
/// `logits`: `[n, vocab]`, `targets`: `[n]`. Positions whose target is
/// `pad_id` don't count, neither in the sum nor in the mean. (Burn's
/// `CrossEntropyLoss` with pad tokens divides by *all* positions, which scales
/// the loss, and so the learning rate, down by the padding fraction.)
pub fn masked_cross_entropy<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    pad_id: u32,
) -> Tensor<B, 1> {
    let [n, _] = logits.dims();
    let log_probs = log_softmax(logits, 1).gather(1, targets.clone().reshape([n, 1])).reshape([n]);
    let mask = targets.not_equal_elem(pad_id as i64).float();
    let count = mask.clone().sum().clamp_min(1.0);
    -(log_probs * mask).sum() / count
}

/// [`masked_cross_entropy`] for `[batch, seq, vocab]` logits and `[batch, seq]`
/// labels.
pub fn cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 3>,
    labels: Tensor<B, 2, Int>,
    pad_id: u32,
) -> Tensor<B, 1> {
    let [batch, seq, vocab] = logits.dims();
    masked_cross_entropy(logits.reshape([batch * seq, vocab]), labels.reshape([batch * seq]), pad_id)
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;
    use crate::backend::InferBackend as B;

    #[test]
    fn padding_is_excluded_from_the_mean() {
        let device = Default::default();
        // rows [0, 2], [1, 1], [3, 0] with targets 1, 0, 0
        let logits = Tensor::<B, 2>::from_data(TensorData::new(vec![0.0f32, 2.0, 1.0, 1.0, 3.0, 0.0], [3, 2]), &device);
        let targets = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![1i32, 0, 0], [3]), &device);
        let nll = |a: f32, b: f32, target: f32| (a.exp() + b.exp()).ln() - target;
        let (r0, r1, r2) = (nll(0.0, 2.0, 2.0), nll(1.0, 1.0, 1.0), nll(3.0, 0.0, 3.0));

        // Nothing is padding (pad id 7): plain mean over the three rows.
        let all: f32 = masked_cross_entropy(logits.clone(), targets.clone(), 7).into_scalar();
        assert!((all - (r0 + r1 + r2) / 3.0).abs() < 1e-5, "{all}");
        // Target 0 is padding: only row 0 counts, and it isn't divided by 3.
        let masked: f32 = masked_cross_entropy(logits, targets, 0).into_scalar();
        assert!((masked - r0).abs() < 1e-5, "{masked} vs {r0}");
    }
}
