use burn::{
    nn::loss::CrossEntropyLossConfig,
    tensor::{Int, Tensor, backend::AutodiffBackend},
};

/// Cross-entropy loss over a batch of token predictions.
///
/// `logits` has shape `[batch, seq, vocab]`.
/// `labels` has shape `[batch, seq]`.
/// `pad_id` positions are masked out (do not contribute to loss).
pub fn cross_entropy_loss<B: AutodiffBackend>(
    logits: Tensor<B, 3>,
    labels: Tensor<B, 2, Int>,
    pad_id: u32,
    device: &B::Device,
) -> Tensor<B, 1> {
    let [batch, seq, vocab] = logits.dims();
    let logits_2d = logits.reshape([batch * seq, vocab]);
    let labels_1d = labels.reshape([batch * seq]);

    CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![pad_id as usize]))
        .init(device)
        .forward(logits_2d, labels_1d)
}
