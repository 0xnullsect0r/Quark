use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

/// Default LoRA scaling: α / rank.
const LORA_ALPHA_DEFAULT: f32 = 16.0;

/// A linear layer augmented with Low-Rank Adaptation (LoRA) adapters.
///
/// During the forward pass the output is:
/// ```text
/// base(x)  +  scale * lora_b(lora_a(x))
/// ```
/// where `scale = alpha / rank`.  The base weights should be frozen during
/// fine-tuning so only `lora_a` and `lora_b` receive gradient updates.
#[derive(Module, Debug)]
pub struct LoraLinear<B: Backend> {
    base: Linear<B>,
    lora_a: Linear<B>,
    lora_b: Linear<B>,
    #[module(skip)]
    scale: f32,
    #[module(skip)]
    freeze_base: bool,
}

impl<B: Backend> LoraLinear<B> {
    /// Create a new [`LoraLinear`] layer.
    ///
    /// * `in_features`  – input dimensionality.
    /// * `out_features` – output dimensionality.
    /// * `rank`         – LoRA rank *r* (typical values: 4, 8, 16).
    /// * `device`       – target device for parameter initialisation.
    pub fn new(in_features: usize, out_features: usize, rank: usize, device: &B::Device) -> Self {
        let scale = LORA_ALPHA_DEFAULT / rank as f32;
        Self {
            base: LinearConfig::new(in_features, out_features)
                .with_bias(false)
                .init(device),
            lora_a: LinearConfig::new(in_features, rank)
                .with_bias(false)
                .init(device),
            lora_b: LinearConfig::new(rank, out_features)
                .with_bias(false)
                .init(device),
            scale,
            freeze_base: true,
        }
    }

    /// Forward pass for 2-D tensors `[batch, features]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let base_out = self.base.forward(x.clone());
        let lora_out = self.lora_b.forward(self.lora_a.forward(x));
        base_out + lora_out.mul_scalar(self.scale)
    }

    /// Forward pass for 3-D tensors `[batch, seq, features]`.
    ///
    /// Reshapes to 2-D internally, applies [`Self::forward`], then restores
    /// the original batch and sequence dimensions.
    pub fn forward_3d(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, features] = x.dims();
        let x_2d = x.reshape([batch * seq, features]);
        let out_2d = self.forward(x_2d);
        let out_features = out_2d.dims()[1];
        out_2d.reshape([batch, seq, out_features])
    }
}

/// Configuration for LoRA fine-tuning.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoraConfig {
    /// LoRA rank *r* — the bottleneck dimension of the adapter matrices.
    pub rank: usize,
    /// Scaling factor α; the effective scale is `alpha / rank`.
    pub alpha: f32,
    /// Names of attention projection modules to apply LoRA to
    /// (e.g. `["q_proj", "v_proj"]`).
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
        }
    }
}
