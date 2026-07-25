# Safe Clinical Context + DICOM FiLM Segmentation

## Input contract

- Vision input: three-window 3D CT patch.
- Clinical text: `extracted_cc` and `chief_complaint` only.
- Prohibited text: report body/conclusion, radiology history, raw/refined EMR,
  `class`, and `subclass`.
- DICOM metadata is never converted to text.
- CFG and LLM LoRA are disabled in the primary experiments.

`audit_dataset_contract.py` reports missing fields, within-split duplicates and
train/valid overlap. The loader deterministically removes duplicate rows and
removes validation case IDs from the training frame. DICOM normalization and
category vocabularies are fit after that filtering on training rows only.

## DICOM acquisition path

Numeric fields are KVP, PixelSpacing X/Y, SliceThickness and XRayTubeCurrent.
Each has a missingness bit, giving 10 numeric inputs. Manufacturer and
ConvolutionKernel use train-fitted categorical embeddings with explicit
`<UNK>` and `<MISSING>` IDs.

The metadata encoder conditions the bottleneck and the first three decoder
stages:

`F' = F + alpha_d * (delta_gamma(e_d) * Norm(F) + delta_beta(e_d))`

The gamma/beta projection is zero initialized, so enabling DICOM FiLM is an
exact identity at initialization. `alpha_d` starts at 0.01 and its value and
residual RMS are logged at every scale.

## Clinical text path

The frozen LLM is compressed into structured concepts:

- hemorrhage burden (presence and severity combined)
- trauma
- neurologic symptom
- anatomy
- laterality
- uncertainty
- two open-ended context queries

Text fusion remains a confidence-controlled residual in the first three
decoder stages. A burden head uses a mask-derived continuous target whose zero
means no hemorrhage and whose positive magnitude reflects lesion volume. A
trauma head uses a keyword-derived target from the two allowed text fields.

Correct text compatibility is contrasted with a shuffled case from the local
batch or another distributed rank. This supports batch size one per GPU
without an extra LLM forward.

## Anti-collapse and safety path

During the initial text grounding epochs, residual strength is forced on while
the residual projection still begins near zero. Coarse vision patches are
randomly hidden only in the text-conditioned pass. The unmasked vision pass is
trained simultaneously.

One LLM forward produces the text concepts. The model returns:

- `vision_logits`: DICOM-conditioned vision path with text bypassed
- `raw_fused_logits`: provisional text-conditioned result
- `fused_logits`: uncertainty-safe result

For binary foreground probability `p_v`, uncertainty is
`u = 4 * p_v * (1 - p_v)`, and:

`fused_logits = vision_logits + u * (raw_fused_logits - vision_logits)`

Thus text can act most strongly where vision is uncertain, while confident
vision regions remain protected. Training also applies vision segmentation,
confident-region consistency, residual magnitude, burden, trauma, and
compatibility-margin losses.

## Recommended experiment order

1. `vision_only`
2. `dicom_film`
3. `text_safe`
4. `dicom_text_safe`
5. Frozen-vision conditioning phase from a vision checkpoint
6. Limited joint fine-tuning from the best conditioning checkpoint

The four core ablations are configured in
`run_train_dicom_ablation_8gpu.sh`. Frozen-vision extras require `PRETRAINED`
and are enabled with `RUN_EXTRA_MODES=1`.
