# V7 Frozen Post-Hoc Suppressor

V7 tests one narrow hypothesis: safe chief-complaint context can suppress
borderline vision false positives in cases that are likely normal. It cannot
create a lesion, recover a vision false negative, or update the vision model.

## Equation

```text
Z_final = stop_grad(Z_vision) - beta * s * w(stop_grad(P_vision))
```

`w` is zero below `p_min` and at or above `p_protect`. Between the two limits it
decreases linearly from the background side toward the protected positive side.
The initial sweep uses `p_min=0.2`, `p_protect=0.85`, and
`beta=0.25,0.5,1,2,3`.

The default `TEXT_THRESHOLDS=0` gives `s=P(normal)`. A later conservative sweep
can use, for example, `TEXT_THRESHOLDS=0.8,0.9,0.95`; then
`s=clip((P(normal)-threshold)/(1-threshold), 0, 1)`.

## Data contract

- Text inputs: `extracted_cc` and `chief_complaint` only.
- LLM input contains no report, refined EMR, label, demographic, or DICOM field.
- Normal classifier target: a readable GT hemorrhage mask that is exactly empty.
- Cases without a readable GT mask are scored but excluded from classifier and
  segmentation metrics.
- The LLM is frozen. Only a calibrated linear classifier is fitted on its cached
  final `<SEG>` hidden state.
- TF-IDF is trained on the same prompts and targets.

Every deterministic CC-only cache must include the explicit empty control
`<SEG>`. If an already completed cache predates this addition, running the cache
command again appends only the missing control entry:

```bash
python precompute_text_features.py \
  --csv /mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx \
  --csv /mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx \
  --llm-repo /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf \
  --output /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk/text_feature_cache/llama2_safe_cc_nosoft_deterministic.sqlite3 \
  --dicom-prompt-mode none --device cuda:0
```

## Full two-GPU run

The vision checkpoint is inferred once. FP16 probability volumes for labeled
validation cases are then reused for every CPU suppression condition; unlabeled
cases do not consume probability-map storage.

```bash
cd /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk

GPU_PAIR="0,1" \
VISION_CHECKPOINT="/absolute/path/to/frozen/vision_checkpoint.pth" \
TEXT_FEATURE_CACHE="/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk/text_feature_cache/llama2_safe_cc_nosoft_deterministic.sqlite3" \
USE_EMA=0 \
bash run_posthoc_suppression_v7_2gpu.sh
```

## Six-GPU five-condition experiment

The six-GPU launcher runs two frozen base models concurrently:

- GPU 0,1,2: Vision-only
- GPU 3,4,5: true DICOM-FiLM-only (`context=False`, `use_dicom=True`)

The post-hoc matrix then evaluates exactly these groups:

1. Vision-Only (Baseline)
2. Vision + DICOM FiLM
3. V7: DICOM + Empty CC
4. V7: DICOM + Shuffled CC
5. V7: DICOM + Real CC (Proposed)

All three V7 conditions use the same frozen DICOM-FiLM probability volumes.
Only the case-level suppression score changes, so the CC ablation is paired and
does not confound the image/DICOM base model.

```bash
cd /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk

GPU_IDS="0 1 2 3 4 5" \
VISION_CHECKPOINT="/absolute/path/to/frozen/vision_checkpoint.pth" \
DICOM_FILM_CHECKPOINT="/absolute/path/to/true_dicom_film_only_checkpoint.pth" \
TEXT_FEATURE_CACHE="/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk/text_feature_cache/llama2_safe_cc_nosoft_deterministic.sqlite3" \
VISION_USE_EMA=0 \
DICOM_USE_EMA=0 \
bash run_posthoc_suppression_v7_6gpu.sh
```

The launcher verifies each checkpoint's sibling `args.json`. It rejects a
context-enabled checkpoint as the DICOM-FiLM baseline. In particular, the old
Wave1 job named `dicom_film` actually launched `dicom_text_safe` and must not be
used for condition 2. It also requires the Vision and DICOM-FiLM checkpoints to
match on dataset paths, epoch/iteration schedule, batch/accumulation, learning
rate, loss, deep supervision, and EMA settings.

Training checkpoints produced by the current `train.py` save EMA module weights
under the ordinary `model` state dictionary, so inference should normally use
`USE_EMA=0` (or `VISION_USE_EMA=0`, `DICOM_USE_EMA=0`).

To reuse completed classifier scores and probability extraction:

```bash
RESULT_ROOT="/existing/v7/result" \
SKIP_CLASSIFIER=1 \
SKIP_VISION_INFERENCE=1 \
TEXT_THRESHOLDS="0,0.8,0.9,0.95" \
bash run_posthoc_suppression_v7_2gpu.sh
```

## Outputs

- `normal_classifier/normal_classifier_report.json`: classification-only AUROC,
  average precision, calibration, and shuffled/empty controls.
- `normal_classifier/valid_normal_scores.csv`: per-case normal confidence.
- `vision_probability/annotation.csv`: probability, image, and GT paths.
- `suppression_sweep/suppression_sweep_summary.csv`: isolated positive and normal
  cohort metrics with the one-percentage-point sensitivity safety flag.
- `suppression_sweep/per_case_suppression_metrics.csv`: paired case-level results.

No condition is accepted unless positive-cohort sensitivity decreases by no more
than `0.01` in absolute value relative to the same frozen vision probabilities.
