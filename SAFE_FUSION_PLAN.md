# Safe Fusion 개발 계획

## 확정된 입력

- Text: `extracted_cc`, `chief_complaint`
- DICOM numeric: `KVP`, `PixelSpacing` X/Y, `SliceThickness`,
  `XRayTubeCurrent`와 각 값의 missing indicator
- DICOM categorical: `Manufacturer`, `ConvolutionKernel`
- 사용 금지: 판독문 본문/결론/history, raw/refined EMR, class/subclass,
  demographics

## 구현 순서와 현재 상태

### Phase 0 — 데이터 계약 및 누수 방지 (완료)

- 독립 실행 가능한 `audit_dataset_contract.py`
- split 내부 case ID 중복 제거
- validation case ID와 겹치는 train 행 제거
- DICOM 정규화와 category vocabulary를 train split에서만 fit
- validation의 미등록 category는 `<UNK>`, 결측은 `<MISSING>`

현재 workbook 감사 결과:

- train: 31,748 rows / 31,721 unique case IDs / duplicate rows 27
- valid: 7,938 rows / duplicate rows 0
- train-valid overlap: 9 case IDs
- train XRayTubeCurrent missing: 10
- extracted_cc missing: train 18, valid 1

### Stage 1 — DICOM FiLM (완료)

- numeric MLP + categorical embeddings
- bottleneck 및 decoder semantic 3 scales에 residual FiLM
- gamma/beta projection zero initialization
- scale별 alpha 및 residual RMS logging
- conditioning-only 학습을 위한 `--freeze_vision`

기대 효과:

- scanner/protocol 차이에 대한 feature calibration
- DICOM 문자열 shortcut과 LLM token 낭비 제거
- 시작 시 vision checkpoint를 정확히 보존

### Stage 2 — 안전한 clinical concept (완료)

- frozen LLM, CFG off, primary experiment LoRA off
- burden(presence+severity), trauma, neurologic symptom, anatomy,
  laterality, uncertainty + open query 2개
- mask-derived hemorrhage burden auxiliary target
- 두 허용 text column에서만 계산한 trauma auxiliary target
- batch 또는 다른 DDP rank의 case를 이용한 shuffled compatibility

기대 효과:

- case-level prior를 명시적 concept으로 제한
- presence와 severity의 중복 head 제거
- batch size 1에서도 compatibility negative 확보

### Stage 3 — anti-collapse 및 safe residual (완료)

- unmasked `vision_logits`
- masked, text-conditioned `raw_fused_logits`
- vision uncertainty로 제한한 `fused_logits`
- 초기 epoch full-strength grounding
- text pass에만 coarse vision patch masking
- confident-region/null consistency, residual penalty,
  compatibility margin, burden/trauma loss
- evaluation compatibility hard bypass

기대 효과:

- confidence를 0으로 보내는 modality collapse 억제
- vision이 확신하는 boundary/normal region 보호
- text 오류 시 vision-only로 case-level fallback

### Stage 4 — 실험 (실행 대기)

Core:

1. `vision_only`
2. `dicom_film`
3. `text_safe`
4. `dicom_text_safe`

Extra:

5. `dicom_film_frozen`
6. `dicom_text_safe_frozen`

Frozen 실험은 vision checkpoint를 `PRETRAINED`로 지정해야 한다. Core 네
실험은 `run_train_dicom_ablation_8gpu.sh`에서 4개 GPU pair에 배치된다.

## 평가 기준

- 전체/hemo/normal Dice와 sensitivity
- 작은 병변 Dice=0 비율
- FP voxels와 predicted-volume ratio
- vision 대비 Dice +0.01/+0.1 및 -0.01/-0.1 case 수
- null/shuffled text에서 vision 회귀 여부
- DICOM FiLM residual RMS 및 text compatibility 분포
- 기존 context-benefit 405 cases의 개선 유지 여부
