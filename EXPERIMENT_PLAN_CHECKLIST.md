# Safe Fusion 실험 계획표

## 실행 파일

- [x] `audit_dataset_contract.py`
  - train/valid 중복, 누수, safe text 및 DICOM 컬럼을 점검한다.
- [x] `check_image_paths.py`
  - 원본 경로 rewrite 후 실제 image 존재 여부를 점검한다.
- [x] `run_train_safe_fusion_2gpu.sh`
  - GPU 두 장에서 선택한 실험을 우선순위 순으로 하나씩 실행한다.
- [x] `run_train_dicom_ablation_8gpu.sh`
  - GPU 두 장씩 네 pair를 사용해 core ablation 네 개를 병렬 실행한다.
- [ ] `inference_eval.py`
  - 각 실험 checkpoint에 대한 전체 validation inference를 실행한다.
- [ ] metric/benefit-case 분석 실행 파일
  - vision 대비 개선·악화 case, FP voxel, volume ratio를 계산한다.

## 공통 실험 조건

- [x] Text backbone: frozen `Llama-2-7b-chat-hf`
- [x] Text fields: `extracted_cc`, `chief_complaint`
- [x] DICOM: numeric/categorical encoder + residual FiLM
- [x] 판독문, raw/refined EMR, class/subclass, demographics 사용 금지
- [x] CFG off
- [x] LoRA off
- [ ] 모든 비교 실험이 동일한 초기 vision checkpoint 또는 동일한 초기화 정책을 사용했는지 확인
- [ ] train/valid data-contract report 보관
- [ ] W&B run name과 checkpoint directory가 서로 겹치지 않는지 확인

## 우선순위별 실행 계획

### P0 — 2-GPU smoke test

- [ ] `dicom_text_safe` 1 epoch, train 2 iteration, validation 1 iteration 실행
- [ ] Llama weight shard preflight 통과
- [ ] train/valid image-path preflight 통과
- [ ] 첫 batch forward/backward 통과
- [ ] 두 GPU 모두 process와 VRAM 할당 확인
- [ ] NaN, CUDA illegal memory access, DDP unused parameter 오류 없음
- [ ] checkpoint와 `dicom_schema.json` 생성 확인

권장 명령:

```bash
GPU_PAIR=2,3 \
EXPERIMENTS_2GPU=dicom_text_safe \
SMOKE_TEST=1 \
bash run_train_safe_fusion_2gpu.sh
```

### P1 — Vision-only 기준선

- [ ] `vision_only` 학습 완료
- [ ] validation Dice, sensitivity, FP voxel 기록
- [ ] normal/hemo 및 병변 크기별 성능 기록
- [ ] 이후 모든 context 결과의 기준 checkpoint와 prediction을 고정

```bash
GPU_PAIR=0,1 \
EXPERIMENTS_2GPU=vision_only \
bash run_train_safe_fusion_2gpu.sh
```

### P2 — DICOM FiLM 단독 효과

- [ ] `dicom_film` 학습 완료
- [ ] FiLM residual RMS가 초기에는 0에 가깝고 점진적으로 증가하는지 확인
- [ ] manufacturer/kernel/KVP별 성능 편향 확인
- [ ] vision-only 대비 전체 Dice와 FP 변화 확인
- [ ] DICOM shortcut 가능성 점검

```bash
GPU_PAIR=0,1 \
EXPERIMENTS_2GPU=dicom_film \
bash run_train_safe_fusion_2gpu.sh
```

### P3 — Safe text 단독 효과

- [ ] `text_safe` 학습 완료
- [ ] text burden, trauma, compatibility loss 수렴 확인
- [ ] context confidence가 전부 0 또는 1로 붕괴하지 않는지 확인
- [ ] null/shuffled text가 vision 출력으로 회귀하는지 확인
- [ ] vision 대비 +0.01/+0.1 개선 case 수 기록
- [ ] FP 증가 및 크게 악화된 case 수 기록

```bash
GPU_PAIR=0,1 \
EXPERIMENTS_2GPU=text_safe \
bash run_train_safe_fusion_2gpu.sh
```

### P4 — DICOM + safe text 결합

- [ ] `dicom_text_safe` 학습 완료
- [ ] P2와 P3의 이득이 결합되는지 확인
- [ ] acquisition residual과 clinical residual의 규모를 각각 확인
- [ ] vision confidence가 높은 영역의 mask가 보호되는지 확인
- [ ] 기존 context-benefit case에서 선택적 개선이 유지되는지 확인
- [ ] core 네 실험 중 최종 후보 선정

```bash
GPU_PAIR=0,1 \
EXPERIMENTS_2GPU=dicom_text_safe \
bash run_train_safe_fusion_2gpu.sh
```

### P5 — Frozen-vision conditioning

- [ ] P1의 vision checkpoint를 `PRETRAINED`로 지정
- [ ] `dicom_film_frozen` 실행
- [ ] `dicom_text_safe_frozen` 실행
- [ ] conditioning branch만 학습되는지 trainable parameter 수 확인
- [ ] frozen 결과가 joint 결과보다 안정적인지 비교

```bash
GPU_PAIR=0,1 \
RUN_EXTRA_MODES=1 \
EXPERIMENTS_2GPU=dicom_film_frozen,dicom_text_safe_frozen \
PRETRAINED=/path/to/vision_checkpoint.pth \
bash run_train_safe_fusion_2gpu.sh
```

## 한 번에 순차 실행

아래 명령은 동일한 GPU pair에서 core 네 실험을 표의 순서대로 실행한다.
각 실험이 끝난 뒤 다음 실험을 시작한다.

```bash
GPU_PAIR=0,1 bash run_train_safe_fusion_2gpu.sh
```

특정 실험만 선택할 때:

```bash
GPU_PAIR=2,3 \
EXPERIMENTS_2GPU=dicom_film,dicom_text_safe \
bash run_train_safe_fusion_2gpu.sh
```

## 최종 선택 기준

- [ ] 전체 Dice가 vision-only보다 개선
- [ ] 작은 병변 Dice=0 비율이 감소
- [ ] normal case FP voxel이 허용 범위 이내
- [ ] predicted-volume ratio 과대 증가 없음
- [ ] null/shuffled/corrupted context에서 vision fallback 유지
- [ ] 기존 context-benefit 405 cases의 개선 수가 증가
- [ ] 크게 악화된 case 수가 감소
- [ ] scanner/vendor별 성능 격차가 증가하지 않음
- [ ] 학습 재현성과 inference runtime이 허용 범위 이내
