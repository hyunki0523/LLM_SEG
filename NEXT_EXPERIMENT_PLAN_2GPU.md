# v3a 다음 실험 계획 — 2 GPU

## 실행 구조

두 GPU를 하나의 DDP 학습에 묶지 않는다. GPU마다 single-process 실험을
하나씩 실행하고, 네 비교 실험을 두 번의 wave로 나눈다.

| Wave | GPU 0 | GPU 1 |
|---|---|---|
| 1 | vision-only control | learned soft prompt + online Llama |
| 2 | no soft prompt + online Llama | no soft prompt + cached Llama |

이 구성은 online Llama가 2-GPU DDP의 `accelerator.prepare`에서 멈췄던
경로를 피한다. 각 실험은 `batch_size=2`, `grad_accum=16`, world size 1을
사용하여 기존 effective batch 32를 유지한다.

## P0 — 이전 실행 및 환경 정리

- [ ] 기존 2-GPU DDP `accelerate/train.py` 프로세스 종료
- [ ] `nvidia-smi`에서 선택할 두 GPU가 비어 있는지 확인
- [ ] `EXPECTED_GPUS=2 bash install_requirements_cu132.sh` 통과
- [ ] PyTorch 2.12.1, CUDA runtime 13.0 이상, Blackwell BF16 확인
- [ ] cache 파일과 train/valid prompt 동치성 확인

CUDA 13.0 runtime은 조건을 만족하므로 cu132로 다시 설치할 필요가 없다.

```bash
cd /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk
EXPECTED_GPUS=2 bash install_requirements_cu132.sh
```

## P1 — 2-GPU smoke test

- [ ] Wave 1 두 실험 모두 `accelerator.prepare` 완료
- [ ] Wave 2 두 실험 모두 forward/backward 완료
- [ ] cached 실험에서 Llama loading log가 없음
- [ ] 네 실험 모두 `[DONE]`
- [ ] NaN, CUDA illegal address, DDP timeout 없음
- [ ] checkpoint/W&B 이름이 서로 겹치지 않음

```bash
SMOKE_TEST=1 \
GPU_PAIR=0,1 \
bash run_train_v3a_text_ablation_2gpu.sh
```

## P2 — 5-epoch pilot

- [ ] 네 실험의 loss가 finite
- [ ] learned soft prompt gradient/norm이 0이 아님
- [ ] no-soft online/cache validation 곡선이 유사
- [ ] vision 대비 context FP voxel과 predicted volume 확인
- [ ] normal/hemo 및 작은 병변 성능 분리

```bash
EPOCHS=5 \
N_ITER_PER_EPOCH=64 \
N_ITER_VALID=10 \
AUTO_RESUME=0 \
OVERWRITE_TRAIN=1 \
GPU_PAIR=0,1 \
bash run_train_v3a_text_ablation_2gpu.sh
```

중단 기준:

- online/cache no-soft Dice 차이가 반복적으로 0.01 초과
- context 실험에서 FP 또는 predicted volume 급증
- soft prompt gradient가 계속 0
- CUDA/NaN 오류 재발

## P3 — 본 학습

P0~P2 통과 후 실행한다.

```bash
EPOCHS=300 \
N_ITER_PER_EPOCH=256 \
N_ITER_VALID=50 \
AUTO_RESUME=1 \
OVERWRITE_TRAIN=0 \
GPU_PAIR=0,1 \
bash run_train_v3a_text_ablation_2gpu.sh
```

최종 비교 지표:

- 전체 및 hemorrhage/normal validation Dice
- sensitivity와 작은 병변 Dice=0 비율
- FP voxel과 predicted-volume ratio
- vision 대비 `+0.01`, `+0.1`, `-0.1` case 수
- context-benefit case에서의 성능
- peak VRAM, step time, cache I/O 시간
- learned soft prompt와 no-soft cached의 효과 크기

## 다음 구조 변경 조건

v3a 결과에서 context 이득 또는 learned soft prompt 이득이 재현된 뒤에만
v3b correlation/spatial grounding을 추가한다. DICOM FiLM, deep supervision,
LoRA는 v3b 첫 실험과 동시에 변경하지 않는다.
