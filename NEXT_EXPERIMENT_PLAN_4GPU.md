# v3a 다음 실험 계획 — 4 GPU

## 현재 판단

2026-07-31의 `soft_prompt_online` 2-GPU smoke run은 두 rank 모두
`accelerator.prepare` 진입 후 2시간 이상 진행되지 않았다. frozen Llama
7B 전체가 DDP model에 포함되어 초기 parameter broadcast가 발생하는
경로이므로, GPU 수를 늘린 4-way DDP로 반복하지 않는다.

RTX PRO 6000 Blackwell은 GPU당 약 98 GB이므로 v3a에서는 한 GPU에 한
실험을 배치한다. 네 실험은 동시에 실행하되 각 실험은 single-process로
유지한다.

| GPU | 실험 | 검증할 질문 |
|---|---|---|
| 0 | `vision_control` | 같은 환경의 vision-only 기준 성능은 얼마인가? |
| 1 | `soft_prompt_online` | 학습 가능한 16-token prefix가 도움이 되는가? |
| 2 | `no_soft_online` | soft prompt 제거 자체의 효과는 무엇인가? |
| 3 | `no_soft_cached` | frozen Llama cache가 online 경로를 재현하는가? |

single-GPU 본 학습 기본값은 `batch_size=2`, `grad_accum=16`으로 두어
기존 2-GPU의 `2 × 8 × 2 = 32`와 같은 effective batch 32를 유지한다.

## P0 — 환경 고정

- [ ] 기존 학습 프로세스가 남아 있지 않은지 확인
- [ ] Python 3.11 확인
- [ ] PyTorch `2.12.1+cu132` 확인
- [ ] CUDA runtime 13.2 확인
- [ ] GPU 4장과 `sm_120` 확인
- [ ] 사용하지 않는 `torchaudio`/`torchvision`이 설치되어 있지 않은지 확인
- [ ] Transformers/Accelerate/MONAI/W&B 고정 버전 확인
- [ ] 네 GPU의 BF16 matmul 검사 통과

새 컨테이너 또는 환경을 다시 구성할 때:

```bash
cd /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk
EXPECTED_GPUS=4 bash install_requirements_cu132.sh
```

설치 스크립트는 이 저장소에서 사용하지 않으면서 dependency 충돌을
일으키는 `tiatoolbox`, `timm`, `ninja`, `torchaudio`, `torchvision`을
제거한다. 기존 `torch==2.12.1+cu130`은 CUDA 13 조건을 만족하므로
그대로 사용한다. cu132로 정확히 교체해야 할 때만 다음을 사용한다.

```bash
FORCE_TORCH_REINSTALL=1 TORCH_CUDA_INDEX=cu132 \
  EXPECTED_GPUS=4 bash install_requirements_cu132.sh
```

기존 환경을 검사만 할 때:

```bash
EXPECTED_GPUS=4 python verify_runtime_environment.py
```

LoRA는 v3a에서 사용하지 않는다. 이후 별도 실험이 필요할 때만 다음과
같이 설치한다.

```bash
INSTALL_LORA=1 EXPECTED_GPUS=4 bash install_requirements_cu132.sh
```

## P1 — 캐시 동치성

- [ ] cache metadata의 Llama/config/tokenizer/max length 확인
- [ ] train/valid prompt 누락 0건 확인
- [ ] online/cache valid-token hidden state `max_abs=0` 확인
- [ ] cache 파일을 읽을 때 Llama가 로드되지 않는지 확인

```bash
python verify_text_feature_cache.py \
  --csv /mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx \
  --csv /mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx \
  --llm-repo /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf \
  --cache text_feature_cache/llama2_safe_cc_nosoft.sqlite3
```

## P2 — 4-GPU 병렬 smoke test

- [ ] 네 실험 모두 `accelerator.prepare` 완료
- [ ] 첫 forward/backward 완료
- [ ] cached 경로에서 Llama loading log가 없는지 확인
- [ ] NaN/CUDA illegal address/DDP timeout 없음
- [ ] 네 checkpoint 및 W&B run 이름이 서로 다름
- [ ] online Llama single-GPU의 peak VRAM 기록
- [ ] cached 경로의 peak VRAM 및 step time 기록

```bash
SMOKE_TEST=1 GPU_IDS="0 1 2 3" \
  bash run_train_v3a_4gpu_parallel.sh
```

통과 조건은 네 실험 모두 `[DONE]`이며, online Llama의
`accelerator.prepare`가 수 분 이내에 끝나는 것이다.

## P3 — 5-epoch pilot

full training 전에 짧은 곡선으로 구조적 실패를 거른다.

- [ ] 모든 실험을 동일 seed/effective batch로 실행
- [ ] loss와 gradient가 finite
- [ ] learned soft prompt norm/gradient가 0이 아님
- [ ] no-soft online/cache validation curve가 유사
- [ ] vision 대비 context의 FP 증가 여부 확인
- [ ] normal/hemo 및 작은 병변 성능을 분리해 기록

```bash
EPOCHS=5 \
N_ITER_PER_EPOCH=64 \
N_ITER_VALID=10 \
AUTO_RESUME=0 \
OVERWRITE_TRAIN=1 \
GPU_IDS="0 1 2 3" \
bash run_train_v3a_4gpu_parallel.sh
```

중단 기준:

- online/cache no-soft Dice 차이가 반복적으로 0.01을 초과
- context branch의 FP voxel 또는 predicted volume이 급증
- learned soft prompt gradient가 계속 0
- 한 실험에서 NaN 또는 CUDA 오류 재발

## P4 — 본 학습

P0~P3가 모두 통과한 뒤에만 실행한다.

```bash
EPOCHS=300 \
N_ITER_PER_EPOCH=256 \
N_ITER_VALID=50 \
AUTO_RESUME=1 \
OVERWRITE_TRAIN=0 \
GPU_IDS="0 1 2 3" \
bash run_train_v3a_4gpu_parallel.sh
```

최종 비교:

- 전체 validation Dice
- hemorrhage/normal 분리 Dice와 sensitivity
- 작은 병변 Dice=0 비율
- FP voxel과 predicted-volume ratio
- vision 대비 `+0.01`, `+0.1`, `-0.1` case 수
- 기존 context-benefit case에서의 성능
- GPU peak memory, epoch time, cache I/O 시간
- soft prompt online 대 no-soft cached의 효과 크기

## v3b 진입 조건

다음 중 하나를 확인한 뒤에만 correlation/spatial grounding을 추가한다.

1. context가 평균 또는 특정 사전 정의 subgroup에서 반복적으로 이득
2. learned soft prompt가 no-soft보다 의미 있게 우수
3. cache 경로가 online no-soft를 재현하여 이후 구조 실험의 기본 경로로
   사용할 수 있음

v3b에서는 correlation grounding 하나만 추가하고 DICOM FiLM,
deep supervision, LoRA를 동시에 변경하지 않는다.
