# v3a soft-prompt/cache 실험 체크리스트

> 현재 우선 계획은 `NEXT_EXPERIMENT_PLAN_2GPU.md`와
> `run_train_v3a_text_ablation_2gpu.sh`이다. 이 launcher는 2-GPU DDP가
> 아니라 GPU별 single-process 실험을 두 wave로 실행한다.

이번 단계에서는 fusion 구조와 loss를 v2로 고정하고 soft prompt와 Llama
feature cache만 비교합니다. DICOM FiLM, correlation grounding, deep
supervision은 섞지 않습니다.

## P0 — 캐시 생성과 동치성 검증

- [ ] `extracted_cc`, `chief_complaint`만 캐시에 포함되는지 확인
- [ ] soft prompt가 `disabled`로 기록되었는지 확인
- [ ] train/valid의 모든 unique prompt가 캐시에 존재하는지 확인
- [ ] online/cached valid-token hidden state 동치성 검사 통과
- [ ] 누락된 prompt가 있으면 학습이 즉시 실패하는지 확인

```bash
cd /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk

python precompute_text_features.py \
  --csv /mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx \
  --csv /mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx \
  --llm-repo /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf \
  --output text_feature_cache/llama2_safe_cc_nosoft.sqlite3

python verify_text_feature_cache.py \
  --csv /mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx \
  --csv /mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx \
  --llm-repo /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf \
  --cache text_feature_cache/llama2_safe_cc_nosoft.sqlite3
```

## P1 — 세 실험 smoke test

- [ ] `soft_prompt_online`: 기존 learned prefix 경로 forward/backward 통과
- [ ] `no_soft_online`: context length 0 경로 forward/backward 통과
- [ ] `no_soft_cached`: 학습 중 Llama가 DDP/GPU 모델에서 제외되는지 확인
- [ ] 세 실험 모두 checkpoint와 W&B run 이름이 겹치지 않는지 확인
- [ ] NaN, CUDA illegal address, missing cache key가 없는지 확인

```bash
SMOKE_TEST=1 GPU_PAIR=0,1 bash run_train_v3a_text_ablation_2gpu.sh
```

## P2 — 본 학습

- [ ] 세 실험이 동일한 vision initialization과 seed를 쓰는지 확인
- [ ] soft prompt 유무의 validation Dice/FP/volume ratio 비교
- [ ] no-soft online/cached 학습 곡선과 최종 metric이 허용 오차 내인지 확인
- [ ] Llama cache의 epoch time 및 GPU memory 절감량 기록

```bash
GPU_PAIR=0,1 bash run_train_v3a_text_ablation_2gpu.sh
```

개별 실험만 실행할 때:

```bash
V3A_EXPERIMENTS=no_soft_cached GPU_PAIR=0,1 \
  bash run_train_v3a_text_ablation_2gpu.sh
```

## 다음 단계로 넘어가는 조건

- [ ] learned soft prompt가 no-soft보다 일관되게 유리한지 결론 확보
- [ ] online/cached no-soft 결과가 수치적으로 재현됨
- [ ] 위 두 조건을 만족한 뒤에만 spatial correlation grounding을 별도 v3b로 추가
