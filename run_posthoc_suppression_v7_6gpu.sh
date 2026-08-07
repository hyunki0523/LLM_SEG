#!/bin/bash
set -euo pipefail

# Six-rank V7 inference wrapper.  This runs one frozen Vision checkpoint and
# partitions validation cases across all six GPUs; it does not launch six
# independent experiments.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_posthoc_suppression_v7_2gpu.sh}"
GPU_IDS="${GPU_IDS:-}"

if [ -z "$GPU_IDS" ]; then
    echo "[ERROR] Explicitly set six GPU IDs, e.g. GPU_IDS='0 1 2 3 4 5'"
    exit 2
fi
gpu_tokens="${GPU_IDS//,/ }"
read -r -a gpu_list <<< "$gpu_tokens"
if [ "${#gpu_list[@]}" -ne 6 ]; then
    echo "[ERROR] GPU_IDS must contain exactly six GPU IDs."
    exit 2
fi
for gpu in "${gpu_list[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid GPU ID: $gpu"
        exit 2
    fi
done
if [ "$(printf '%s\n' "${gpu_list[@]}" | sort -u | wc -l)" -ne 6 ]; then
    echo "[ERROR] GPU_IDS must contain six distinct devices."
    exit 2
fi
if [ ! -f "$BASE_LAUNCHER" ]; then
    echo "[ERROR] Base V7 launcher not found: $BASE_LAUNCHER"
    exit 2
fi

selected_devices="$(IFS=,; echo "${gpu_list[*]}")"
echo "[V7-6GPU] One frozen checkpoint, six inference ranks: $selected_devices"

GPU_PAIR="$selected_devices" \
NUM_PROCESSES=6 \
bash "$BASE_LAUNCHER"
