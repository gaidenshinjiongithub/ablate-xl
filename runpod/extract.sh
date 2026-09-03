#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: bash runpod/extract.sh [profile.env]" >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  # Profiles live in this repository and contain only environment assignments.
  # shellcheck disable=SC1090
  source "$1"
fi

: "${MODEL_ID:?Set MODEL_ID or pass a profile from runpod/profiles}"

PROJECT_DIR="${PROJECT_DIR:-/workspace/ablate-xl}"
HF_HOME="${HF_HOME:-/workspace/huggingface}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-/workspace/ablate-offload}"
OUTPUT_PATH="${OUTPUT_PATH:-/workspace/ablate-output/directions.pt}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EXTRACT_POSITIONS="${EXTRACT_POSITIONS:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
MIN_TOTAL_VRAM_GIB="${MIN_TOTAL_VRAM_GIB:-0}"
MIN_DISK_GIB="${MIN_DISK_GIB:-0}"

export HF_HOME
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "${HF_HOME}" "${OFFLOAD_FOLDER}" "$(dirname "${OUTPUT_PATH}")"

python "${PROJECT_DIR}/runpod/preflight.py" \
  --workspace /workspace \
  --min-total-vram-gib "${MIN_TOTAL_VRAM_GIB}" \
  --min-disk-gib "${MIN_DISK_GIB}"

args=(
  extract
  --model "${MODEL_ID}"
  --dtype "${DTYPE}"
  --device-map auto
  --offload-folder "${OFFLOAD_FOLDER}"
  --batch-size "${BATCH_SIZE}"
  --extract-positions "${EXTRACT_POSITIONS}"
  --output "${OUTPUT_PATH}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  args+=(--trust-remote-code)
fi

echo "Starting frontier-model extraction: ${MODEL_ID}"
echo "Directions will be written to: ${OUTPUT_PATH}"
ablate "${args[@]}"
