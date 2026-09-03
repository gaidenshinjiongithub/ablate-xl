#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/ablate-xl}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
  echo "Ablate XL checkout not found at ${PROJECT_DIR}." >&2
  echo "Clone the repository there or set PROJECT_DIR." >&2
  exit 2
fi

python -m pip install --upgrade pip packaging ninja
python -m pip install --editable "${PROJECT_DIR}[all,runpod]"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  MAX_JOBS="${MAX_JOBS:-8}" python -m pip install --no-build-isolation flash-attn
fi

echo "Ablate XL is installed from ${PROJECT_DIR}."
