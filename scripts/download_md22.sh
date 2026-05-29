#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/md22}"
BASE_URL="http://www.quantum-machine.org/gdml/repo/datasets"

DATASETS=(
  "DHA"
  "buckyball-catcher"
  "double-walled_nanotube"
  "AT-AT"
  "AT-AT-CG-CG"
  "stachyose"
)

mkdir -p "${DATA_DIR}"

for DATASET in "${DATASETS[@]}"; do
  FILE="md22_${DATASET}.npz"
  URL="${BASE_URL}/${FILE}"
  OUT="${DATA_DIR}/${FILE}"

  if [[ -f "${OUT}" ]]; then
    echo "Found ${OUT}; skipping."
  else
    echo "Downloading ${FILE}..."
    wget -O "${OUT}" "${URL}"
  fi
done

echo "MD22 datasets are available in ${DATA_DIR}."
