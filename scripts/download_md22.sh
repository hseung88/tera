#!/usr/bin/env bash
set -euo pipefail

data_dir="${1:-data/md22}"
base_url="http://www.quantum-machine.org/gdml/repo/datasets"

datasets=(
  "DHA"
  "buckyball-catcher"
  "double-walled_nanotube"
  "AT-AT"
  "AT-AT-CG-CG"
  "stachyose"
)

mkdir -p "${data_dir}"

for dataset in "${datasets[@]}"; do
  file="md22_${dataset}.npz"
  url="${base_url}/${file}"
  out="${data_dir}/${file}"

  if [[ -f "${out}" ]]; then
    echo "Found ${out}; skipping."
  else
    echo "Downloading ${file}..."
    wget -O "${out}" "${url}"
  fi
done

echo "MD22 datasets are available in ${data_dir}."
