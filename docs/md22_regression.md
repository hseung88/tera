# Large-Scale GP Regression on MD22

This document describes the MD22 regression experiments used in the paper.
The default settings in `configs/md22/main.yaml` match the paper experiments.

---
## Download MD22 Data

Download the MD22 datasets used in the paper:

```bash
bash scripts/download_md22.sh
```

## How to Run

Set the common dataset list once.

```bash
datasets=DHA,AT-AT,stachyose,AT-AT-CG-CG,buckyball-catcher,double-walled-nanotube
```

### Main Setting: Isotropic SE Kernel

```bash
python scripts/run_md22_fair_pipeline.py \
  --config configs/md22/main.yaml \
  --outdir outputs/md22_tera_rbf_no_ard \
  --datasets ${datasets} \
  --device cuda:0 \
  --kernel rbf \
  --no-use-ard \
  --methods tera \
  --verbose
```

Omit `--methods tera` to run all configured methods: tera, exact gp, dsoftki, and ddsvgp.

### Additional Setting: ARD Matérn-5/2 Kernel

```bash
python scripts/run_md22_fair_pipeline.py \
  --config configs/md22/main.yaml \
  --outdir outputs/md22_tera_matern52_ard \
  --datasets ${datasets} \
  --device cuda:0 \
  --kernel matern52 \
  --use-ard \
  --methods tera \
  --tera-gradient-noise-model scaled \
  --verbose
```