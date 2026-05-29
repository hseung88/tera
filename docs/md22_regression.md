# Large-Scale GP Regression on MD22

This document describes the MD22 regression experiments used in the paper. The task is scalar molecular energy 
prediction, with observed forces converted to gradient information. The reported metric is raw energy RMSE per atom 
after converting predictions back to physical units.

The main comparison uses isotropic squared exponential kernels. The ARD Matérn-5/2 kernel is included as an additional 
setting. The default settings in `configs/md22_regression/md22_main.yaml` match the paper experiments, including the 
TERA conditioning size, noise initialization, training schedule, batch size, and learning rate.

---
## Download MD22 Data

Download the MD22 datasets used in the paper:

```bash
bash scripts/download_md22.sh
```

## How to Run

Set the common dataset list once.

```bash
DATASETS=DHA,AT-AT,stachyose,AT-AT-CG-CG,buckyball-catcher,double-walled-nanotube
```

### Main Setting: Isotropic SE Kernel

```bash
python scripts/run_md22_fair_pipeline.py \
  --config configs/md22_regression/md22_main.yaml \
  --outdir outputs/md22_tera_rbf_no_ard \
  --datasets ${DATASETS} \
  --device cuda:0 \
  --kernel rbf \
  --no-use-ard \
  --internal-methods "TERA" \
  --skip-external \
  --verbose
```

### Additional Setting: ARD Matérn-5/2 Kernel

```bash
python scripts/run_md22_fair_pipeline.py \
  --config configs/md22_regression/md22_main.yaml \
  --outdir outputs/md22_tera_matern52_ard \
  --datasets ${DATASETS} \
  --device cuda:0 \
  --kernel matern52 \
  --use-ard \
  --internal-methods "TERA" \
  --skip-external \
  --tera-gradient-noise-model scaled \
  --verbose
```