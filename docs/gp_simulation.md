# GP-Simulated Posterior Fidelity

This document describes the GP-simulated posterior fidelity experiments used in the paper. The goal is to measure how 
closely TERA approximates exact derivative GP inference when the data are generated from the same GP prior. 
Training inputs and evaluation inputs are sampled separately, and the main metric is the average marginal KL divergence 
to the exact derivative GP posterior at the evaluation points.

The default configs use a Matérn-5/2 kernel, noiseless function and gradient observations, Sobol designs, and the target median prior correlation calibration reported in the paper.

---

## How to Run

The three main sweeps use the same run script. The compared methods, kernel, data sizes, conditioning sizes, and sweep variables are specified in the corresponding config files.

```bash
python scripts/run_gp_sim_kl_subprocess.py \
  --config configs/gp_sim_kl/matern52_m_sweep.yaml \
  --outdir outputs/main_m_sweep \
  --verbose

python scripts/run_gp_sim_kl_subprocess.py \
  --config configs/gp_sim_kl/matern52_d_sweep.yaml \
  --outdir outputs/main_d_sweep \
  --verbose

python scripts/run_gp_sim_kl_subprocess.py \
  --config configs/gp_sim_kl/matern52_n_sweep.yaml \
  --outdir outputs/main_n_sweep \
  --verbose
```