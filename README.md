# Scalable Derivative Gaussian Processes via Exact Gradient Reduction

> Accepted at **the Conference on Neural Information Processing Systems (NeurIPS 2026)**, December 2026. 

This repository provides the official implementation of **TERA**, a scalable derivative Gaussian process method for 
scalar function-value prediction with observed input gradients. For each target and its Vecchia conditioning set, 
TERA replaces the full gradient by target-specific reduced directional derivatives. For stationary kernels, this 
reduction preserves the corresponding full-gradient conditional density exactly, while avoiding the dense derivative 
covariance over all gradient coordinates.

TERA training and prediction use batched reduced conditional systems.
Method names in commands, results, and plot legends are lowercase; existing result labels remain supported.

The code reproduces the experiments for:

- GP-simulated posterior fidelity and computational scaling relative to exact derivative GP inference.
- Large-scale GP regression with observed gradients on the MD22 benchmark.
- High-dimensional Bayesian optimization on synthetic function benchmarks.

---

## Main Results

The figures below show the original submission results; camera-ready experiments are being rerun.

### Posterior Fidelity and Scaling

![GP-simulated posterior fidelity and scaling](figures/fidelity_scaling_combined.png)

**Figure 1:** Posterior fidelity and computational scaling under GP simulations. Rows show sweeps over the conditioning 
size `m` at `(d, n) = (500, 150)`, input dimension `d` at `(m, n) = (20, 150)`, and training size `n` at `d = 150` with 
`m = max(20, n / 10)`. Columns report average marginal KL divergence to the exact derivative GP posterior, 
wall-clock time, and peak GPU memory.

### Large-Scale GP Regression

![Large-scale scalar energy prediction on MD22](figures/md22_main_metrics_rbf.png)

**Figure 2:** Large-scale scalar energy prediction on MD22. Panels show raw energy test RMSE per atom after conversion 
to physical units, end-to-end wall-clock time, and peak GPU memory. All methods use isotropic squared exponential kernels 
and identical train and test splits. Lower values are better in all panels.

### High-Dimensional Bayesian Optimization

![High-dimensional Bayesian optimization](figures/bo_main_regret.png)

**Figure 3:** High-dimensional BO under a fixed budget of 150 queried locations, including the initial design. The first two 
panels report simple regret on the full-dimensional `Levy-200D` and `Ackley-200D` benchmarks, respectively. 
The right panel reports wall-clock time and peak GPU memory on `Ackley-200D`.

---

## Installation

Use Python 3.10 or newer. A CUDA-enabled PyTorch installation is recommended for the main experiments.

```bash
python -m venv tera
source tera/bin/activate
pip install -e '.[md22,bo]'
```
---

## Experiments

Each experiment has a default config for the camera-ready experiments and a dedicated reproduction guide with run commands.


| Experiment                        | Run command                            | Default config                           | Documentation |
|:----------------------------------|:--------------------------------------|:-----------------------------------------|:--|
| Posterior fidelity and scaling    | `tera-gp-sim-subprocess` | `configs/gp_sim/matern52_*.yaml`      | `docs/gp_simulation.md` |
| Large-scale GP regression on MD22 | `tera-md22`   | `configs/md22/main.yaml` | `docs/md22_regression.md` |
| High-dimensional BO               | `tera-bo`         | `configs/bo/main.yaml`      | `docs/bo_synthetic.md` |

---

## Citation

```bibtex
@inproceedings{seung2026scalabledg,
  author        = {Seung, Hyunseok and Katzfuss, Matthias},
  title         = {Scalable Derivative Gaussian Processes via Exact Gradient Reduction},
  booktitle     = {Advances in Neural Information Processing Systems},
  year          = {2026}
}
```
