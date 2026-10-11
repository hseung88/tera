# High-Dimensional Bayesian Optimization

The main benchmarks are `Levy-200D` and `Ackley-200D`. The main comparison uses Sobol, dimension-scaled vanilla BO, 
TuRBO-1 with LogEI, and TERA. The default settings in `configs/bo/main.yaml` reproduce the main BO experiments 
reported in the paper, including the evaluation budget, initial design size, TERA training schedule, acquisition optimization settings, 
and random seeds.
TERA uses batched training and prediction, with initial lengthscale 1.0 (`tera_base_lengthscale`).
The budget is 150 queried locations, including 30 initial points shared across methods for each seed.

---
## How to Run

Use the following template to run one method on one benchmark.

```bash
benchmark=ackley_200
method=tera
outdir=outputs/bo_main_ackley200_tera

tera-bo \
  --config configs/bo/main.yaml \
  --outdir ${outdir} \
  --benchmarks ${benchmark} \
  --methods ${method} \
  --device cuda \
  --verbose
```

The main experiments use the following.

```bash
for benchmark in ackley_200 levy_200; do
  for method in tera vbo turbo-logei sobol; do
    outdir=outputs/bo_main_${benchmark}_${method}

    tera-bo \
      --config configs/bo/main.yaml \
      --outdir ${outdir} \
      --benchmarks ${benchmark} \
      --methods ${method} \
      --device cuda \
      --verbose
  done
done
```

## Original Large-Batch TuRBO Implementation

```bash
benchmark=ackley_200
method=turbo
outdir=outputs/bo_ackley_turbo_ts

tera-bo \
  --config configs/bo/turbo_ts_original.yaml \
  --outdir ${outdir} \
  --benchmarks ${benchmark} \
  --methods ${method} \
  --device cuda \
  --verbose
```