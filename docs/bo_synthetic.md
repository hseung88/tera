# High-Dimensional Bayesian Optimization

The default settings in `configs/bo/main.yaml` reproduce the main BO experiments 
reported in the paper.

---
## How to Run

Use the following template to run one method on one benchmark.

```bash
benchmark=ackley_200
method=tera
outdir=outputs/bo_main_ackley200_tera

python scripts/run_bo_synthetic.py \
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

    python scripts/run_bo_synthetic.py \
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

python scripts/run_bo_synthetic.py \
  --config configs/bo/turbo_ts_original.yaml \
  --outdir ${outdir} \
  --benchmarks ${benchmark} \
  --methods ${method} \
  --device cuda \
  --verbose
```