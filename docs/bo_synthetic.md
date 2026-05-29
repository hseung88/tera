# High-Dimensional Bayesian Optimization

The main benchmarks are `Levy-200D` and `Ackley-200D`. The main comparison uses Sobol, dimension-scaled vanilla BO, 
TuRBO-1 with LogEI, and TERA. The default settings in `configs/bo_synthetic/bo_main.yaml` reproduce the main BO experiments 
reported in the paper, including the evaluation budget, initial design size, TERA training schedule, acquisition optimization settings, 
and random seeds.

---
## How to Run

Use the following template to run one method on one benchmark.

```bash
BENCHMARK=ackley_200
METHOD=tera
OUTDIR=outputs/bo_main_ackley200_tera

python scripts/run_bo_synthetic.py \
  --config configs/bo_synthetic/bo_main.yaml \
  --outdir ${OUTDIR} \
  --benchmarks ${BENCHMARK} \
  --methods ${METHOD} \
  --device cuda \
  --verbose
```

The main experiments use the following.

```bash
for BENCHMARK in ackley_200 levy_200; do
  for METHOD in tera vbo turbo-logei sobol; do
    OUTDIR=outputs/bo_main_${BENCHMARK}_${METHOD}

    python scripts/run_bo_synthetic.py \
      --config configs/bo_synthetic/bo_main.yaml \
      --outdir ${OUTDIR} \
      --benchmarks ${BENCHMARK} \
      --methods ${METHOD} \
      --device cuda \
      --verbose
  done
done
```

## Original Large-Batch TuRBO Implementation

```bash
BENCHMARK=ackley_200
METHOD=turbo
OUTDIR=outputs/bo_ackley_turbo_ts

python scripts/run_bo_synthetic.py \
  --config configs/bo_synthetic/bo_turbo_ackley200_original.yaml \
  --outdir ${OUTDIR} \
  --benchmarks ${BENCHMARK} \
  --methods ${METHOD} \
  --device cuda \
  --verbose
```

