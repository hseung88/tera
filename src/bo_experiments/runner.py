from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from bo_experiments.benchmarks import SyntheticProblem, get_problem, initial_design
from bo_experiments.config import BOConfig, load_config
from bo_experiments.models import TERABOState, fit_single_task_gp, fit_tera_surrogate, optimize_log_ei
from bo_experiments.turbo import (
    TurboState,
    append_turbo_observation,
    make_turbo_state,
    select_turbo_candidate,
    select_turbo_logei_candidate,
)
from gp_sim_kl.utils import dtype_from_name

@dataclass(slots=True)
class BORow:
    experiment_name: str
    benchmark: str
    method: str
    seed: int
    tera_m: int
    dim: int
    active_dim: int
    eval_index: int
    y: float
    best_y: float
    regret: float
    fit_time_sec: float
    acq_time_sec: float
    eval_time_sec: float
    fit_peak_memory_gb: float
    acq_peak_memory_gb: float
    iter_peak_memory_gb: float
    trust_region_length: float
    status: str

def _standardize(y: torch.Tensor, g: torch.Tensor, cfg: BOConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if not cfg.standardize_y:
        return y, g
    std = y.std(unbiased=True).clamp_min(torch.finfo(y.dtype).eps)
    return (y - y.mean()) / std, g / std

def _unit_bounds(dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack([torch.zeros(dim, device=device, dtype=dtype), torch.ones(dim, device=device, dtype=dtype)])

def _peak_reset(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)

def _peak_gb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        return float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 3))
    return 0.0

def _row(cfg: BOConfig, problem: SyntheticProblem, method: str, seed: int, eval_index: int, y: float, best_y: float, fit_t: float, acq_t: float, eval_t: float, fit_mem: float, acq_mem: float, tr_len: float, status: str) -> BORow:
    return BORow(
        experiment_name=cfg.experiment_name, benchmark=problem.name, method=method, seed=seed,
        tera_m=int(getattr(cfg, "tera_m", 0)),
        dim=problem.dim, active_dim=problem.active_dim, eval_index=eval_index,
        y=float(y), best_y=float(best_y), regret=float(problem.optimum_value - best_y),
        fit_time_sec=float(fit_t), acq_time_sec=float(acq_t), eval_time_sec=float(eval_t),
        fit_peak_memory_gb=float(fit_mem), acq_peak_memory_gb=float(acq_mem),
        iter_peak_memory_gb=float(max(fit_mem, acq_mem)), trust_region_length=float(tr_len), status=status,
    )

def _print_progress(row: BORow) -> None:
    print(
        "[bo] "
        f"benchmark={row.benchmark} method={row.method} seed={row.seed} "
        f"iter={row.eval_index} "
        f"status={row.status} y={row.y:.6g} best_y={row.best_y:.6g} "
        f"regret={row.regret:.6g} fit={row.fit_time_sec:.3f}s "
        f"acq={row.acq_time_sec:.3f}s eval={row.eval_time_sec:.3f}s "
        f"mem={row.iter_peak_memory_gb:.3f}GB tr={row.trust_region_length:.4g}",
        flush=True,
    )

def _fit_and_select(method: str, X: torch.Tensor, y: torch.Tensor, g: torch.Tensor, cfg: BOConfig, *, seed: int, turbo_state: TurboState | None, tera_state: TERABOState | None, q: int = 1) -> tuple[torch.Tensor, float, float, float, float, float]:
    device = X.device
    if method == "sobol":
        candidate = initial_design(q, X.shape[-1], seed=seed, device=X.device, dtype=X.dtype)
        return candidate, 0.0, 0.0, 1.0, 0.0, 0.0
    if method == "turbo":
        if turbo_state is None:
            raise RuntimeError("TuRBO method requires a TurboState.")
        _peak_reset(device)
        t0 = time.perf_counter()
        candidate = select_turbo_candidate(turbo_state, cfg, seed=seed, q=q)
        total_t = time.perf_counter() - t0
        peak = _peak_gb(device)

        return candidate, total_t, 0.0, float(turbo_state.length), peak, peak

    if method == "turbo-logei":
        if turbo_state is None:
            raise RuntimeError("turbo-logei method requires a TurboState.")
        _peak_reset(device)
        candidate, fit_t, acq_t = select_turbo_logei_candidate(turbo_state, cfg, seed=seed)
        peak = _peak_gb(device)
        return candidate, fit_t, acq_t, float(turbo_state.length), peak, peak

    bounds = _unit_bounds(X.shape[-1], device=X.device, dtype=X.dtype)
    train_y, train_g = _standardize(y, g, cfg)
    _peak_reset(device)
    t0 = time.perf_counter()
    if method in {"vbo"}:
        model = fit_single_task_gp(X, train_y, cfg, method="vbo")
    elif method in {"tera"}:
        model = fit_tera_surrogate(X, train_y, train_g, cfg, seed=seed, state=tera_state)
    else:
        raise ValueError(f"Unknown BO method: {method}")
    fit_t = time.perf_counter() - t0
    fit_mem = _peak_gb(device)
    _peak_reset(device)
    t1 = time.perf_counter()
    candidate = optimize_log_ei(model, bounds, train_y.max().detach(), cfg, seed=seed + 7919 * int(X.shape[0]), method=method)
    acq_t = time.perf_counter() - t1
    acq_mem = _peak_gb(device)
    return candidate, fit_t, acq_t, 1.0, fit_mem, acq_mem

def run_one(cfg: BOConfig, problem: SyntheticProblem, method: str, seed: int, *, verbose: bool = False) -> list[BORow]:
    q_cfg = max(1, int(cfg.batch_size))
    if q_cfg != 1 and method != "turbo":
        raise ValueError(
            "Batch BO (batch_size > 1) is currently implemented only for the original TuRBO-TS method. "
            "Use --methods turbo, or set batch_size=1 for VBO, TuRBO-LogEI, and TERA."
        )

    torch.manual_seed(int(seed))
    X = initial_design(cfg.n_init, problem.dim, seed=seed, device=problem.lower.device, dtype=problem.lower.dtype)
    t_eval = time.perf_counter()
    y, g = problem.observe(X, noise_std=cfg.objective_noise_std)
    eval_t_total = time.perf_counter() - t_eval

    rows: list[BORow] = []
    for i in range(cfg.n_init):
        row = _row(
            cfg, problem, method, seed, i + 1,
            float(y[i].cpu()), float(y[: i + 1].max().cpu()),
            0, 0, eval_t_total / max(1, cfg.n_init), 0, 0, 1.0, "init",
        )
        rows.append(row)
        if verbose:
            _print_progress(row)

    turbo_state = make_turbo_state(X, y, cfg, seed=seed) if method in {"turbo", "turbo-logei"} else None
    tera_state = TERABOState() if method in {"tera"} else None

    next_eval = int(cfg.n_init) + 1
    while next_eval <= int(cfg.budget):
        q_step = min(q_cfg, int(cfg.budget) - next_eval + 1)
        candidate, fit_t, acq_t, tr_len, fit_mem, acq_mem = _fit_and_select(
            method,
            X,
            y,
            g,
            cfg,
            seed=seed + next_eval,
            turbo_state=turbo_state,
            tera_state=tera_state,
            q=q_step,
        )
        candidate = candidate.reshape(-1, candidate.shape[-1])
        t2 = time.perf_counter()
        y_new, g_new = problem.observe(candidate, noise_std=cfg.objective_noise_std)
        eval_t = time.perf_counter() - t2
        y_new = y_new.reshape(-1)
        g_new = g_new.reshape(candidate.shape[0], -1)

        X = torch.cat([X, candidate], dim=0)
        y = torch.cat([y, y_new], dim=0)
        g = torch.cat([g, g_new], dim=0)

        if turbo_state is not None:
            tr_len = append_turbo_observation(turbo_state, candidate, y_new, cfg, seed=seed + next_eval)

        q_actual = int(candidate.shape[0])
        fit_per = fit_t / max(1, q_actual)
        acq_per = acq_t / max(1, q_actual)
        eval_per = eval_t / max(1, q_actual)
        for j in range(q_actual):
            eval_idx = next_eval + j
            best_y = float(y[: int(cfg.n_init) + (eval_idx - int(cfg.n_init))].max().cpu())
            row = _row(
                cfg,
                problem,
                method,
                seed,
                eval_idx,
                float(y_new[j].cpu()),
                best_y,
                fit_per,
                acq_per,
                eval_per,
                fit_mem,
                acq_mem,
                tr_len,
                "ok",
            )
            rows.append(row)
            if verbose:
                _print_progress(row)
        next_eval += q_actual
    return rows

def _error_row(cfg: BOConfig, problem: SyntheticProblem, method: str, seed: int, message: str) -> BORow:
    return _row(cfg, problem, method, seed, 0, float("nan"), float("nan"), 0, 0, 0, 0, 0, 1.0, "error:" + message[:180].replace("\n", " "))

def run(cfg: BOConfig, outdir: str | Path, *, verbose: bool = False, append_results: bool = False) -> list[BORow]:
    device = torch.device(cfg.device if torch.cuda.is_available() or not cfg.device.startswith("cuda") else "cpu")
    dtype = dtype_from_name(cfg.dtype)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / cfg.out_csv_name
    existing_df = pd.read_csv(out_csv) if append_results and out_csv.exists() else None
    rows: list[BORow] = []

    def _write_rows() -> None:
        new_df = pd.DataFrame([asdict(r) for r in rows])
        if existing_df is not None:
            pd.concat([existing_df, new_df], ignore_index=True).to_csv(out_csv, index=False)
        else:
            new_df.to_csv(out_csv, index=False)
    for benchmark in cfg.benchmarks:
        problem = get_problem(benchmark, device=device, dtype=dtype)
        for seed in cfg.seeds:
            for method in cfg.methods:
                print(f"benchmark={benchmark} seed={seed} method={method}", flush=True)
                try:
                    method_rows = run_one(cfg, problem, method, int(seed), verbose=verbose)
                except Exception as exc:
                    if not cfg.continue_on_error:
                        raise
                    traceback.print_exc(); method_rows = [_error_row(cfg, problem, method, int(seed), str(exc))]
                rows.extend(method_rows)
                _write_rows()
    return rows

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run differentiable high-dimensional synthetic BO experiments.")
    p.add_argument("--config", required=True); p.add_argument("--outdir", required=True)
    p.add_argument("--benchmarks", default=None); p.add_argument("--methods", default=None); p.add_argument("--seeds", default=None)
    p.add_argument("--budget", type=int, default=None); p.add_argument("--n-init", type=int, default=None); p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--vbo-acq", default=None, choices=["log_nei", "log_ei"], help="Override VBO acquisition. Default is qLogNEI.")
    p.add_argument("--vbo-nei-mc-samples", type=int, default=None, help="Override qLogNEI Sobol MC sample count for VBO.")
    p.add_argument("--tera-m", type=int, default=None, help="Override TERA conditioning set size m.")
    p.add_argument("--tera-train-steps", type=int, default=None, help="Override TERA Adam steps. Sets initial and update train steps unless those are also specified.")
    p.add_argument("--tera-batch-size", type=int, default=None, help="Override TERA likelihood minibatch size.")
    p.add_argument("--tera-lr", type=float, default=None, help="Override TERA Adam learning rate.")
    p.add_argument("--tera-base-lengthscale", type=float, default=None, help="Override TERA base lengthscale used when tera_lengthscale_init='base'.")
    p.add_argument("--tera-sigma-f", type=float, default=None, help="Override TERA function-value noise level.")
    p.add_argument("--tera-sigma-g", type=float, default=None, help="Override TERA gradient noise level.")
    p.add_argument("--tera-acq-raw-samples", type=int, default=None, help="Override raw Sobol starts for TERA-LogEI.")
    p.add_argument("--tera-acq-restarts", type=int, default=None, help="Override TERA-LogEI restart count.")
    p.add_argument("--tera-acq-maxiter", type=int, default=None, help="Override TERA-LogEI LBFGS steps per restart.")
    p.add_argument("--tera-refit-every", type=int, default=None, help="Override TERA refit frequency in BO steps.")
    p.add_argument("--tera-initial-train-steps", type=int, default=None, help="Override Adam steps for the first TERA fit.")
    p.add_argument("--tera-update-train-steps", type=int, default=None, help="Override Adam steps for warm-started TERA refits.")
    p.add_argument("--tera-lengthscale-min", type=float, default=None, help="Override lower bound for TERA lengthscales.")
    p.add_argument("--tera-lengthscale-max", type=float, default=None, help="Override upper bound for TERA lengthscales.")
    p.add_argument("--device", default=None); p.add_argument("--dtype", default=None, choices=["float32", "float64"])
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--append-results", action="store_true", help="Append this run to an existing results.csv in --outdir instead of overwriting it.")
    p.add_argument("--verbose", action="store_true", help="Print per-evaluation BO progress with y, best_y, regret, timing, and memory.")
    return p.parse_args()

def main() -> None:
    args = _parse_args(); cfg = load_config(args.config)
    if args.benchmarks: cfg.benchmarks = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    if args.methods:
        from bo_experiments.config import _validate_methods
        cfg.methods = _validate_methods([x.strip() for x in args.methods.split(",") if x.strip()])
    if args.seeds: cfg.seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.budget is not None: cfg.budget = args.budget
    if args.n_init is not None: cfg.n_init = args.n_init
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.vbo_acq is not None: cfg.vbo_acq = args.vbo_acq
    if args.vbo_nei_mc_samples is not None: cfg.vbo_nei_mc_samples = args.vbo_nei_mc_samples
    if args.tera_m is not None: cfg.tera_m = args.tera_m
    if args.tera_train_steps is not None:
        cfg.tera_train_steps = args.tera_train_steps
        if args.tera_initial_train_steps is None:
            cfg.tera_initial_train_steps = args.tera_train_steps
        if args.tera_update_train_steps is None:
            cfg.tera_update_train_steps = args.tera_train_steps
    if args.tera_batch_size is not None: cfg.tera_batch_size = args.tera_batch_size
    if args.tera_lr is not None: cfg.tera_lr = args.tera_lr
    if args.tera_base_lengthscale is not None: cfg.tera_base_lengthscale = args.tera_base_lengthscale
    if args.tera_sigma_f is not None: cfg.tera_sigma_f = args.tera_sigma_f
    if args.tera_sigma_g is not None: cfg.tera_sigma_g = args.tera_sigma_g
    if args.tera_acq_raw_samples is not None: cfg.tera_acq_raw_samples = args.tera_acq_raw_samples
    if args.tera_acq_restarts is not None: cfg.tera_acq_restarts = args.tera_acq_restarts
    if args.tera_acq_maxiter is not None: cfg.tera_acq_maxiter = args.tera_acq_maxiter
    if args.tera_refit_every is not None: cfg.tera_refit_every = args.tera_refit_every
    if args.tera_initial_train_steps is not None: cfg.tera_initial_train_steps = args.tera_initial_train_steps
    if args.tera_update_train_steps is not None: cfg.tera_update_train_steps = args.tera_update_train_steps
    if args.tera_lengthscale_min is not None: cfg.tera_lengthscale_min = args.tera_lengthscale_min
    if args.tera_lengthscale_max is not None: cfg.tera_lengthscale_max = args.tera_lengthscale_max
    if args.device is not None: cfg.device = args.device
    if args.dtype is not None: cfg.dtype = args.dtype
    if args.continue_on_error: cfg.continue_on_error = True
    rows = run(cfg, args.outdir, verbose=args.verbose, append_results=args.append_results)
    print(f"Wrote {len(rows)} rows to {Path(args.outdir) / cfg.out_csv_name}")
