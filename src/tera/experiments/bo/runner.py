from __future__ import annotations

import argparse
import gc
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path as path_type

import pandas as pd
import torch

from tera.experiments.bo.acquisition import optimize_log_ei
from tera.experiments.bo.benchmarks import synthetic_problem, get_problem, initial_design
from tera.experiments.bo.cli import build_parser, config_from_args
from tera.experiments.bo.config import bo_config, _validate_methods
from tera.methods.common.random import random_stream, rng_scope, seed_for
from tera.methods.common.utils import dtype_from_name
from tera.methods.tera.bo import terabo_state, fit_tera_surrogate
from tera.methods.turbo import turbo_state, append_turbo_observation, make_turbo_state, select_turbo_candidate, select_turbo_logei_candidate
from tera.methods.vbo import fit_single_task_gp


@dataclass(slots=True)
class bo_row:
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
    elapsed_time_sec: float = float("nan")
    eval_peak_memory_gb: float = 0.0
    tera_prediction_batch_size: int = 0
    objective_loss: float = float("nan")
    best_objective_loss: float = float("nan")
    gradients_observed: bool = True
    optimizer_message: str = ""
    optimizer_converged: bool | None = None
    optimizer_stopped_by_budget: bool | None = None


result_columns = ['experiment_name', 'benchmark', 'method', 'seed', 'tera_m', 'dim', 'active_dim', 'eval_index', 'y', 'best_y', 'regret', 'fit_time_sec', 'acq_time_sec', 'eval_time_sec', 'fit_peak_memory_gb', 'acq_peak_memory_gb', 'iter_peak_memory_gb', 'trust_region_length', 'status']


def result_record(row):
    return {name: getattr(row, name) for name in result_columns}


def _standardize(
    y: torch.Tensor, g: torch.Tensor, cfg: bo_config
) -> tuple[torch.Tensor, torch.Tensor]:
    if not cfg.standardize_y:
        return y, g
    std = y.std(unbiased=True).clamp_min(torch.finfo(y.dtype).eps)
    return (y - y.mean()) / std, g / std


def _unit_bounds(dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack(
        [torch.zeros(dim, device=device, dtype=dtype), torch.ones(dim, device=device, dtype=dtype)]
    )


def _peak_reset(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_gb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        return float(torch.cuda.max_memory_allocated(device) / (1024.0**3))
    return 0.0


def _row(
    cfg: bo_config,
    problem: synthetic_problem,
    method: str,
    seed: int,
    eval_index: int,
    y: float,
    best_y: float,
    fit_t: float,
    acq_t: float,
    eval_t: float,
    fit_mem: float,
    acq_mem: float,
    tr_len: float,
    status: str,
) -> bo_row:
    return bo_row(
        experiment_name=cfg.experiment_name,
        benchmark=problem.name,
        method=method,
        seed=seed,
        tera_m=int(getattr(cfg, "tera_m", 0)),
        tera_prediction_batch_size=(cfg.tera_prediction_batch_size if method == "tera" else 0),
        dim=problem.dim,
        active_dim=problem.active_dim,
        eval_index=eval_index,
        y=float(y),
        best_y=float(best_y),
        regret=float(problem.optimum_value - best_y),
        objective_loss=-float(y),
        best_objective_loss=-float(best_y),
        gradients_observed=method in {"tera", "lbfgsb"},
        fit_time_sec=float(fit_t),
        acq_time_sec=float(acq_t),
        eval_time_sec=float(eval_t),
        fit_peak_memory_gb=float(fit_mem),
        acq_peak_memory_gb=float(acq_mem),
        iter_peak_memory_gb=float(max(fit_mem, acq_mem)),
        trust_region_length=float(tr_len),
        status=status,
    )


def _print_progress(row: bo_row) -> None:
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


def _fit_and_select(
    method: str,
    inputs: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    cfg: bo_config,
    *,
    seed: int,
    turbo_state: turbo_state | None,
    tera_state: terabo_state | None,
    q: int = 1,
) -> tuple[torch.Tensor, float, float, float, float, float]:
    device = inputs.device
    if method == "sobol":
        candidate = initial_design(q, inputs.shape[-1], seed=seed, device=inputs.device, dtype=inputs.dtype)
        return candidate, 0.0, 0.0, 1.0, 0.0, 0.0
    if method == "turbo":
        if turbo_state is None:
            raise RuntimeError("TuRBO method requires a TurboState.")
        _peak_reset(device)
        t0 = time.perf_counter()
        candidate = select_turbo_candidate(turbo_state, cfg, seed=seed, q=q)
        peak = _peak_gb(device)
        total_t = time.perf_counter() - t0

        return candidate, total_t, 0.0, float(turbo_state.length), peak, peak

    if method == "turbo-logei":
        if turbo_state is None:
            raise RuntimeError("turbo-logei method requires a TurboState.")
        _peak_reset(device)
        candidate, fit_t, acq_t = select_turbo_logei_candidate(turbo_state, cfg, seed=seed)
        peak = _peak_gb(device)
        return candidate, fit_t, acq_t, float(turbo_state.length), peak, peak

    bounds = _unit_bounds(inputs.shape[-1], device=inputs.device, dtype=inputs.dtype)
    train_y, train_g = _standardize(y, g, cfg)
    _peak_reset(device)
    t0 = time.perf_counter()
    training_seed = seed_for(seed, random_stream.training)
    with rng_scope(training_seed, device=device):
        if method in {"vbo"}:
            model = fit_single_task_gp(inputs, train_y, cfg, method="vbo")
        elif method in {"tera"}:
            model = fit_tera_surrogate(
                inputs,
                train_y,
                train_g,
                cfg,
                seed=training_seed,
                state=tera_state,
            )
        else:
            raise ValueError(f"Unknown BO method: {method}")
    fit_mem = _peak_gb(device)
    fit_t = time.perf_counter() - t0
    _peak_reset(device)
    t1 = time.perf_counter()
    candidate = optimize_log_ei(
        model,
        bounds,
        train_y.max().detach(),
        cfg,
        seed=seed_for(seed, random_stream.acquisition),
        method=method,
    )
    acq_mem = _peak_gb(device)
    acq_t = time.perf_counter() - t1
    return candidate, fit_t, acq_t, 1.0, fit_mem, acq_mem


def run_one(
    cfg: bo_config, problem: synthetic_problem, method: str, seed: int, *, verbose: bool = False
) -> list[bo_row]:
    method = _validate_methods([method])[0]
    if not 1 <= cfg.n_init <= cfg.budget:
        raise ValueError("Require 1 <= n_init <= budget.")
    q_cfg = max(1, int(cfg.batch_size))
    if q_cfg != 1 and method != "turbo":
        raise ValueError(
            "Batch BO (batch_size > 1) is currently implemented only for the original TuRBO-TS method. "
            "Use --methods turbo, or set batch_size=1 for vbo, TuRBO-LogEI, tera."
        )

    if method == "lbfgsb":
        from tera.experiments.bo.lbfgsb import run_lbfgsb

        return run_lbfgsb(cfg, problem, seed, verbose=verbose)

    _peak_reset(problem.lower.device)
    run_start = time.perf_counter()
    noise_generator = torch.Generator(device=problem.lower.device).manual_seed(
        seed_for(seed, random_stream.objective_noise)
    )
    inputs = initial_design(
        cfg.n_init, problem.dim, seed=seed, device=problem.lower.device, dtype=problem.lower.dtype
    )
    t_eval = time.perf_counter()
    use_gradients = method in {
        "tera",
    }
    y, g = problem.observe(
        inputs, noise_std=cfg.objective_noise_std, with_grad=use_gradients, generator=noise_generator
    )
    if problem.lower.device.type == "cuda":
        torch.cuda.synchronize(problem.lower.device)
    eval_t_total = time.perf_counter() - t_eval
    init_peak = _peak_gb(problem.lower.device)
    init_elapsed = time.perf_counter() - run_start

    rows: list[bo_row] = []
    for i in range(cfg.n_init):
        row = _row(
            cfg,
            problem,
            method,
            seed,
            i + 1,
            float(y[i].cpu()),
            float(y[: i + 1].max().cpu()),
            0,
            0,
            eval_t_total / max(1, cfg.n_init),
            0,
            0,
            1.0,
            "init",
        )
        row.elapsed_time_sec = init_elapsed * (i + 1) / cfg.n_init
        row.eval_peak_memory_gb = init_peak
        rows.append(row)
        if verbose:
            _print_progress(row)

    turbo_state = (
        make_turbo_state(inputs, y, cfg, seed=seed) if method in {"turbo", "turbo-logei"} else None
    )
    tera_state = terabo_state() if method == "tera" else None

    next_eval = int(cfg.n_init) + 1
    while next_eval <= int(cfg.budget):
        q_step = min(q_cfg, int(cfg.budget) - next_eval + 1)
        candidate, fit_t, acq_t, tr_len, fit_mem, acq_mem = _fit_and_select(
            method,
            inputs,
            y,
            g,
            cfg,
            seed=seed_for(seed, random_stream.bo_step, next_eval),
            turbo_state=turbo_state,
            tera_state=tera_state,
            q=q_step,
        )
        candidate = candidate.reshape(-1, candidate.shape[-1])
        _peak_reset(problem.lower.device)
        t2 = time.perf_counter()
        y_new, g_new = problem.observe(
            candidate,
            noise_std=cfg.objective_noise_std,
            with_grad=use_gradients,
            generator=noise_generator,
        )
        if problem.lower.device.type == "cuda":
            torch.cuda.synchronize(problem.lower.device)
        eval_t = time.perf_counter() - t2
        eval_peak = _peak_gb(problem.lower.device)
        y_new = y_new.reshape(-1)
        g_new = g_new.reshape(candidate.shape[0], -1)

        inputs = torch.cat([inputs, candidate], dim=0)
        y = torch.cat([y, y_new], dim=0)
        g = torch.cat([g, g_new], dim=0)

        if turbo_state is not None:
            tr_len = append_turbo_observation(
                turbo_state,
                candidate,
                y_new,
                cfg,
                seed=seed_for(seed, random_stream.bo_step, next_eval),
            )

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
            row.elapsed_time_sec = time.perf_counter() - run_start
            row.eval_peak_memory_gb = eval_peak
            row.iter_peak_memory_gb = max(fit_mem, acq_mem)
            rows.append(row)
            if verbose:
                _print_progress(row)
        next_eval += q_actual
    return rows


def _error_row(
    cfg: bo_config, problem: synthetic_problem, method: str, seed: int, message: str
) -> bo_row:
    return _row(
        cfg,
        problem,
        method,
        seed,
        0,
        float("nan"),
        float("nan"),
        0,
        0,
        0,
        0,
        0,
        1.0,
        "error:" + message[:180].replace("\n", " "),
    )


def run(
    cfg: bo_config, outdir: str | path_type, *, verbose: bool = False, append_results: bool = False
) -> list[bo_row]:
    cfg = replace(cfg, methods=_validate_methods(cfg.methods))
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; set device=cpu explicitly for smoke runs."
        )
    dtype = dtype_from_name(cfg.dtype)
    outdir = path_type(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = outdir / cfg.out_csv_name
    existing_df = pd.read_csv(out_csv) if append_results and out_csv.exists() else None
    rows: list[bo_row] = []

    def _write_rows() -> None:
        new_df = pd.DataFrame([result_record(r) for r in rows])
        if existing_df is not None:
            pd.concat([existing_df, new_df], ignore_index=True).to_csv(out_csv, index=False)
        else:
            new_df.to_csv(out_csv, index=False)

    for benchmark in cfg.benchmarks:
        problem = get_problem(benchmark, device=device, dtype=dtype, config=cfg)
        for seed in cfg.seeds:
            for method in cfg.methods:
                print(f"benchmark={benchmark} seed={seed} method={method}", flush=True)
                try:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.empty_cache()
                    method_rows = run_one(cfg, problem, method, int(seed), verbose=verbose)
                except Exception as exc:
                    if not cfg.continue_on_error:
                        raise
                    traceback.print_exc()
                    method_rows = [_error_row(cfg, problem, method, int(seed), str(exc))]
                rows.extend(method_rows)
                _write_rows()
    return rows


def _parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> None:
    args = _parse_args()
    cfg = config_from_args(args)
    rows = run(cfg, args.outdir, verbose=args.verbose, append_results=args.append_results)
    print(f"Wrote {len(rows)} rows to {path_type(args.outdir) / cfg.out_csv_name}")


if __name__ == "__main__":
    main()
