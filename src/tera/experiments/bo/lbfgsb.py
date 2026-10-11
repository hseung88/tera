"""Bound-constrained gradient baseline with an exact observation budget."""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.optimize import minimize

from tera.experiments.bo.benchmarks import initial_design
from tera.methods.common.random import random_stream, seed_for


def run_lbfgsb(cfg, problem, seed, *, verbose=False):
    from tera.experiments.bo.runner import _peak_gb, _peak_reset, _print_progress, _row

    if cfg.objective_noise_std != 0:
        raise ValueError("l-bfgs-b cost comparison requires a deterministic objective.")
    device, dtype = problem.lower.device, problem.lower.dtype
    generator = torch.Generator(device=device).manual_seed(
        seed_for(seed, random_stream.objective_noise)
    )
    _peak_reset(device)
    start = time.perf_counter()
    inputs = initial_design(cfg.n_init, problem.dim, seed=seed, device=device, dtype=dtype)
    t_eval = time.perf_counter()
    y, g = problem.observe(inputs, with_grad=True, generator=generator)
    peak = _peak_gb(device)
    eval_total = time.perf_counter() - t_eval
    elapsed_init = time.perf_counter() - start
    rows = []
    for i in range(cfg.n_init):
        row = _row(
            cfg,
            problem,
            "lbfgsb",
            seed,
            i + 1,
            float(y[i]),
            float(y[: i + 1].max()),
            0,
            0,
            eval_total / cfg.n_init,
            0,
            0,
            1,
            "init",
        )
        row.eval_peak_memory_gb = row.iter_peak_memory_gb = peak
        row.elapsed_time_sec = elapsed_init * (i + 1) / cfg.n_init
        rows.append(row)
    best = float(y.max())
    incumbent = int(y.argmax())
    # The initial incumbent is already observed; scipy's first fun/jac request is cached.
    cache_x = inputs[incumbent].detach().cpu().double().numpy().copy()
    cache_value = (-float(y[incumbent]), -g[incumbent].detach().cpu().double().numpy().copy())
    initial_x = cache_x.copy()

    class budget_exhausted(Exception):
        pass

    def fun_jac(x):
        nonlocal cache_x, cache_value, best
        if np.array_equal(x, cache_x):
            return cache_value
        if len(rows) >= cfg.budget:
            raise budget_exhausted
        _peak_reset(device)
        t0 = time.perf_counter()
        value, gradient = problem.observe(
            torch.as_tensor(x, device=device, dtype=dtype)[None], with_grad=True
        )
        peak_eval = _peak_gb(device)
        eval_t = time.perf_counter() - t0
        reward = float(value[0])
        grad = gradient[0].detach().cpu().double().numpy().copy()
        if not np.isfinite(reward) or not np.isfinite(grad).all():
            raise FloatingPointError("Non-finite l-bfgs-b objective or gradient.")
        best = max(best, reward)
        elapsed = time.perf_counter() - start
        overhead = max(0.0, elapsed - rows[-1].elapsed_time_sec - eval_t)
        row = _row(
            cfg,
            problem,
            "lbfgsb",
            seed,
            len(rows) + 1,
            reward,
            best,
            0,
            overhead,
            eval_t,
            0,
            0,
            1,
            "ok",
        )
        row.eval_peak_memory_gb = row.iter_peak_memory_gb = peak_eval
        row.elapsed_time_sec = elapsed
        rows.append(row)
        if verbose:
            _print_progress(row)
        cache_x, cache_value = np.array(x, copy=True), (-reward, -grad)
        return cache_value

    message, converged, stopped = "Initial design exhausted the budget", False, True
    if len(rows) < cfg.budget:
        try:
            result = minimize(
                fun_jac,
                initial_x,
                method='l-bfgs-b',
                jac=True,
                bounds=[(0.0, 1.0)] * problem.dim,
                options={
                    "maxiter": cfg.lbfgsb_maxiter,
                    "maxls": cfg.lbfgsb_maxls,
                    "ftol": cfg.lbfgsb_ftol,
                    "gtol": cfg.lbfgsb_gtol,
                },
            )
            message, converged, stopped = str(result.message), bool(result.success), False
        except budget_exhausted:
            message, converged, stopped = (
                "Observation budget exhausted (including initial design and line searches)",
                False,
                True,
            )
    final_elapsed = time.perf_counter() - start
    rows[-1].acq_time_sec += max(0.0, final_elapsed - rows[-1].elapsed_time_sec)
    rows[-1].elapsed_time_sec = final_elapsed
    for row in rows:
        row.optimizer_message = message
        row.optimizer_converged = converged
        row.optimizer_stopped_by_budget = stopped
    return rows
