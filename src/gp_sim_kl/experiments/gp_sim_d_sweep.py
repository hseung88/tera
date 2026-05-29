from __future__ import annotations
from dataclasses import replace

import gc
import math
import torch

from gp_sim_kl.config import (
    METHOD_EXACT_DGP,
    METHOD_EXACT_DGP_DEROOS,
    METHOD_VECCHIA_DGP,
    METHOD_VECCHIA_DGP_DEROOS,
    ExperimentConfig,
    validate_methods,
)
from gp_sim_kl.data import PredictiveMarginals
from gp_sim_kl.metrics import average_marginal_kl, sum_marginal_kl
from gp_sim_kl.models import create_predictor
from gp_sim_kl.results import ExperimentRow
from gp_sim_kl.simulation import simulate_dataset
from gp_sim_kl.timing import time_build_predict_eval_peak_alloc

def resolve_conditioning_size(m: int | str, n_train: int) -> int:
    if isinstance(m, str):
        m_txt = m.strip().lower()
        if m_txt == "adaptive":
            return max(20, n_train // 10)
        raise ValueError(f"Unknown m specification: {m}")
    return int(m)

def is_method_feasible(cfg: ExperimentConfig, method_name: str, n_train: int, d: int, m_run: int) -> tuple[bool, str]:
    obs_dim = n_train * (d + 1)
    if method_name == METHOD_EXACT_DGP:
        if d > cfg.exact_dense_max_d:
            return False, f"skipped exact dense because d={d} > exact_dense_max_d={cfg.exact_dense_max_d}"
        if obs_dim > cfg.exact_dense_max_obs_dim:
            return False, f"skipped exact dense because obs_dim={obs_dim} > exact_dense_max_obs_dim={cfg.exact_dense_max_obs_dim}"
    if method_name == METHOD_EXACT_DGP_DEROOS:
        if cfg.sigma_g != 0.0:
            return False, "skipped deRoos exact dGP because sigma_g must be 0"
        if n_train > cfg.exact_deroos_max_n:
            return False, f"skipped deRoos exact dGP because n={n_train} > exact_deroos_max_n={cfg.exact_deroos_max_n}"
    if method_name == METHOD_VECCHIA_DGP:
        local_dim = m_run * d
        if local_dim > cfg.vecchia_dense_max_local_dim:
            return False, (
                f"skipped dense full-gradient Vecchia because m*d={local_dim} "
                f"> vecchia_dense_max_local_dim={cfg.vecchia_dense_max_local_dim}"
            )
    if method_name == METHOD_VECCHIA_DGP_DEROOS and cfg.sigma_g != 0.0:
        return False, "skipped deRoos Vecchia because sigma_g must be 0"
    return True, "ok"

def choose_exact_reference_method(
    methods: list[str],
    cfg: ExperimentConfig,
    *,
    n_train: int,
    d: int,
    m_run: int,
) -> str | None:
    for candidate in [METHOD_EXACT_DGP, METHOD_EXACT_DGP_DEROOS]:
        if candidate not in methods:
            continue
        ok, _ = is_method_feasible(cfg, candidate, n_train=n_train, d=d, m_run=m_run)
        if ok:
            return candidate
    return None

def make_skipped_row(cfg: ExperimentConfig, repeat: int, d: int, n_train: int, n_eval: int, m: int, method: str, status: str) -> ExperimentRow:
    return ExperimentRow(
        experiment_name=cfg.experiment_name,
        repeat=repeat,
        kernel=cfg.kernel,
        d=d,
        n_train=n_train,
        n_eval=n_eval,
        m=m,
        method=method,
        avg_marginal_kl=float("nan"),
        sum_marginal_kl=float("nan"),
        maxabs_mean_diff_to_exact_dgp=float("nan"),
        maxabs_var_diff_to_exact_dgp=float("nan"),
        maxabs_mean_diff_to_vecchia_deroos_reference=float("nan"),
        maxabs_var_diff_to_vecchia_deroos_reference=float("nan"),
        build_time_sec=float("nan"),
        predict_time_sec=float("nan"),
        peak_mem_gb=float("nan"),
        status=status,
    )

def cleanup_after_method(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

def compute_reference_prediction(
    data,
    method_name: str,
    m_run: int,
    device: torch.device,
) -> PredictiveMarginals:
    """Compute a reference prediction and keep it on the active device.

    This helper is intentionally outside the per-method timing window. The
    reference marginals are evaluation inputs, not part of the candidate method.
    For CUDA runs they remain on CUDA so metric evaluation does not shuttle
    intermediate prediction arrays through CPU memory.
    """
    method = create_predictor(method_name, m=m_run)
    pred: PredictiveMarginals | None = None
    try:
        with torch.no_grad():
            method.build(data)
            pred = method.predict_f_marginals(data.X_eval)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return PredictiveMarginals(
            mean=pred.mean.detach().contiguous(),
            var=pred.var.detach().contiguous(),
            mean_weights=None if pred.mean_weights is None else pred.mean_weights.detach().contiguous(),
        )
    finally:
        del pred
        del method
        cleanup_after_method(device)

def compute_prediction_metrics(
    pred: PredictiveMarginals,
    ref_pred: PredictiveMarginals | None,
    vec_ref: PredictiveMarginals | None,
) -> tuple[float, float, float, float, float, float]:
    pred_mean = pred.mean
    pred_var = pred.var

    if ref_pred is not None:
        ref_mean = ref_pred.mean
        ref_var = ref_pred.var
        avg_kl = float(average_marginal_kl(ref_mean, ref_var, pred_mean, pred_var).item())
        sum_kl = float(sum_marginal_kl(ref_mean, ref_var, pred_mean, pred_var).item())
        max_mean_exact = float(torch.max(torch.abs(ref_mean - pred_mean)).item())
        max_var_exact = float(torch.max(torch.abs(ref_var - pred_var)).item())
    else:
        avg_kl = float("nan")
        sum_kl = float("nan")
        max_mean_exact = float("nan")
        max_var_exact = float("nan")

    if vec_ref is not None:
        vec_mean = vec_ref.mean
        vec_var = vec_ref.var
        max_mean_vec = float(torch.max(torch.abs(vec_mean - pred_mean)).item())
        max_var_vec = float(torch.max(torch.abs(vec_var - pred_var)).item())
    else:
        max_mean_vec = float("nan")
        max_var_vec = float("nan")

    return (
        avg_kl,
        sum_kl,
        max_mean_exact,
        max_var_exact,
        max_mean_vec,
        max_var_vec,
    )

def _run_method(
    data,
    method_name: str,
    m_run: int,
    device: torch.device,
    ref_pred: PredictiveMarginals | None,
    vec_ref: PredictiveMarginals | None,
):
    method = create_predictor(method_name, m=m_run)
    try:
        metrics, build_time, predict_time, peak_mem = time_build_predict_eval_peak_alloc(
            device,
            lambda: method.build(data),
            lambda: method.predict_f_marginals(data.X_eval),
            lambda pred: compute_prediction_metrics(pred, ref_pred, vec_ref),
        )
        return metrics, build_time, predict_time, peak_mem
    finally:
        del method
        cleanup_after_method(device)

def _mean_sem(values: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) == 0:
        return float("nan"), float("nan")
    t = torch.tensor(vals, dtype=torch.float64)
    mean = float(t.mean().item())
    if t.numel() <= 1:
        return mean, 0.0
    sem = float((t.std(unbiased=True) / math.sqrt(t.numel())).item())
    return mean, sem

def _fmt_mean_sem(mean: float, sem: float, width: int = 10) -> str:
    if not math.isfinite(mean):
        return f"{'nan':>{width}}"
    if sem == 0.0 or not math.isfinite(sem):
        return f"{mean:{width}.2e}"
    return f"{mean:.2e}+/-{sem:.1e}"

def log_setting_summary(
    setting_rows: list[ExperimentRow],
    *,
    n_train: int,
    d: int,
    m_run: int,
    repeats: int,
) -> None:
    print(
        f"\nSummary after {repeats} repeat(s): "
        f"n={n_train} d={d} m={m_run}",
        flush=True,
    )
    print(
        f"{'method':<22} | "
        f"{'sum_KL':>16} | "
        f"{'mean_err':>16} | "
        f"{'var_err':>16} | "
        f"{'time_s':>16} | "
        f"{'peak_GB':>16} | "
        f"{'status':<8}",
        flush=True,
    )
    print("-" * 126, flush=True)

    methods_seen: list[str] = []
    for row in setting_rows:
        if row.method not in methods_seen:
            methods_seen.append(row.method)

    for method_name in methods_seen:
        sub = [r for r in setting_rows if r.method == method_name]
        ok = [r for r in sub if r.status == "ok"]

        if len(ok) == 0:
            status = sub[-1].status if sub else "missing"
            print(
                f"{method_name:<22} | "
                f"{'nan':>16} | "
                f"{'nan':>16} | "
                f"{'nan':>16} | "
                f"{'nan':>16} | "
                f"{'nan':>16} | "
                f"{status:<8}",
                flush=True,
            )
            continue

        kl_mean, kl_sem = _mean_sem([r.sum_marginal_kl for r in ok])
        mean_err_mean, mean_err_sem = _mean_sem(
            [r.maxabs_mean_diff_to_exact_dgp for r in ok]
        )
        var_err_mean, var_err_sem = _mean_sem(
            [r.maxabs_var_diff_to_exact_dgp for r in ok]
        )
        time_mean, time_sem = _mean_sem(
             [r.build_time_sec + r.predict_time_sec for r in ok]
        )
        peak_mean, peak_sem = _mean_sem([r.peak_mem_gb for r in ok])

        print(
            f"{method_name:<22} | "
            f"{_fmt_mean_sem(kl_mean, kl_sem, 16)} | "
            f"{_fmt_mean_sem(mean_err_mean, mean_err_sem, 16)} | "
            f"{_fmt_mean_sem(var_err_mean, var_err_sem, 16)} | "
            f"{_fmt_mean_sem(time_mean, time_sem, 16)} | "
            f"{_fmt_mean_sem(peak_mean, peak_sem, 16)} | "
            f"{'ok':<8}",
            flush=True,
        )

def run_gp_sim_d_sweep(cfg: ExperimentConfig, verbose: bool = False) -> list[ExperimentRow]:
    rows: list[ExperimentRow] = []
    device = torch.device(cfg.device)
    n_train_values = cfg.n_train_values if cfg.n_train_values is not None else [cfg.n_train]
    m_values_raw = cfg.m_values if cfg.m_values is not None else [cfg.m]
    methods = validate_methods(list(cfg.methods))

    total_settings = len(cfg.d_values) * len(n_train_values) * len(m_values_raw)
    setting_ctr = 0

    for n_train in n_train_values:
        cfg_n = cfg if n_train == cfg.n_train else replace(cfg, n_train=n_train)

        for d in cfg.d_values:
            for m_spec in m_values_raw:
                m_run = resolve_conditioning_size(m_spec, n_train)
                setting_ctr += 1
                setting_rows: list[ExperimentRow] = []

                if verbose:
                    print(
                        f"\n[setting {setting_ctr}/{total_settings}] "
                        f"n={n_train} d={d} m={m_run} | "
                        f"running {cfg.repeats} repeat(s)",
                        flush=True,
                    )

                for repeat in range(cfg.repeats):
                    if verbose:
                        print(
                            f"  repeat {repeat + 1}/{cfg.repeats} | simulate data",
                            flush=True,
                        )

                    def add_row(row: ExperimentRow) -> None:
                        rows.append(row)
                        setting_rows.append(row)

                    try:
                        data = simulate_dataset(cfg_n, d=d, repeat=repeat)
                    except Exception as exc:
                        status = f"data simulation failed: {type(exc).__name__}: {exc}"
                        if verbose:
                            print(f"  repeat {repeat + 1}/{cfg.repeats} | {status}", flush=True)

                        for method_name in methods:
                            add_row(
                                make_skipped_row(
                                    cfg,
                                    repeat,
                                    d,
                                    n_train,
                                    cfg.n_eval,
                                    m_run,
                                    method_name,
                                    status,
                                )
                            )
                        continue

                    ref_pred: PredictiveMarginals | None = None
                    vec_ref: PredictiveMarginals | None = None

                    ref_method = choose_exact_reference_method(
                        methods,
                        cfg_n,
                        n_train=n_train,
                        d=d,
                        m_run=m_run,
                    )
                    if ref_method is not None:
                        try:
                            ref_pred = compute_reference_prediction(
                                data,
                                ref_method,
                                m_run,
                                device,
                            )
                        except Exception as exc:
                            if verbose:
                                print(
                                    f"  repeat {repeat + 1}/{cfg.repeats} | "
                                    f"reference {ref_method} failed: {type(exc).__name__}: {exc}",
                                    flush=True,
                                )
                            ref_pred = None

                    if METHOD_VECCHIA_DGP_DEROOS in methods:
                        ok_vec, _ = is_method_feasible(
                            cfg_n,
                            METHOD_VECCHIA_DGP_DEROOS,
                            n_train=n_train,
                            d=d,
                            m_run=m_run,
                        )
                        if ok_vec:
                            try:
                                vec_ref = compute_reference_prediction(
                                    data,
                                    METHOD_VECCHIA_DGP_DEROOS,
                                    m_run,
                                    device,
                                )
                            except Exception as exc:
                                if verbose:
                                    print(
                                        f"  repeat {repeat + 1}/{cfg.repeats} | "
                                        f"Vecchia dGP-deRoos diagnostic reference failed: "
                                        f"{type(exc).__name__}: {exc}",
                                        flush=True,
                                    )
                                vec_ref = None

                    for method_name in methods:
                        ok, reason = is_method_feasible(
                            cfg_n,
                            method_name,
                            n_train=n_train,
                            d=d,
                            m_run=m_run,
                        )
                        if not ok:
                            add_row(
                                make_skipped_row(
                                    cfg,
                                    repeat,
                                    d,
                                    n_train,
                                    cfg.n_eval,
                                    m_run,
                                    method_name,
                                    reason,
                                )
                            )
                            continue

                        try:
                            metrics, build_time, predict_time, peak_mem = _run_method(
                                data,
                                method_name,
                                m_run,
                                device,
                                ref_pred,
                                vec_ref,
                            )
                            (
                                avg_kl,
                                sum_kl,
                                max_mean_exact,
                                max_var_exact,
                                max_mean_vec,
                                max_var_vec,
                            ) = metrics

                            add_row(
                                ExperimentRow(
                                    experiment_name=cfg.experiment_name,
                                    repeat=repeat,
                                    kernel=cfg.kernel,
                                    d=d,
                                    n_train=n_train,
                                    n_eval=cfg.n_eval,
                                    m=m_run,
                                    method=method_name,
                                    avg_marginal_kl=avg_kl,
                                    sum_marginal_kl=sum_kl,
                                    maxabs_mean_diff_to_exact_dgp=max_mean_exact,
                                    maxabs_var_diff_to_exact_dgp=max_var_exact,
                                    maxabs_mean_diff_to_vecchia_deroos_reference=max_mean_vec,
                                    maxabs_var_diff_to_vecchia_deroos_reference=max_var_vec,
                                    build_time_sec=build_time,
                                    predict_time_sec=predict_time,
                                    peak_mem_gb=peak_mem,
                                    status="ok",
                                )
                            )

                        except Exception as exc:
                            status = f"failed: {type(exc).__name__}: {exc}"
                            if verbose:
                                print(
                                    f"  repeat {repeat + 1}/{cfg.repeats} | "
                                    f"{method_name} | {status}",
                                    flush=True,
                                )
                            add_row(
                                make_skipped_row(
                                    cfg,
                                    repeat,
                                    d,
                                    n_train,
                                    cfg.n_eval,
                                    m_run,
                                    method_name,
                                    status,
                                )
                            )

                    del ref_pred
                    del vec_ref
                    cleanup_after_method(device)

                if verbose:
                    log_setting_summary(
                        setting_rows,
                        n_train=n_train,
                        d=d,
                        m_run=m_run,
                        repeats=cfg.repeats,
                    )

    return rows
