from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path as path_type
from typing import Any as any_type

import torch

from tera.experiments.gp_sim.config import experiment_config
from tera.experiments.gp_sim.data import predictive_marginals, simulated_dataset
from tera.experiments.gp_sim.experiments.gp_sim_d_sweep import (
    compute_prediction_metrics,
    compute_reference_prediction,
    is_method_feasible,
    make_skipped_row,
)
from tera.experiments.gp_sim.models import create_predictor
from tera.experiments.gp_sim.results import experiment_row
from tera.experiments.gp_sim.simulation import simulate_dataset
from tera.experiments.gp_sim.timing import time_build_predict_eval_peak_alloc


def config_from_json(text: str) -> experiment_config:
    raw = json.loads(text)
    return experiment_config(**raw)


def config_to_json(cfg: experiment_config) -> str:
    return json.dumps(asdict(cfg), separators=(",", ":"))


def _predictive_to_payload(
    pred: predictive_marginals | None,
) -> dict[str, torch.Tensor | None] | None:
    if pred is None:
        return None
    return {
        "mean": pred.mean.detach().contiguous(),
        "var": pred.var.detach().contiguous(),
        "mean_weights": None
        if pred.mean_weights is None
        else pred.mean_weights.detach().contiguous(),
    }


def _predictive_from_payload(
    payload: dict[str, torch.Tensor | None] | None,
) -> predictive_marginals | None:
    if payload is None:
        return None
    return predictive_marginals(
        mean=payload["mean"],
        var=payload["var"],
        mean_weights=payload.get("mean_weights"),
    )


def _dataset_to_payload(data: simulated_dataset) -> dict[str, any_type]:
    return {
        "X_train": data.x_train.detach().contiguous(),
        "X_train_scaled": data.x_train_scaled.detach().contiguous(),
        "X_eval": data.x_eval.detach().contiguous(),
        "X_eval_scaled": data.x_eval_scaled.detach().contiguous(),
        "lengthscale": data.lengthscale.detach().contiguous(),
        "outputscale": data.outputscale,
        "sigma_f": data.sigma_f,
        "sigma_g": data.sigma_g,
        "kernel_name": data.kernel_name,
        "f_train_obs": data.f_train_obs.detach().contiguous(),
        "g_train_obs": data.g_train_obs.detach().contiguous(),
        "z_train_obs": data.z_train_obs.detach().contiguous(),
        "sampling_backend": data.sampling_backend,
    }


def _dataset_from_payload(payload: dict[str, any_type]) -> simulated_dataset:
    return simulated_dataset(
        x_train=payload["X_train"],
        x_train_scaled=payload["X_train_scaled"],
        x_eval=payload["X_eval"],
        x_eval_scaled=payload["X_eval_scaled"],
        lengthscale=payload["lengthscale"],
        outputscale=float(payload["outputscale"]),
        sigma_f=float(payload["sigma_f"]),
        sigma_g=float(payload["sigma_g"]),
        kernel_name=str(payload["kernel_name"]),
        f_train_obs=payload["f_train_obs"],
        g_train_obs=payload["g_train_obs"],
        z_train_obs=payload["z_train_obs"],
        sampling_backend=str(payload["sampling_backend"]),
    )


def prepare_cache(
    *,
    cfg: experiment_config,
    cache_path: str | path_type,
    d: int,
    n_train: int,
    m_run: int,
    repeat: int,
    ref_method: str | None,
    vec_ref_method: str | None,
) -> None:
    """Cache the data and KL references outside method timing.
    """
    cfg_n = cfg if n_train == cfg.n_train else replace(cfg, n_train=n_train)
    device = torch.device(cfg_n.device)
    data = simulate_dataset(cfg_n, d=d, repeat=repeat)

    ref_pred: predictive_marginals | None = None
    if ref_method is not None:
        ref_pred = compute_reference_prediction(data, ref_method, m_run, device)

    vec_ref: predictive_marginals | None = None
    if vec_ref_method is not None:
        vec_ref = compute_reference_prediction(data, vec_ref_method, m_run, device)

    payload = {
        "cfg": asdict(cfg_n),
        "d": d,
        "n_train": n_train,
        "m": m_run,
        "repeat": repeat,
        "data": _dataset_to_payload(data),
        "ref_pred": _predictive_to_payload(ref_pred),
        "vec_ref": _predictive_to_payload(vec_ref),
    }
    cache_path = path_type(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def run_method_from_cache(
    *,
    cfg: experiment_config,
    cache_path: str | path_type,
    method_name: str,
    d: int,
    n_train: int,
    n_eval: int,
    m_run: int,
    repeat: int,
) -> experiment_row:
    """Run one method in one fresh process and return one result row."""
    cfg_n = cfg if n_train == cfg.n_train else replace(cfg, n_train=n_train)
    device = torch.device(cfg_n.device)
    ok, reason = is_method_feasible(cfg_n, method_name, n_train=n_train, d=d, m_run=m_run)
    if not ok:
        return make_skipped_row(cfg, repeat, d, n_train, n_eval, m_run, method_name, reason)

    try:
        payload = torch.load(cache_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location=device)
    data = _dataset_from_payload(payload["data"])
    ref_pred = _predictive_from_payload(payload.get("ref_pred"))
    vec_ref = _predictive_from_payload(payload.get("vec_ref"))

    method = create_predictor(method_name, m=m_run, prediction_batch_size=cfg.prediction_batch_size)
    try:
        metrics, build_time, predict_time, peak_mem = time_build_predict_eval_peak_alloc(
            device,
            lambda: method.build(data),
            lambda: method.predict_f_marginals(data.x_eval),
            lambda pred: compute_prediction_metrics(pred, ref_pred, vec_ref),
        )
        (
            avg_kl,
            sum_kl,
            max_mean_exact,
            max_var_exact,
            max_mean_vec,
            max_var_vec,
        ) = metrics
        return experiment_row(
            experiment_name=cfg.experiment_name,
            repeat=repeat,
            kernel=cfg.kernel,
            d=d,
            n_train=n_train,
            n_eval=n_eval,
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
    except Exception as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        return make_skipped_row(cfg, repeat, d, n_train, n_eval, m_run, method_name, status)
    finally:
        method = None
