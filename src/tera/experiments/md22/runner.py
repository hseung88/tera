from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path as path_type

import pandas as pd
import torch

from tera.experiments.md22.config import md22_config, load_config, resolve_dataset_config, resolve_method_config
from tera.experiments.md22.data import add_observation_noise, load_md22_raw, make_split, save_fair_split
from tera.experiments.md22.metrics import normalized_energy_rmse_per_atom, raw_energy_rmse_per_atom
from tera.experiments.md22.models import ddsvgp_model, dsoftki_model, exact_gp_model, tera_model
from tera.experiments.options import add_shared_arguments, resolve_cli
from tera.methods.common.random import random_stream, seed_for
from tera.methods.common.utils import dtype_from_name
from tera.methods.names import normalize_md22_method_name, md22_method_label
from tera.methods.vecchia.model import vecchia_gp_model


@dataclass(slots=True)
class md22_result_row:
    experiment_name: str
    dataset: str
    seed: int
    method: str
    n_train: int
    n_test: int
    d: int
    n_atoms: int
    split_id: str
    preprocessing_version: str
    x_scale: float
    m: int
    prediction_batch_size: int
    kernel: str
    normalized_energy_rmse_per_atom: float
    raw_energy_rmse_per_atom: float
    fit_time_sec: float
    predict_time_sec: float
    wall_time_sec: float
    peak_mem_gb: float
    status: str
    fit_peak_mem_gb: float = math.nan
    predict_peak_mem_gb: float = math.nan
    normalized_rmse_scale: str = "train_energy_std"
    model_target_scale: float = math.nan
    model_dtype: str = ""
    device: str = ""
    observation_noise_fraction: float = 0.0
    energy_noise_fraction: float = 0.0
    force_noise_fraction: float = 0.0
    observation_noise_id: str = "clean"
    energy_noise_std: float = 0.0
    force_noise_std: float = 0.0
    added_function_noise_var: float = 0.0
    added_gradient_noise_var: float = 0.0


result_columns = ['experiment_name', 'dataset', 'seed', 'method', 'n_train', 'n_test', 'd', 'n_atoms', 'split_id', 'preprocessing_version', 'x_scale', 'm', 'kernel', 'normalized_energy_rmse_per_atom', 'raw_energy_rmse_per_atom', 'fit_time_sec', 'predict_time_sec', 'wall_time_sec', 'peak_mem_gb', 'status']


def result_record(row):
    return {name: getattr(row, name) for name in result_columns}


def observation_settings(split) -> dict:
    return {
        name: getattr(split, name)
        for name in (
            "observation_noise_fraction",
            "energy_noise_fraction",
            "force_noise_fraction",
            "observation_noise_id",
            "energy_noise_std",
            "force_noise_std",
            "added_function_noise_var",
            "added_gradient_noise_var",
        )
    }


def config_with_observation_noise(cfg, split):
    if (
        not cfg.initialize_noise_from_observations
        or split.energy_noise_fraction == split.force_noise_fraction == 0
    ):
        return cfg
    return replace(
        cfg,
        sigma_f=cfg.sigma_f + split.added_function_noise_var,
        sigma_g=cfg.sigma_g + split.added_gradient_noise_var,
    )


def _clear_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0**3)


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Use --device cpu explicitly."
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} does not exist.")
        device = torch.device("cuda", index)
    return device


def _standard_gp_lr_for_dataset(cfg: md22_config, dataset_name: str) -> float:
    if cfg.standard_gp_lr is not None:
        return float(cfg.standard_gp_lr)
    return float(cfg.standard_gp_lr_by_dataset.get(dataset_name, cfg.standard_gp_lr_default))


def _make_model(method: str, cfg: md22_config, seed: int, *, dataset_name: str):
    method = normalize_md22_method_name(method)
    cfg = resolve_method_config(cfg, method)
    seed = seed_for(seed, random_stream.training)
    if method == "standard_gp":
        model = exact_gp_model(
            kernel=cfg.kernel,
            outputscale=cfg.outputscale,
            sigma_f=cfg.sigma_f,
            lengthscale=cfg.lengthscale,
            lengthscale_init=cfg.lengthscale_init,
            lengthscale_init_max_points=cfg.lengthscale_init_max_points,
            use_ard=cfg.use_ard,
            max_train=cfg.standard_gp_max_train,
            train_steps=cfg.standard_gp_train_steps,
            train_epochs=cfg.standard_gp_train_epochs,
            lr=_standard_gp_lr_for_dataset(cfg, dataset_name),
            weight_decay=cfg.standard_gp_weight_decay,
            learn_lengthscale=cfg.standard_gp_learn_lengthscale,
            learn_outputscale=cfg.standard_gp_learn_outputscale,
            learn_sigma_f=cfg.standard_gp_learn_sigma_f,
            min_sigma_f=cfg.standard_gp_min_sigma_f,
            log_every=cfg.standard_gp_log_every,
        )
    elif method in {"tera", "tera_batched", "vecchia"}:
        prefix = method if method in {"vecchia"} else "tera"
        training_fields = (
            "train_steps",
            "train_epochs",
            "graph_refresh_epochs",
            "train_batch_size",
            "lr",
            "weight_decay",
            "learn_lengthscale",
            "learn_outputscale",
            "learn_sigma_f",
            "learn_sigma_g",
            "min_sigma_f",
            "min_sigma_g",
            "log_every",
            "gradient_noise_model",
        )
        options = {
            name: getattr(cfg, f"{prefix}_{name}")
            for name in training_fields
            if hasattr(cfg, f"{prefix}_{name}")
        }
        options["prediction_batch_size"] = getattr(cfg, f"{prefix}_prediction_batch_size")
        model_class = {
            "tera": tera_model,
            "vecchia": vecchia_gp_model,
        }[method]
        model = model_class(
            m=cfg.m,
            kernel=cfg.kernel,
            outputscale=cfg.outputscale,
            sigma_f=cfg.sigma_f,
            sigma_g=cfg.sigma_g,
            lengthscale=cfg.lengthscale,
            lengthscale_init=cfg.lengthscale_init,
            lengthscale_init_max_points=cfg.lengthscale_init_max_points,
            use_ard=cfg.use_ard,
            seed=seed,
            **options,
        )
    elif method == "dsoftki":
        model = dsoftki_model(config=cfg, seed=seed)
    elif method == "ddsvgp":
        model = ddsvgp_model(config=cfg, seed=seed)
    else:
        raise ValueError(f"Unknown MD22 method: {method}")
    model.name = md22_method_label(method) if method in {"tera", "standard_gp", "vecchia"} else method
    model.validation_enabled = cfg.log_training_curves
    return model


def _run_one_method(
    cfg: md22_config,
    split,
    method: str,
    seed: int,
    device: torch.device,
    history: list | None = None,
    on_fit=None,
) -> md22_result_row:
    cfg = resolve_method_config(cfg, method)
    model = _make_model(method, cfg, seed, dataset_name=split.name)
    try:
        _clear_cuda(device)
        t0 = time.perf_counter()
        model.fit(split)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fit_time = time.perf_counter() - t0
        fit_peak = _peak_gb(device)
        if on_fit is not None:
            on_fit(model, cfg, seed, split)
        if history is not None:
            cumulative_fit = 0.0
            baseline_history = getattr(getattr(model, "model", None), "training_history", [])
            for index, rec in enumerate(getattr(model, "training_history", [])):
                if md22_method_label(method) in {'dsoftki', 'ddsvgp'}:
                    source = baseline_history[index]
                    cumulative_fit += float(source.get("epoch_time", 0.0)) + float(source.get("fit_time", 0.0))
                    rec = dict(rec, fit_time_sec=cumulative_fit)
                history.append(
                    dict(
                        rec,
                        dataset=split.name,
                        method=md22_method_label(method),
                        seed=seed,
                        m=cfg.m,
                        split_id=split.split_id,
                        status="ok",
                    )
                )
        t1 = time.perf_counter()
        with torch.no_grad():
            pred = model.predict(split.x_test)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            predict_time = time.perf_counter() - t1
            predict_peak = _peak_gb(device)
            if not bool(torch.isfinite(pred.y_mean).all()):
                raise FloatingPointError("Non-finite predictive mean.")
            if pred.y_var is not None and not bool(torch.isfinite(pred.y_var).all()):
                raise FloatingPointError("Non-finite predictive variance.")
            norm_rmse = normalized_energy_rmse_per_atom(pred.y_mean, split.y_test, split.n_atoms)
            raw_rmse = raw_energy_rmse_per_atom(
                pred.y_mean,
                split.e_test,
                energy_mean=split.scaler.energy_mean,
                energy_std=split.scaler.energy_std,
                n_atoms=split.n_atoms,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        predict_time = time.perf_counter() - t1
        peak = max(fit_peak, predict_peak)
        return md22_result_row(
            experiment_name=cfg.experiment_name,
            dataset=split.name,
            seed=seed,
            method=md22_method_label(method),
            n_train=int(split.x_train.shape[0]),
            n_test=int(split.x_test.shape[0]),
            d=split.d,
            n_atoms=split.n_atoms,
            split_id=split.split_id,
            preprocessing_version=split.preprocessing_version,
            x_scale=float(split.scaler.x_scale),
            m=cfg.m,
            prediction_batch_size=getattr(
                model,
                "prediction_batch_size",
                1,
            ),
            kernel=getattr(model, "kernel", cfg.kernel),
            normalized_energy_rmse_per_atom=norm_rmse,
            raw_energy_rmse_per_atom=raw_rmse,
            fit_time_sec=fit_time,
            predict_time_sec=predict_time,
            wall_time_sec=fit_time + predict_time,
            peak_mem_gb=peak,
            fit_peak_mem_gb=fit_peak,
            predict_peak_mem_gb=predict_peak,
            model_target_scale=getattr(model, "target_scale", float(split.scaler.energy_std)),
            model_dtype=getattr(
                cfg, {"dsoftki": "dsoftki_dtype", "ddsvgp": "ddsvgp_dtype"}.get(method, "dtype")
            ),
            device=str(device),
            status="ok",
            **observation_settings(split),
        )
    except Exception as exc:
        return _failed_row(cfg, split, method, seed, str(exc))


def _failed_row(cfg: md22_config, split, method: str, seed: int, status: str) -> md22_result_row:
    cfg = resolve_method_config(cfg, method)
    return md22_result_row(
        experiment_name=cfg.experiment_name,
        dataset=split.name,
        seed=seed,
        method=md22_method_label(method),
        n_train=int(split.x_train.shape[0]),
        n_test=int(split.x_test.shape[0]),
        d=split.d,
        n_atoms=split.n_atoms,
        split_id=split.split_id,
        preprocessing_version=split.preprocessing_version,
        x_scale=float(split.scaler.x_scale),
        m=cfg.m,
        prediction_batch_size=(cfg.tera_prediction_batch_size if method == "tera" else 1),
        kernel=cfg.kernel,
        normalized_energy_rmse_per_atom=math.nan,
        raw_energy_rmse_per_atom=math.nan,
        fit_time_sec=math.nan,
        predict_time_sec=math.nan,
        wall_time_sec=math.nan,
        peak_mem_gb=math.nan,
        status=f"failed: {status}",
        **observation_settings(split),
    )


def run(cfg: md22_config, outdir: str | path_type, *, on_fit=None) -> list[md22_result_row]:
    cfg = replace(cfg, methods=list(dict.fromkeys(map(normalize_md22_method_name, cfg.methods))))
    device = resolve_device(cfg.device)
    dtype = dtype_from_name(cfg.dtype)
    rows: list[md22_result_row] = []
    history: list[dict] = []
    outdir = path_type(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for dataset in cfg.datasets:
        raw = load_md22_raw(cfg.data_dir, dataset, device=device, dtype=dtype)
        dataset_cfg = resolve_dataset_config(cfg, raw.d)
        for seed in cfg.seeds:
            split = make_split(
                raw,
                seed=seed,
                train_frac=cfg.train_frac,
                test_frac=cfg.test_frac,
                n_train=cfg.n_train,
                n_test=cfg.n_test,
                x_scale=cfg.x_scale,
                preprocessing_version=cfg.preprocessing_version,
            )
            split = add_observation_noise(
                split,
                cfg.observation_noise_fraction,
                energy_fraction=cfg.energy_noise_fraction,
                force_fraction=cfg.force_noise_fraction,
                seed=seed if cfg.observation_noise_seed is None else cfg.observation_noise_seed,
            )
            if cfg.export_fair_data:
                fair_dir = path_type(cfg.fair_data_dir) if cfg.fair_data_dir else outdir / "fair_splits"
                save_fair_split(
                    split, fair_dir / f"{dataset}_seed{seed}_{split.split_id}.npz",
                    seed=seed, train_frac=cfg.train_frac, test_frac=cfg.test_frac,
                    n_train=cfg.n_train, n_test=cfg.n_test,
                )
            if cfg.verbose:
                print(f"dataset={dataset} seed={seed} n_train={len(split.x_train)} n_test={len(split.x_test)} d={split.d}")
            run_cfg = config_with_observation_noise(dataset_cfg, split)
            for method in cfg.methods:
                method_cfg = resolve_method_config(run_cfg, method)
                if cfg.verbose:
                    print(f"  running {md22_method_label(method)} ...", flush=True)
                row = _run_one_method(
                    method_cfg,
                    split,
                    method,
                    seed,
                    device,
                    history=history if cfg.log_training_curves else None,
                    on_fit=on_fit,
                )
                rows.append(row)
                if history:
                    pd.DataFrame(history).to_csv(outdir / "training_curves.csv", index=False)
                if cfg.verbose:
                    print(
                        f"  {md22_method_label(method)}: normalized_rmse={row.normalized_energy_rmse_per_atom} raw_rmse={row.raw_energy_rmse_per_atom} time={row.wall_time_sec} mem={row.peak_mem_gb} status={row.status}"
                    )
                pd.DataFrame([result_record(r) for r in rows]).to_csv(
                    outdir / cfg.out_csv_name, index=False
                )
    return rows


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integer seeds, e.g. 1,2,3.") from exc
    if any(seed < 0 or seed >= 2**64 for seed in seeds):
        raise argparse.ArgumentTypeError("Seeds must be between 0 and 2**64 - 1.")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MD22 GP regression with observed gradients.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated: exact gp,dsoftki,ddsvgp,tera.",
    )
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset override.")
    p.add_argument(
        "--seeds", type=_parse_seeds, default=None, help="Comma-separated seed override."
    )
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--kernel", type=str, default=None, choices=["rbf", "matern52"])
    p.add_argument("--m", type=int, default=None)
    p.add_argument(
        "--sigma-f",
        "--sigma_f",
        dest="sigma_f",
        type=float,
        default=None,
        help="Function/value noise variance.",
    )
    p.add_argument(
        "--sigma-g",
        "--sigma_g",
        dest="sigma_g",
        type=float,
        default=None,
        help="Gradient noise variance.",
    )
    p.add_argument(
        "--lengthscale", type=float, default=None, help="Explicit isotropic lengthscale override."
    )
    p.add_argument("--lengthscale-init", type=str, default=None, choices=["median", "one"])
    p.add_argument(
        "--learn-lengthscale", dest="learn_lengthscale", action="store_true", default=None
    )
    p.add_argument("--no-learn-lengthscale", dest="learn_lengthscale", action="store_false")
    p.add_argument("--use-ard", dest="use_ard", action="store_true", default=None)
    p.add_argument("--no-use-ard", dest="use_ard", action="store_false")
    p.add_argument(
        "--standard-gp-max-train",
        "--exact-max-train",
        type=int,
        default=None,
        help="Maximum n_train allowed for dense exact gp before skipping.",
    )
    p.add_argument("--standard-gp-train-epochs", "--exact-train-epochs", type=int, default=None)
    p.add_argument("--standard-gp-train-steps", "--exact-train-steps", type=int, default=None)
    p.add_argument(
        "--standard-gp-lr",
        "--exact-lr",
        "--exact_lr",
        "--standard_gp_lr",
        dest="standard_gp_lr",
        type=float,
        default=None,
    )
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None)
    for method in ("tera",):
        p.add_argument(
            f"--{method}-train-batch-size",
            f"--{method}-batch-size",
            type=int,
            help="Conditional likelihood factors per parameter update.",
        )
        p.add_argument(f"--{method}-gradient-noise-model", choices=("iid", "scaled"))
    p.add_argument(
        "--tera-prediction-batch-size",
        type=int,
        help="Prediction targets processed together at fitted parameters.",
    )
    p.add_argument("--no-export-fair-data", dest="export_fair_data", action="store_false", default=None)
    p.add_argument("--fair-data-dir", default=None)
    p.add_argument("--log-training-curves", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--curve-log-every", type=int, default=None)
    for prefix in ("dsoftki", "ddsvgp"):
        for field in ("train-epochs", "num-inducing", "batch-size", "prediction-batch-size"):
            p.add_argument(f"--{prefix}-{field}", type=int, default=None)
        p.add_argument(f"--{prefix}-lr", type=float, default=None)
    from tera.experiments.cli import add_config_arguments

    add_config_arguments(
        p,
        md22_config,
        prefixes=("standard_gp_", "tera_", "vecchia_", "dsoftki_", "ddsvgp_", "baseline_"),
        names=(
            "data_dir",
            "dtype",
            "outputscale",
            "observation_noise_fraction",
            "energy_noise_fraction",
            "force_noise_fraction",
            "observation_noise_seed",
            "initialize_noise_from_observations",
        ),
    )
    add_shared_arguments(p, md22_config, "md22")
    return p


def _parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def config_from_args(args) -> md22_config:
    cfg = load_config(args.config)
    if args.methods is not None:
        cfg.methods = [
            normalize_md22_method_name(m.strip()) for m in args.methods.split(",") if m.strip()
        ]
    if args.datasets is not None:
        cfg.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.seeds is not None:
        cfg.seeds = args.seeds
    if not cfg.methods or not cfg.datasets or not cfg.seeds:
        raise ValueError("At least one method, dataset, and seed is required.")
    cfg = resolve_cli(cfg, args, "md22")
    for prefix in ("standard_gp", "tera", "vecchia"):
        if (
            cfg.log_training_curves
            and getattr(args, f"{prefix}_log_every", None) is None
            and getattr(args, "log_every", None) is None
        ):
            setattr(cfg, f"{prefix}_log_every", cfg.curve_log_every)
    return cfg


def main() -> None:
    args = _parse_args()
    cfg = config_from_args(args)
    rows = run(cfg, args.outdir)
    print(f"Wrote {len(rows)} rows to {path_type(args.outdir) / cfg.out_csv_name}")


if __name__ == "__main__":
    main()
