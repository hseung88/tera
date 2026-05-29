from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from gp_sim_kl.utils import dtype_from_name
from md22_regression.config import MD22Config, load_config
from md22_regression.data import load_md22_raw, make_split, save_fair_split
from md22_regression.metrics import normalized_energy_rmse_per_atom, raw_energy_rmse_per_atom
from md22_regression.models import DDSVGPModel, DSoftKIModel, ExactGPModel, TERAModel
from md22_regression.models.external import ExternalResultsModel

@dataclass(slots=True)
class MD22ResultRow:
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
    kernel: str
    normalized_energy_rmse_per_atom: float
    raw_energy_rmse_per_atom: float
    fit_time_sec: float
    predict_time_sec: float
    wall_time_sec: float
    peak_mem_gb: float
    status: str

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
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3)

def _exact_gp_lr_for_dataset(cfg: MD22Config, dataset_name: str) -> float:
    if cfg.exact_lr is not None:
        return float(cfg.exact_lr)
    return float(cfg.exact_lr_by_dataset.get(dataset_name, cfg.exact_lr_default))

def _make_model(method: str, cfg: MD22Config, seed: int, *, dataset_name: str):
    if method == "Exact GP":
        return ExactGPModel(
            kernel=cfg.kernel,
            outputscale=cfg.outputscale,
            sigma_f=cfg.sigma_f,
            lengthscale=cfg.lengthscale,
            lengthscale_init=cfg.lengthscale_init,
            lengthscale_init_max_points=cfg.lengthscale_init_max_points,
            use_ard=cfg.use_ard,
            max_train=cfg.exact_max_train,
            train_steps=cfg.exact_train_steps,
            train_epochs=cfg.exact_train_epochs,
            lr=_exact_gp_lr_for_dataset(cfg, dataset_name),
            weight_decay=cfg.exact_weight_decay,
            learn_lengthscale=cfg.exact_learn_lengthscale,
            learn_outputscale=cfg.exact_learn_outputscale,
            learn_sigma_f=cfg.exact_learn_sigma_f,
            min_sigma_f=cfg.exact_min_sigma_f,
            log_every=cfg.exact_log_every,
        )
    if method == "TERA":
        return TERAModel(
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
            train_steps=cfg.tera_train_steps,
            train_epochs=cfg.tera_train_epochs,
            graph_refresh_epochs=cfg.tera_graph_refresh_epochs,
            batch_size=cfg.tera_batch_size,
            lr=cfg.tera_lr,
            weight_decay=cfg.tera_weight_decay,
            learn_lengthscale=cfg.tera_learn_lengthscale,
            learn_outputscale=cfg.tera_learn_outputscale,
            learn_sigma_f=cfg.tera_learn_sigma_f,
            learn_sigma_g=cfg.tera_learn_sigma_g,
            min_sigma_f=cfg.tera_min_sigma_f,
            min_sigma_g=cfg.tera_min_sigma_g,
            log_every=cfg.tera_log_every,
            gradient_noise_model=cfg.tera_gradient_noise_model,
        )
    if method == "DSoftKI":
        return DSoftKIModel(
            csv_path=cfg.external_results_csv,
            seed=seed,
            strict_fairness=cfg.strict_external_fairness,
            preprocessing_version=cfg.preprocessing_version,
            x_scale=cfg.x_scale,
        )
    if method == "DDSVGP":
        return DDSVGPModel(
            csv_path=cfg.external_results_csv,
            seed=seed,
            strict_fairness=cfg.strict_external_fairness,
            preprocessing_version=cfg.preprocessing_version,
            x_scale=cfg.x_scale,
        )
    raise ValueError(f"Unknown MD22 method: {method}")

def _run_one_method(cfg: MD22Config, split, method: str, seed: int, device: torch.device) -> MD22ResultRow:
    model = _make_model(method, cfg, seed, dataset_name=split.name)
    if isinstance(model, ExternalResultsModel):
        try:
            model.fit(split)
            return MD22ResultRow(
                experiment_name=cfg.experiment_name,
                dataset=split.name,
                seed=seed,
                method=method,
                n_train=int(split.X_train.shape[0]),
                n_test=int(split.X_test.shape[0]),
                d=split.d,
                n_atoms=split.n_atoms,
                split_id=split.split_id,
                preprocessing_version=split.preprocessing_version,
                x_scale=float(split.scaler.x_scale),
                m=cfg.m,
                kernel=cfg.kernel,
                normalized_energy_rmse_per_atom=float(model.normalized_energy_rmse_per_atom),
                raw_energy_rmse_per_atom=float(model.raw_energy_rmse_per_atom),
                fit_time_sec=float("nan"),
                predict_time_sec=float("nan"),
                wall_time_sec=float(model.wall_time_sec or 0.0),
                peak_mem_gb=float(model.peak_mem_gb or 0.0),
                status="external_csv",
            )
        except Exception as exc:
            return _failed_row(cfg, split, method, seed, f"external_missing: {exc}")

    try:
        _clear_cuda(device)
        t0 = time.perf_counter()
        model.fit(split)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        with torch.no_grad():
            pred = model.predict(split.X_test)
            norm_rmse = normalized_energy_rmse_per_atom(pred.y_mean, split.y_test, split.n_atoms)
            raw_rmse = raw_energy_rmse_per_atom(
                pred.y_mean,
                split.E_test,
                energy_mean=split.scaler.energy_mean,
                energy_std=split.scaler.energy_std,
                n_atoms=split.n_atoms,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        predict_time = time.perf_counter() - t1
        peak = _peak_gb(device)
        return MD22ResultRow(
            experiment_name=cfg.experiment_name,
            dataset=split.name,
            seed=seed,
            method=method,
            n_train=int(split.X_train.shape[0]),
            n_test=int(split.X_test.shape[0]),
            d=split.d,
            n_atoms=split.n_atoms,
            split_id=split.split_id,
            preprocessing_version=split.preprocessing_version,
            x_scale=float(split.scaler.x_scale),
            m=cfg.m,
            kernel=cfg.kernel,
            normalized_energy_rmse_per_atom=norm_rmse,
            raw_energy_rmse_per_atom=raw_rmse,
            fit_time_sec=fit_time,
            predict_time_sec=predict_time,
            wall_time_sec=fit_time + predict_time,
            peak_mem_gb=peak,
            status="ok",
        )
    except Exception as exc:
        return _failed_row(cfg, split, method, seed, str(exc))

def _failed_row(cfg: MD22Config, split, method: str, seed: int, status: str) -> MD22ResultRow:
    return MD22ResultRow(
        experiment_name=cfg.experiment_name,
        dataset=split.name,
        seed=seed,
        method=method,
        n_train=int(split.X_train.shape[0]),
        n_test=int(split.X_test.shape[0]),
        d=split.d,
        n_atoms=split.n_atoms,
        split_id=split.split_id,
        preprocessing_version=split.preprocessing_version,
        x_scale=float(split.scaler.x_scale),
        m=cfg.m,
        kernel=cfg.kernel,
        normalized_energy_rmse_per_atom=math.nan,
        raw_energy_rmse_per_atom=math.nan,
        fit_time_sec=math.nan,
        predict_time_sec=math.nan,
        wall_time_sec=math.nan,
        peak_mem_gb=math.nan,
        status=f"failed: {status}",
    )

def run(cfg: MD22Config, outdir: str | Path) -> list[MD22ResultRow]:
    device = torch.device(cfg.device if torch.cuda.is_available() or not cfg.device.startswith("cuda") else "cpu")
    dtype = dtype_from_name(cfg.dtype)
    rows: list[MD22ResultRow] = []
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for dataset in cfg.datasets:
        raw = load_md22_raw(cfg.data_dir, dataset, device=device, dtype=dtype)
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
            if cfg.export_fair_data:
                fair_dir = Path(cfg.fair_data_dir) if cfg.fair_data_dir else (outdir / "fair_splits")
                fair_path = fair_dir / f"{dataset}_seed{seed}_{split.split_id}.npz"
                save_fair_split(
                    split,
                    fair_path,
                    seed=seed,
                    train_frac=cfg.train_frac,
                    test_frac=cfg.test_frac,
                    n_train=cfg.n_train,
                    n_test=cfg.n_test,
                )
            if cfg.verbose:
                print(f"dataset={dataset} seed={seed} n_train={split.X_train.shape[0]} n_test={split.X_test.shape[0]} d={split.d}")
            for method in cfg.methods:
                if cfg.verbose:
                    print(f"  running {method} ...", flush=True)
                row = _run_one_method(cfg, split, method, seed, device)
                rows.append(row)
                if cfg.verbose:
                    print(f"  {method}: normalized_rmse={row.normalized_energy_rmse_per_atom} raw_rmse={row.raw_energy_rmse_per_atom} time={row.wall_time_sec} mem={row.peak_mem_gb} status={row.status}")
                pd.DataFrame([asdict(r) for r in rows]).to_csv(outdir / cfg.out_csv_name, index=False)
    return rows

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MD22 GP regression with observed gradients.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--methods", type=str, default=None, help="Comma-separated method override.")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset override.")
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--kernel", type=str, default=None, choices=["rbf", "matern52"])
    p.add_argument("--m", type=int, default=None)
    p.add_argument("--sigma-f", "--sigma_f", dest="sigma_f", type=float, default=None, help="Function/value noise variance.")
    p.add_argument("--sigma-g", "--sigma_g", dest="sigma_g", type=float, default=None, help="Gradient noise variance.")
    p.add_argument("--external-results-csv", type=str, default=None)
    p.add_argument("--no-export-fair-data", action="store_true")
    p.add_argument("--fair-data-dir", type=str, default=None)
    p.add_argument("--allow-nonstrict-external", action="store_true")
    p.add_argument("--lengthscale", type=float, default=None, help="Explicit isotropic lengthscale override.")
    p.add_argument("--lengthscale-init", type=str, default=None, choices=["median", "one"])
    p.add_argument("--learn-lengthscale", dest="learn_lengthscale", action="store_true", default=None)
    p.add_argument("--no-learn-lengthscale", dest="learn_lengthscale", action="store_false")
    p.add_argument("--use-ard", dest="use_ard", action="store_true", default=None)
    p.add_argument("--no-use-ard", dest="use_ard", action="store_false")
    p.add_argument("--exact-max-train", type=int, default=None, help="Maximum n_train allowed for dense Exact GP before skipping.")
    p.add_argument("--exact-train-epochs", type=int, default=None)
    p.add_argument("--exact-train-steps", type=int, default=None)
    p.add_argument("--exact-lr", "--exact_lr", dest="exact_lr", type=float, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--tera-train-steps", type=int, default=None)
    p.add_argument("--tera-train-epochs", type=int, default=None)
    p.add_argument("--tera-graph-refresh-epochs", type=int, default=None)
    p.add_argument("--tera-batch-size", type=int, default=None)
    p.add_argument("--tera-lr", type=float, default=None)
    p.add_argument("--tera-gradient-noise-model", choices=["iid", "scaled"], default=None)
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.methods:
        cfg.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.datasets:
        cfg.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.n_train is not None:
        cfg.n_train = args.n_train
    if args.n_test is not None:
        cfg.n_test = args.n_test
    if args.device is not None:
        cfg.device = args.device
    if args.kernel is not None:
        cfg.kernel = args.kernel
    if args.m is not None:
        cfg.m = args.m
    if args.sigma_f is not None:
        cfg.sigma_f = args.sigma_f
    if args.sigma_g is not None:
        cfg.sigma_g = args.sigma_g
    if args.external_results_csv is not None:
        cfg.external_results_csv = args.external_results_csv
    if args.no_export_fair_data:
        cfg.export_fair_data = False
    if args.fair_data_dir is not None:
        cfg.fair_data_dir = args.fair_data_dir
    if args.allow_nonstrict_external:
        cfg.strict_external_fairness = False
    if args.lengthscale is not None:
        cfg.lengthscale = args.lengthscale
    if args.lengthscale_init is not None:
        cfg.lengthscale_init = args.lengthscale_init
    if args.learn_lengthscale is not None:
        cfg.exact_learn_lengthscale = args.learn_lengthscale
        cfg.tera_learn_lengthscale = args.learn_lengthscale
    if getattr(args, "use_ard", None) is not None:
        cfg.use_ard = bool(args.use_ard)
    if args.exact_max_train is not None:
        cfg.exact_max_train = args.exact_max_train
    if args.exact_train_epochs is not None:
        cfg.exact_train_epochs = args.exact_train_epochs
    if args.exact_train_steps is not None:
        cfg.exact_train_steps = args.exact_train_steps
    if args.exact_lr is not None:
        cfg.exact_lr = args.exact_lr
    if args.verbose:
        cfg.verbose = True
    if args.tera_train_steps is not None:
        cfg.tera_train_steps = args.tera_train_steps
    if args.tera_train_epochs is not None:
        cfg.tera_train_epochs = args.tera_train_epochs
    if args.tera_graph_refresh_epochs is not None:
        cfg.tera_graph_refresh_epochs = args.tera_graph_refresh_epochs
    if args.tera_batch_size is not None:
        cfg.tera_batch_size = args.tera_batch_size
    if args.tera_lr is not None:
        cfg.tera_lr = args.tera_lr
    if args.tera_gradient_noise_model is not None:
        cfg.tera_gradient_noise_model = args.tera_gradient_noise_model
    rows = run(cfg, args.outdir)
    print(f"Wrote {len(rows)} rows to {Path(args.outdir) / cfg.out_csv_name}")

if __name__ == "__main__":
    main()
