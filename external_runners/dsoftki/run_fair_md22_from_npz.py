
"""Run DSoftKI or DDSVGP on a canonical MD22 fair split.

This script intentionally keeps the released DSoftKI/DDSVGP model and training
code intact. It replaces only the data-loading layer: instead of constructing an
MD22 dataset inside this repository, it reads a preprocessed NPZ exported by the
TERA/scalable_dgp repository.

Expected NPZ keys:
    X_train, y_train, g_train, X_test, y_test, g_test,
    E_test, energy_mean, energy_std, x_scale, n_atoms, d,
    split_id, preprocessing_version, metadata_json

By default, DSoftKI/DDSVGP are trained with the released MD22 normalization: a
joint value-derivative training scale. The reported final metrics are normalized
energy RMSE per atom and raw energy RMSE per atom. Use --external-energy-scale
only for diagnostics that force the canonical energy-only scaling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception as exc:
    raise RuntimeError("This script requires PyTorch in the DSoftKI environment.") from exc

try:
    from omegaconf import OmegaConf
except Exception as exc:
    raise RuntimeError("This script requires omegaconf in the DSoftKI environment.") from exc

import gp.dsoft_ki.train as dsoftki_train
import gp.ddsvgp.train as ddsvgp_train

def my_collate_fn(batch):
    """Collate `(x, {energy, neg_force})` batches without importing `gp.util`.

    The released `gp.util.my_collate_fn` imports matplotlib at module import time.
    That is unnecessary for MD22 training and can fail on clusters with an older
    system libstdc++.
    """
    xs = [item[0] for item in batch]
    ys = []
    for _, target in batch:
        energy = target["energy"]
        if energy.ndim == 0:
            energy = energy.reshape(1)
        force = target["neg_force"].reshape(-1)
        ys.append(torch.cat([energy, force], dim=0))
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

@dataclass(frozen=True)
class FairMetadata:
    dataset: str
    seed: int
    split_id: str
    preprocessing_version: str
    n_train: int
    n_test: int
    d: int
    n_atoms: int
    x_scale: float
    energy_mean: float
    energy_std: float

class FairMD22Dataset(Dataset):
    """Dataset wrapper matching the released DSoftKI collate function.

    Each item returns `(x, {"energy": y, "neg_force": g})`. The key name
    `neg_force` is retained because the released collate function expects it.
    Here it stores the canonical observed gradient of `y=-E` with respect to the
    model coordinate `X=R/x_scale`.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, g: np.ndarray, *, dtype: torch.dtype) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            y = y.reshape(-1)
        if g.ndim != 2:
            raise ValueError(f"g must be 2D, got shape {g.shape}")
        if len(X) != len(y) or len(X) != len(g):
            raise ValueError("X, y, and g have inconsistent first dimensions")
        self.X = torch.as_tensor(X, dtype=dtype)
        self.y = torch.as_tensor(y, dtype=dtype)
        self.g = torch.as_tensor(g, dtype=dtype)
        self.dim = int(self.X.shape[1])
        self.scale = 1.0
        self.shift = 0.0

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], {"energy": self.y[idx], "neg_force": self.g[idx]}

def _as_python_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value

def _load_metadata(npz: np.lib.npyio.NpzFile, fallback_dataset: str | None, fallback_seed: int | None) -> FairMetadata:
    metadata_json = None
    if "metadata_json" in npz.files:
        raw = _as_python_scalar(npz["metadata_json"])
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        metadata_json = json.loads(str(raw))
    else:
        metadata_json = {}

    dataset = str(metadata_json.get("dataset", fallback_dataset or "unknown"))
    seed = int(metadata_json.get("seed", fallback_seed if fallback_seed is not None else -1))
    split_id = str(_as_python_scalar(npz["split_id"]))
    preprocessing_version = str(_as_python_scalar(npz["preprocessing_version"]))
    return FairMetadata(
        dataset=dataset,
        seed=seed,
        split_id=split_id,
        preprocessing_version=preprocessing_version,
        n_train=int(np.asarray(npz["X_train"]).shape[0]),
        n_test=int(np.asarray(npz["X_test"]).shape[0]),
        d=int(_as_python_scalar(npz["d"])),
        n_atoms=int(_as_python_scalar(npz["n_atoms"])),
        x_scale=float(_as_python_scalar(npz["x_scale"])),
        energy_mean=float(_as_python_scalar(npz["energy_mean"])),
        energy_std=float(_as_python_scalar(npz["energy_std"])),
    )

def _std_positive(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    ddof = 1 if arr.size > 1 else 0
    scale = float(np.std(arr, ddof=ddof))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1e-12
    return max(scale, 1e-12)

def _external_arrays_and_scale(npz: np.lib.npyio.NpzFile, meta: FairMetadata, args: argparse.Namespace):
    E_train = np.asarray(npz["E_train"], dtype=np.float64).reshape(-1)
    E_test = np.asarray(npz["E_test"], dtype=np.float64).reshape(-1)
    F_train = np.asarray(npz["F_train"], dtype=np.float64)
    F_test = np.asarray(npz["F_test"], dtype=np.float64)

    centered_train = -E_train + float(meta.energy_mean)
    centered_test = -E_test + float(meta.energy_mean)

    grad_train_raw = F_train.reshape(F_train.shape[0], -1)
    grad_test_raw = F_test.reshape(F_test.shape[0], -1)
    if not args.external_original_force_scale:
        grad_train_raw = float(meta.x_scale) * grad_train_raw
        grad_test_raw = float(meta.x_scale) * grad_test_raw

    if args.external_joint_scale:
        scale = _std_positive(np.concatenate([centered_train.reshape(-1), grad_train_raw.reshape(-1)]))
        scale_name = "joint_value_derivative_train_scale"
        y_train = centered_train / scale
        y_test = centered_test / scale
        g_train = grad_train_raw / scale
        g_test = grad_test_raw / scale
    else:
        scale = float(meta.energy_std)
        scale_name = "train_energy_std"
        y_train = np.asarray(npz["y_train"]).reshape(-1)
        y_test = np.asarray(npz["y_test"]).reshape(-1)
        g_train = np.asarray(npz["g_train"])
        g_test = np.asarray(npz["g_test"])
        if args.external_original_force_scale:
            g_train = g_train / float(meta.x_scale)
            g_test = g_test / float(meta.x_scale)

    return y_train, y_test, g_train, g_test, scale, scale_name

def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype {name!r}")

def _dsoftki_kernel_target(kernel: str) -> str:
    if kernel == "rbf":
        return "RBFKernel"
    if kernel == "matern52":
        return "MaternKernel"
    raise ValueError(f"Unsupported DSoftKI kernel {kernel!r}.")

def _make_config(args: argparse.Namespace, meta: FairMetadata):
    if args.method == "DSoftKI":
        cfg = {
            "dataset": {"name": meta.dataset, "num_workers": args.num_workers},
            "wandb": {"watch": False, "group": "fair_md22", "entity": "", "project": ""},
            "model": {
                "name": "dsoftki",
                "kernel": {"_target_": _dsoftki_kernel_target(args.kernel), "lengthscale": args.lengthscale, "nu": 2.5, "ard_num_dims": None},
                "embed_dim": -1,
                "hidden_dim": 64,
                "per_interp_T": True,
                "min_T": 0.00005,
                "use_dot": True,
                "grad_only": False,
                "lengthscale": args.lengthscale,
                "use_ard": args.use_ard,
                "use_scale": True,
                "num_interp": args.num_inducing,
                "interp_init": "kmeans",
                "noise": args.noise,
                "deriv_noise": args.deriv_noise,
                "learn_noise": args.learn_noise,
                "solver": args.solver,
                "cg_tolerance": args.cg_tolerance,
                "mll_approx": args.mll_approx,
                "fit_chunk_size": args.fit_chunk_size,
                "use_qr": args.use_qr,
                "dtype": args.dtype,
                "device": args.device,
                "fit_device": args.fit_device,
                "skip_nll": True,
            },
            "training": {
                "seed": args.seed,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "embed_lr": args.embed_lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "curve_log_every": args.curve_log_every if args.log_training_curves else 0,
            },
        }
    elif args.method == "DDSVGP":
        cfg = {
            "dataset": {"name": meta.dataset, "num_workers": args.num_workers},
            "wandb": {"watch": False, "group": "fair_md22", "entity": "", "project": ""},
            "model": {
                "name": "ddsvgp",
                "kernel": {"_target_": "RBFKernelDirectionalGrad", "ard_num_dims": None},
                "lengthscale": args.lengthscale,
                "use_scale": True,
                "use_ard": False,
                "num_inducing": args.num_inducing,
                "num_directions": args.num_directions,
                "induce_init": "kmeans",
                "noise": args.noise,
                "dtype": args.dtype,
                "device": args.device,
                "mll_type": args.mll_type,
            },
            "training": {
                "seed": args.seed,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "gamma": args.gamma,
                "lr_sched": None,
                "curve_log_every": args.curve_log_every if args.log_training_curves else 0,
            },
        }
    else:
        raise ValueError(f"Unsupported method {args.method}")
    return OmegaConf.create(cfg)

def _predict_y_mean_dsoftki(model, X: torch.Tensor, *, device: str, batch_size: int) -> torch.Tensor:
    model.eval()
    out = []
    loader = DataLoader(X, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            preds = model.pred(xb)
            out.append(preds[: xb.shape[0]].detach().cpu())
    return torch.cat(out, dim=0).reshape(-1)

def _predict_y_mean_ddsvgp(model, likelihood, X: torch.Tensor, *, device: str, batch_size: int, num_directions: int) -> torch.Tensor:
    model.eval()
    likelihood.eval()
    out = []
    dim = int(X.shape[1])
    loader = DataLoader(X, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            derivative_directions = torch.eye(dim, device=device, dtype=xb.dtype)[:num_directions]
            kwargs = {"derivative_directions": derivative_directions.repeat(xb.shape[0], 1)}
            preds = likelihood(model(xb, **kwargs))
            out.append(preds.mean[:: num_directions + 1].detach().cpu())
    return torch.cat(out, dim=0).reshape(-1)

def _normalized_energy_rmse_per_atom(y_pred: torch.Tensor, y_test: np.ndarray, *, meta: FairMetadata) -> float:
    """RMSE in the external normalized energy scale, divided by atom count."""
    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    y_true_np = np.asarray(y_test).reshape(-1)
    return float(np.sqrt(np.mean((y_pred_np - y_true_np) ** 2)) / meta.n_atoms)

def _raw_energy_rmse_per_atom(y_pred: torch.Tensor, E_test: np.ndarray, *, meta: FairMetadata, scale: float | None = None) -> float:
    """RMSE after inverse-transforming to physical energy units, divided by atoms."""
    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    inv_scale = float(meta.energy_std if scale is None else scale)
    E_pred = meta.energy_mean - inv_scale * y_pred_np
    E_true = np.asarray(E_test).reshape(-1)
    return float(np.sqrt(np.mean((E_pred - E_true) ** 2)) / meta.n_atoms)

def _release_curve_rmse_to_per_atom(release_normalized_rmse: float, *, meta: FairMetadata, scale: float) -> tuple[float, float]:
    """Convert released DSoftKI/DDSVGP curve RMSE to per-atom normalized/raw metrics.

    The released training-history value is a normalized scalar-energy RMSE, not
    an RMSE per atom.  Therefore both curve metrics must divide by n_atoms; the
    raw curve additionally multiplies by the normalization scale used by that
    external run.
    """
    if not math.isfinite(float(release_normalized_rmse)):
        return math.nan, math.nan
    normalized_per_atom = float(release_normalized_rmse) / float(meta.n_atoms)
    raw_per_atom = float(scale) * normalized_per_atom
    return normalized_per_atom, raw_per_atom

def _peak_mem_gb(device: str) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 3))
    return 0.0

def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "seed",
        "method",
        "split_id",
        "preprocessing_version",
        "n_train",
        "n_test",
        "d",
        "n_atoms",
        "x_scale",
        "external_original_force_scale",
        "external_joint_scale",
        "external_scale_name",
        "external_scale",
        "normalized_energy_rmse_per_atom",
        "raw_energy_rmse_per_atom",
        "wall_time_sec",
        "fit_time_sec",
        "predict_time_sec",
        "peak_mem_gb",
        "status",
        "error",
        "source_npz",
    ]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})

def _append_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "seed",
        "method",
        "split_id",
        "preprocessing_version",
        "n_train",
        "n_test",
        "d",
        "n_atoms",
        "x_scale",
        "external_joint_scale",
        "external_scale_name",
        "external_scale",
        "step",
        "normalized_energy_rmse_per_atom",
        "raw_energy_rmse_per_atom",
        "release_normalized_energy_rmse",
        "fit_time_sec",
        "status",
        "source_npz",
    ]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

def _released_md22_batch_size(d: int) -> int:
    if d >= 1000:
        return 128
    if d >= 300:
        return 256
    if d >= 180:
        return 512
    return 1024

def _released_md22_lr(d: int, method: str) -> float:
    if method == "DDSVGP":
        if d >= 1000:
            return 0.0015
        if d >= 300:
            return 0.003
        if d >= 180:
            return 0.006
        return 0.012
    if d >= 1000:
        return 0.001
    if d >= 300:
        return 0.002
    if d >= 180:
        return 0.004
    return 0.008


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, type=Path, help="Canonical fair MD22 NPZ exported by scalable_dgp.")
    parser.add_argument("--method", choices=["DSoftKI", "DDSVGP"], required=True)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--history-out-csv", default=None, type=Path)
    parser.add_argument("--dataset", default=None, help="Optional dataset name override if absent from NPZ metadata.")
    parser.add_argument("--seed", type=int, default=6535)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit-device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-inducing", type=int, default=512)
    parser.add_argument("--lengthscale", type=float, default=1.0)
    parser.add_argument("--kernel", choices=["rbf", "matern52"], default="rbf", help="DSoftKI kernel. DDSVGP is RBF-only in the released implementation.")
    parser.add_argument("--sigma-f", "--sigma_f", dest="sigma_f", type=float, default=None,
                        help="Value-noise variance. Overrides --noise.")
    parser.add_argument("--sigma-g", "--sigma_g", dest="sigma_g", type=float, default=None,
                        help="Gradient-noise variance. DDSVGP has one Gaussian likelihood noise; this argument is ignored unless it matches --sigma-f.")
    parser.add_argument(
        "--external-joint-scale",
        dest="external_joint_scale",
        action="store_true",
        default=True,
        help=(
            "Use the training joint value-derivative scale for external y and gradients. "
            "This is the default and matches the released DSoftKI/DDSVGP MD22 setup."
        ),
    )
    parser.add_argument(
        "--external-energy-scale",
        dest="external_joint_scale",
        action="store_false",
        help=(
            "Use the canonical energy-only scale for external y and gradients. "
            "This is a diagnostic mode, not the released DSoftKI/DDSVGP MD22 normalization."
        ),
    )
    parser.add_argument(
        "--external-original-force-scale",
        action="store_true",
        help=(
            "Divide canonical chain-rule-correct gradients by x_scale before fitting/prediction. "
            "Use only for released-style DSoftKI/DDSVGP reproduction diagnostics, not for the common derivative-GP protocol."
        ),
    )
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--deriv-noise", type=float, default=None)
    parser.add_argument("--learn-noise", action="store_true", default=True)
    parser.add_argument("--no-learn-noise", dest="learn_noise", action="store_false")
    parser.add_argument("--use-ard", action="store_true", default=True)
    parser.add_argument("--no-use-ard", dest="use_ard", action="store_false")
    parser.add_argument("--solver", default="cg")
    parser.add_argument("--cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--mll-approx", default="hutchinson_fallback")
    parser.add_argument("--fit-chunk-size", type=int, default=256)
    parser.add_argument("--use-qr", action="store_true", default=True)
    parser.add_argument("--no-use-qr", dest="use_qr", action="store_false")
    parser.add_argument("--embed-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-directions", type=int, default=2)
    parser.add_argument("--mll-type", choices=["ELBO", "PLL"], default="PLL")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument(
        "--log-training-curves",
        action="store_true",
        help="Evaluate and log raw energy RMSE during training. This is expensive and intended for a separate curve run.",
    )
    parser.add_argument("--curve-log-every", type=int, default=5, help="Training-step interval for RMSE curve logging.")
    args = parser.parse_args()
    if not hasattr(args, "external_original_force_scale"):
        args.external_original_force_scale = False

    if args.method == "DDSVGP" and args.kernel != "rbf":
        raise ValueError("DDSVGP in the released DSoftKI repository only supports RBFKernelDirectionalGrad. Use --kernel rbf or run DSoftKI only for Mat\u00e9rn-5/2 experiments.")

    if args.fit_device is None:
        args.fit_device = args.device

    if args.noise is None:
        args.noise = 0.001 if args.method == "DSoftKI" else 0.1
    if args.deriv_noise is None:

        args.deriv_noise = math.nan

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        if args.device.startswith("cuda"):
            device_index = 0 if args.device == "cuda" else int(args.device.split(":")[1])
            torch.cuda.set_device(device_index)
        torch.cuda.reset_peak_memory_stats(device=args.device)

    status = "ok"
    error = ""
    normalized_metric = float("nan")
    raw_metric = float("nan")
    fit_time = float("nan")
    pred_time = float("nan")
    wall_time = float("nan")
    peak_mem = 0.0

    with np.load(args.npz, allow_pickle=False) as npz:
        meta = _load_metadata(npz, args.dataset, args.seed)

        if args.sigma_f is not None:
            args.noise = float(args.sigma_f)
        if args.method == "DSoftKI":
            if args.noise is None:
                args.noise = 0.001
            args.deriv_noise = float(args.noise) * float(meta.d)
        else:
            if args.sigma_g is not None and args.noise is None:
                args.noise = float(args.sigma_g)

        if args.batch_size is None:
            args.batch_size = _released_md22_batch_size(meta.d)
        if args.lr is None:
            args.lr = _released_md22_lr(meta.d, args.method)
        if math.isnan(args.deriv_noise):
            args.deriv_noise = 0.001 * meta.d
        dtype = _dtype_from_name(args.dtype)
        y_train_ext, y_test, g_train_ext, g_test_ext, external_scale, external_scale_name = _external_arrays_and_scale(npz, meta, args)
        train_dataset = FairMD22Dataset(npz["X_train"], y_train_ext, g_train_ext, dtype=dtype)
        test_dataset = FairMD22Dataset(npz["X_test"], y_test, g_test_ext, dtype=dtype)
        E_test = np.asarray(npz["E_test"]).reshape(-1)
        X_test = torch.as_tensor(npz["X_test"], dtype=dtype)

    cfg = _make_config(args, meta)
    history_rows: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    try:
        curve_dataset = test_dataset if args.log_training_curves else None
        if args.method == "DSoftKI":
            model = dsoftki_train.train_gp(cfg, train_dataset, curve_dataset, collate_fn=my_collate_fn)
            t1 = time.perf_counter()
            y_pred = _predict_y_mean_dsoftki(model, X_test, device=args.device, batch_size=args.eval_batch_size)
            t2 = time.perf_counter()
        else:
            model, likelihood = ddsvgp_train.train_gp(cfg, train_dataset, curve_dataset, collate_fn=my_collate_fn)
            t1 = time.perf_counter()
            y_pred = _predict_y_mean_ddsvgp(
                model,
                likelihood,
                X_test,
                device=args.device,
                batch_size=args.eval_batch_size,
                num_directions=args.num_directions,
            )
            t2 = time.perf_counter()
        normalized_metric = _normalized_energy_rmse_per_atom(y_pred, y_test, meta=meta)
        raw_metric = _raw_energy_rmse_per_atom(y_pred, E_test, meta=meta, scale=external_scale)
        fit_time = t1 - t0
        pred_time = t2 - t1
        wall_time = t2 - t0
        peak_mem = _peak_mem_gb(args.device)
        cumulative_fit = 0.0
        for rec in getattr(model, "training_history", []):
            cumulative_fit += float(rec.get("epoch_time", 0.0)) + float(rec.get("fit_time", 0.0))
            release_norm_rmse = float(rec.get("normalized_energy_rmse_per_atom", math.nan))
            norm_hist, raw_hist = _release_curve_rmse_to_per_atom(release_norm_rmse, meta=meta, scale=external_scale)
            history_rows.append(
                {
                    "dataset": meta.dataset,
                    "seed": args.seed,
                    "method": args.method,
                    "split_id": meta.split_id,
                    "preprocessing_version": meta.preprocessing_version,
                    "n_train": meta.n_train,
                    "n_test": meta.n_test,
                    "d": meta.d,
                    "n_atoms": meta.n_atoms,
                    "x_scale": meta.x_scale,
                    "external_joint_scale": bool(args.external_joint_scale),
                    "external_scale_name": external_scale_name,
                    "external_scale": external_scale,
                    "step": float(rec.get("step", math.nan)),
                    "normalized_energy_rmse_per_atom": norm_hist,
                    "raw_energy_rmse_per_atom": raw_hist,
                    "release_normalized_energy_rmse": release_norm_rmse,
                    "fit_time_sec": cumulative_fit,
                    "status": status,
                    "source_npz": str(args.npz),
                }
            )
    except Exception as exc:
        status = "error"
        error = repr(exc)
        wall_time = time.perf_counter() - t0
        peak_mem = _peak_mem_gb(args.device)

    row = {
        "dataset": meta.dataset,
        "seed": args.seed,
        "method": args.method,
        "split_id": meta.split_id,
        "preprocessing_version": meta.preprocessing_version,
        "n_train": meta.n_train,
        "n_test": meta.n_test,
        "d": meta.d,
        "n_atoms": meta.n_atoms,
        "x_scale": meta.x_scale,
        "external_original_force_scale": bool(args.external_original_force_scale),
        "external_joint_scale": bool(args.external_joint_scale),
        "external_scale_name": external_scale_name,
        "external_scale": external_scale,
        "normalized_energy_rmse_per_atom": normalized_metric,
        "raw_energy_rmse_per_atom": raw_metric,
        "wall_time_sec": wall_time,
        "fit_time_sec": fit_time,
        "predict_time_sec": pred_time,
        "peak_mem_gb": peak_mem,
        "status": status,
        "error": error,
        "source_npz": str(args.npz),
    }
    _append_csv(args.out_csv, row)
    if args.history_out_csv is not None:
        _append_history_csv(args.history_out_csv, history_rows)
    print(json.dumps(row, indent=2, sort_keys=True))
    if status != "ok":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
