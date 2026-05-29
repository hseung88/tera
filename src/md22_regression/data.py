from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import hashlib
import io
import json
import zipfile

import numpy as np
import torch

MD22_FILES: Mapping[str, str] = {
    "Ac-Ala3-NHMe": "md22_Ac-Ala3-NHMe.npz",
    "DHA": "md22_DHA.npz",
    "stachyose": "md22_stachyose.npz",
    "AT-AT": "md22_AT-AT.npz",
    "AT-AT-CG-CG": "md22_AT-AT-CG-CG.npz",
    "buckyball-catcher": "md22_buckyball-catcher.npz",
    "double-walled-nanotube": "md22_double-walled_nanotube.npz",
}

MD22_ATOMS: Mapping[str, int] = {
    "Ac-Ala3-NHMe": 42,
    "DHA": 56,
    "stachyose": 87,
    "AT-AT": 60,
    "AT-AT-CG-CG": 118,
    "buckyball-catcher": 148,
    "double-walled-nanotube": 370,
}

@dataclass(slots=True)
class MD22Raw:
    name: str
    R: torch.Tensor
    E: torch.Tensor
    F: torch.Tensor
    z: torch.Tensor | None
    n_atoms: int

    @property
    def n(self) -> int:
        return int(self.E.shape[0])

    @property
    def d(self) -> int:
        return int(self.n_atoms * 3)

@dataclass(slots=True)
class EnergyForceScaler:
    energy_mean: torch.Tensor
    energy_std: torch.Tensor
    x_scale: float

    def transform(self, R: torch.Tensor, E: torch.Tensor, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return X, y, g for GP training.

        The scalar target is y=(-E+mean(E))/std(E), so the force F=-grad_R E
        is the gradient of -E. If X=R/x_scale, the chain rule gives
        grad_X y = x_scale * F / std(E).
        """
        X = R.reshape(R.shape[0], -1) / self.x_scale
        y = (-E + self.energy_mean) / self.energy_std
        g = self.x_scale * F.reshape(F.shape[0], -1) / self.energy_std
        return X.contiguous(), y.contiguous(), g.contiguous()

    def inverse_energy(self, y: torch.Tensor) -> torch.Tensor:
        return -self.energy_std * y + self.energy_mean

    def inverse_force_from_input_gradient(self, g: torch.Tensor) -> torch.Tensor:
        return self.energy_std * g / self.x_scale

@dataclass(slots=True)
class MD22Split:
    name: str
    preprocessing_version: str
    split_id: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    g_train: torch.Tensor
    E_train: torch.Tensor
    F_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    g_test: torch.Tensor
    E_test: torch.Tensor
    F_test: torch.Tensor
    scaler: EnergyForceScaler
    n_atoms: int
    train_indices: torch.Tensor
    test_indices: torch.Tensor

    @property
    def d(self) -> int:
        return int(self.X_train.shape[1])

def _resolve_md22_path(data_dir: str | Path, name: str) -> Path:
    data_dir = Path(data_dir)
    candidates = []
    if name in MD22_FILES:
        candidates.append(data_dir / MD22_FILES[name])
    candidates += [
        data_dir / f"{name}.npz",
        data_dir / f"md22_{name}.npz",
        data_dir / name / f"{name}.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find MD22 file for dataset {name!r}. Tried:\n{tried}")

def load_md22_raw(data_dir: str | Path, name: str, *, device: torch.device, dtype: torch.dtype) -> MD22Raw:
    path = _resolve_md22_path(data_dir, name)
    arr = np.load(path)
    required = {"R", "E", "F"}
    missing = required.difference(arr.files)
    if missing:
        raise KeyError(f"{path} is missing required MD22 key(s): {sorted(missing)}")
    R = torch.as_tensor(arr["R"], dtype=dtype, device=device)
    E = torch.as_tensor(arr["E"].reshape(-1), dtype=dtype, device=device)
    F = torch.as_tensor(arr["F"], dtype=dtype, device=device)
    z = torch.as_tensor(arr["z"], dtype=torch.long, device=device) if "z" in arr.files else None
    n_atoms = int(R.shape[1])
    if name in MD22_ATOMS and MD22_ATOMS[name] != n_atoms:
        raise ValueError(f"Unexpected atom count for {name}: expected {MD22_ATOMS[name]}, got {n_atoms}")
    return MD22Raw(name=name, R=R, E=E, F=F, z=z, n_atoms=n_atoms)

def make_split(
    raw: MD22Raw,
    *,
    seed: int,
    train_frac: float,
    test_frac: float,
    n_train: int | None,
    n_test: int | None,
    x_scale: float,
    preprocessing_version: str = "md22_v1_neg_energy_train_energy_std_chain_rule",
) -> MD22Split:
    if not (0.0 < train_frac < 1.0) and n_train is None:
        raise ValueError("train_frac must be in (0,1) unless n_train is given")
    if not (0.0 < test_frac < 1.0) and n_test is None:
        raise ValueError("test_frac must be in (0,1) unless n_test is given")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(raw.n, generator=gen, device=torch.device('cpu')).to(raw.E.device)
    n_train_eff = int(n_train) if n_train is not None else int(round(train_frac * raw.n))
    n_test_eff = int(n_test) if n_test is not None else int(round(test_frac * raw.n))
    if n_train_eff + n_test_eff > raw.n:
        raise ValueError(f"Requested n_train+n_test={n_train_eff+n_test_eff} exceeds dataset size {raw.n}")
    train_idx = perm[:n_train_eff].contiguous()
    test_idx = perm[n_train_eff:n_train_eff + n_test_eff].contiguous()

    E_train = raw.E[train_idx]
    scaler = EnergyForceScaler(energy_mean=E_train.mean(), energy_std=E_train.std().clamp_min(raw.E.new_tensor(1e-12)), x_scale=float(x_scale))
    split_id = compute_split_id(
        dataset=raw.name,
        train_indices=train_idx,
        test_indices=test_idx,
        energy_mean=scaler.energy_mean,
        energy_std=scaler.energy_std,
        x_scale=float(x_scale),
        preprocessing_version=preprocessing_version,
    )
    X_all, y_all, g_all = scaler.transform(raw.R, raw.E, raw.F)
    return MD22Split(
        name=raw.name,
        preprocessing_version=preprocessing_version,
        split_id=split_id,
        X_train=X_all[train_idx],
        y_train=y_all[train_idx],
        g_train=g_all[train_idx],
        E_train=raw.E[train_idx],
        F_train=raw.F[train_idx].reshape(n_train_eff, -1),
        X_test=X_all[test_idx],
        y_test=y_all[test_idx],
        g_test=g_all[test_idx],
        E_test=raw.E[test_idx],
        F_test=raw.F[test_idx].reshape(n_test_eff, -1),
        scaler=scaler,
        n_atoms=raw.n_atoms,
        train_indices=train_idx,
        test_indices=test_idx,
    )

def _tensor_bytes(t: torch.Tensor) -> bytes:
    arr = t.detach().to(device="cpu").contiguous().numpy()
    return arr.tobytes()

def compute_split_id(
    *,
    dataset: str,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    energy_mean: torch.Tensor,
    energy_std: torch.Tensor,
    x_scale: float,
    preprocessing_version: str,
) -> str:
    """Stable identifier for a fair MD22 split and preprocessing choice."""
    h = hashlib.sha256()
    meta = {
        "dataset": dataset,
        "preprocessing_version": preprocessing_version,
        "x_scale": float(x_scale),
        "energy_mean": float(energy_mean.detach().cpu()),
        "energy_std": float(energy_std.detach().cpu()),
    }
    h.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    h.update(_tensor_bytes(train_indices.to(dtype=torch.long)))
    h.update(_tensor_bytes(test_indices.to(dtype=torch.long)))
    return h.hexdigest()[:16]

def split_metadata(split: MD22Split, *, seed: int, train_frac: float, test_frac: float, n_train: int | None, n_test: int | None) -> dict[str, Any]:
    return {
        "dataset": split.name,
        "seed": int(seed),
        "split_id": split.split_id,
        "preprocessing_version": split.preprocessing_version,
        "target_convention": "y=(-E+mean_train_E)/std_train_E",
        "gradient_convention": "g=x_scale*F/std_train_E with F=-grad_R E and X=R/x_scale",
        "x_scale": float(split.scaler.x_scale),
        "energy_mean_train": float(split.scaler.energy_mean.detach().cpu()),
        "energy_std_train": float(split.scaler.energy_std.detach().cpu()),
        "train_frac": float(train_frac),
        "test_frac": float(test_frac),
        "n_train_requested": None if n_train is None else int(n_train),
        "n_test_requested": None if n_test is None else int(n_test),
        "n_train": int(split.X_train.shape[0]),
        "n_test": int(split.X_test.shape[0]),
        "d": int(split.d),
        "n_atoms": int(split.n_atoms),
    }

def _as_plain_numpy_array(value: Any) -> np.ndarray:
    """Convert tensors and scalar metadata to plain NumPy arrays for NPZ export."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return np.array(arr, copy=True)

def _write_npz_compressed(path: Path, arrays: dict[str, Any]) -> None:
    """Write an NPZ file without using np.savez_compressed dispatch."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, value in arrays.items():
            arr = _as_plain_numpy_array(value)
            buf = io.BytesIO()
            np.lib.format.write_array(buf, arr, allow_pickle=False)
            zf.writestr(f"{name}.npy", buf.getvalue())

def save_fair_split(split: MD22Split, path: str | Path, *, seed: int, train_frac: float, test_frac: float, n_train: int | None, n_test: int | None) -> Path:
    """Export the canonical preprocessed split used by all fair MD22 baselines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = split_metadata(split, seed=seed, train_frac=train_frac, test_frac=test_frac, n_train=n_train, n_test=n_test)

    arrays = {
        "X_train": split.X_train,
        "y_train": split.y_train,
        "g_train": split.g_train,
        "E_train": split.E_train,
        "F_train": split.F_train,
        "X_test": split.X_test,
        "y_test": split.y_test,
        "g_test": split.g_test,
        "E_test": split.E_test,
        "F_test": split.F_test,
        "train_indices": split.train_indices,
        "test_indices": split.test_indices,
        "energy_mean": float(split.scaler.energy_mean.detach().cpu()),
        "energy_std": float(split.scaler.energy_std.detach().cpu()),
        "x_scale": float(split.scaler.x_scale),
        "n_atoms": int(split.n_atoms),
        "d": int(split.d),
        "split_id": split.split_id,
        "preprocessing_version": split.preprocessing_version,
        "metadata_json": json.dumps(meta, sort_keys=True),
    }
    _write_npz_compressed(path, arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return path
