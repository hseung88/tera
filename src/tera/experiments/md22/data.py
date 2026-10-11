from __future__ import annotations

import hashlib
import io
import zipfile
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path as path_type
from typing import Any as any_type, Mapping as mapping_type

import numpy as np
import torch

md22_files: mapping_type[str, str] = {
    "Ac-Ala3-NHMe": "md22_Ac-Ala3-NHMe.npz",
    "DHA": "md22_DHA.npz",
    "stachyose": "md22_stachyose.npz",
    "AT-AT": "md22_AT-AT.npz",
    "AT-AT-CG-CG": "md22_AT-AT-CG-CG.npz",
    "buckyball-catcher": "md22_buckyball-catcher.npz",
    "double-walled-nanotube": "md22_double-walled_nanotube.npz",
}

md22_atoms: mapping_type[str, int] = {
    "Ac-Ala3-NHMe": 42,
    "DHA": 56,
    "stachyose": 87,
    "AT-AT": 60,
    "AT-AT-CG-CG": 118,
    "buckyball-catcher": 148,
    "double-walled-nanotube": 370,
}


@dataclass(slots=True)
class md22_raw:
    name: str
    r_matrix: torch.Tensor
    e: torch.Tensor
    f: torch.Tensor
    z: torch.Tensor | None
    n_atoms: int

    @property
    def n(self) -> int:
        return int(self.e.shape[0])

    @property
    def d(self) -> int:
        return int(self.n_atoms * 3)


@dataclass(slots=True)
class energy_force_scaler:
    energy_mean: torch.Tensor
    energy_std: torch.Tensor
    x_scale: float

    def transform(
        self, r_matrix: torch.Tensor, e: torch.Tensor, f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        inputs = r_matrix.reshape(r_matrix.shape[0], -1) / self.x_scale
        y = (-e + self.energy_mean) / self.energy_std
        g = self.x_scale * f.reshape(f.shape[0], -1) / self.energy_std
        return inputs.contiguous(), y.contiguous(), g.contiguous()

    def inverse_energy(self, y: torch.Tensor) -> torch.Tensor:
        return -self.energy_std * y + self.energy_mean

    def inverse_force_from_input_gradient(self, g: torch.Tensor) -> torch.Tensor:
        return self.energy_std * g / self.x_scale


@dataclass(slots=True)
class md22_split:
    name: str
    preprocessing_version: str
    split_id: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    g_train: torch.Tensor
    e_train: torch.Tensor
    f_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    g_test: torch.Tensor
    e_test: torch.Tensor
    f_test: torch.Tensor
    scaler: energy_force_scaler
    n_atoms: int
    train_indices: torch.Tensor
    test_indices: torch.Tensor
    observation_noise_fraction: float = 0.0
    energy_noise_fraction: float = 0.0
    force_noise_fraction: float = 0.0
    observation_noise_id: str = "clean"
    energy_noise_std: float = 0.0
    force_noise_std: float = 0.0
    added_function_noise_var: float = 0.0
    added_gradient_noise_var: float = 0.0

    @property
    def d(self) -> int:
        return int(self.x_train.shape[1])


def _resolve_md22_path(data_dir: str | path_type, name: str) -> path_type:
    data_dir = path_type(data_dir)
    candidates = []
    if name in md22_files:
        candidates.append(data_dir / md22_files[name])
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


def load_md22_raw(
    data_dir: str | path_type, name: str, *, device: torch.device, dtype: torch.dtype
) -> md22_raw:
    path = _resolve_md22_path(data_dir, name)
    arr = np.load(path)
    required = {"R", "E", "F"}
    missing = required.difference(arr.files)
    if missing:
        raise KeyError(f"{path} is missing required MD22 key(s): {sorted(missing)}")
    r_matrix = torch.as_tensor(arr["R"], dtype=dtype, device=device)
    e = torch.as_tensor(arr["E"].reshape(-1), dtype=dtype, device=device)
    f = torch.as_tensor(arr["F"], dtype=dtype, device=device)
    z = torch.as_tensor(arr["z"], dtype=torch.long, device=device) if "z" in arr.files else None
    n_atoms = int(r_matrix.shape[1])
    if name in md22_atoms and md22_atoms[name] != n_atoms:
        raise ValueError(
            f"Unexpected atom count for {name}: expected {md22_atoms[name]}, got {n_atoms}"
        )
    return md22_raw(name=name, r_matrix=r_matrix, e=e, f=f, z=z, n_atoms=n_atoms)


def make_split(
    raw: md22_raw,
    *,
    seed: int,
    train_frac: float,
    test_frac: float,
    n_train: int | None,
    n_test: int | None,
    x_scale: float,
    preprocessing_version: str = "md22_v1_neg_energy_train_energy_std_chain_rule",
) -> md22_split:
    if not (0.0 < train_frac < 1.0) and n_train is None:
        raise ValueError("train_frac must be in (0,1) unless n_train is given")
    if not (0.0 < test_frac < 1.0) and n_test is None:
        raise ValueError("test_frac must be in (0,1) unless n_test is given")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(raw.n, generator=gen, device=torch.device("cpu")).to(raw.e.device)
    n_train_eff = int(n_train) if n_train is not None else int(round(train_frac * raw.n))
    n_test_eff = int(n_test) if n_test is not None else int(round(test_frac * raw.n))
    if n_train_eff + n_test_eff > raw.n:
        raise ValueError(
            f"Requested n_train+n_test={n_train_eff + n_test_eff} exceeds dataset size {raw.n}"
        )
    train_idx = perm[:n_train_eff].contiguous()
    test_idx = perm[n_train_eff : n_train_eff + n_test_eff].contiguous()

    e_train = raw.e[train_idx]
    scaler = energy_force_scaler(
        energy_mean=e_train.mean(),
        energy_std=e_train.std().clamp_min(raw.e.new_tensor(1e-12)),
        x_scale=float(x_scale),
    )
    split_id = compute_split_id(
        dataset=raw.name,
        train_indices=train_idx,
        test_indices=test_idx,
        energy_mean=scaler.energy_mean,
        energy_std=scaler.energy_std,
        x_scale=float(x_scale),
        preprocessing_version=preprocessing_version,
    )
    x_all, y_all, g_all = scaler.transform(raw.r_matrix, raw.e, raw.f)
    return md22_split(
        name=raw.name,
        preprocessing_version=preprocessing_version,
        split_id=split_id,
        x_train=x_all[train_idx],
        y_train=y_all[train_idx],
        g_train=g_all[train_idx],
        e_train=raw.e[train_idx],
        f_train=raw.f[train_idx].reshape(n_train_eff, -1),
        x_test=x_all[test_idx],
        y_test=y_all[test_idx],
        g_test=g_all[test_idx],
        e_test=raw.e[test_idx],
        f_test=raw.f[test_idx].reshape(n_test_eff, -1),
        scaler=scaler,
        n_atoms=raw.n_atoms,
        train_indices=train_idx,
        test_indices=test_idx,
    )


def add_observation_noise(
    split: md22_split,
    fraction: float = 0.0,
    *,
    seed: int,
    energy_fraction: float | None = None,
    force_fraction: float | None = None,
) -> md22_split:
    """Perturb training labels only; keep clean scaling and all test labels fixed.

    Energy and pooled force scales are estimated from clean training observations.
    Each force coordinate receives independent noise with the same variance,
    matching the common iid gradient-noise model. Draws are shared across m,
    methods, and fractions for the same split and seed, on CPU in float64.
    """
    from tera.experiments.md22.observation_noise import noise_fractions

    energy_fraction, force_fraction = noise_fractions(fraction, energy_fraction, force_fraction)
    if split.observation_noise_id != "clean":
        raise ValueError("Observation noise must be added to a clean split exactly once")
    if energy_fraction == 0 and force_fraction == 0:
        return split
    signature = dict(split_id=split.split_id, seed=int(seed), scheme="md22_training_iid_v1")
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little"))
    energy_std = energy_fraction * float(split.e_train.double().std())
    force_std = force_fraction * float(split.f_train.double().std())
    energy_draw = torch.randn(split.e_train.shape, generator=generator, dtype=torch.float64)
    force_draw = torch.randn(split.f_train.shape, generator=generator, dtype=torch.float64)
    energies = split.e_train.double() + energy_std * energy_draw.to(split.e_train.device)
    forces = split.f_train.double() + force_std * force_draw.to(split.f_train.device)
    scale = split.scaler.energy_std.double()
    y = ((-energies + split.scaler.energy_mean.double()) / scale).to(split.y_train.dtype)
    g = (split.scaler.x_scale * forces / scale).to(split.g_train.dtype)
    condition = (
        dict(fraction=energy_fraction)
        if energy_fraction == force_fraction
        else dict(energy_fraction=energy_fraction, force_fraction=force_fraction)
    )
    noise_id = hashlib.sha256(
        json.dumps(dict(signature, **condition), sort_keys=True).encode()
    ).hexdigest()[:16]
    return replace(
        split,
        e_train=energies,
        f_train=forces,
        y_train=y.contiguous() if energy_fraction else split.y_train,
        g_train=g.contiguous() if force_fraction else split.g_train,
        observation_noise_fraction=energy_fraction
        if energy_fraction == force_fraction
        else math.nan,
        energy_noise_fraction=energy_fraction,
        force_noise_fraction=force_fraction,
        observation_noise_id=noise_id,
        energy_noise_std=energy_std,
        force_noise_std=force_std,
        added_function_noise_var=(energy_std / float(scale)) ** 2,
        added_gradient_noise_var=(split.scaler.x_scale * force_std / float(scale)) ** 2,
    )


def baseline_targets(split: md22_split, *, joint_scale: bool):
    """tera adapter normalization, computed directly from physical training labels."""
    mean = split.scaler.energy_mean.double()
    y_train = -split.e_train.double() + mean
    y_test = -split.e_test.double() + mean
    g_train = split.scaler.x_scale * split.f_train.double()
    g_test = split.scaler.x_scale * split.f_test.double()
    if joint_scale:
        scale = torch.cat([y_train.flatten(), g_train.flatten()]).std().clamp_min(1e-12)
    else:
        scale = split.scaler.energy_std.double()
    if not bool(torch.isfinite(scale)):
        raise ValueError("Baseline normalization scale is not finite.")
    return y_train / scale, g_train / scale, y_test / scale, g_test / scale, scale


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


def split_metadata(split: md22_split, *, seed: int, train_frac: float, test_frac: float, n_train: int | None, n_test: int | None) -> dict[str, any_type]:
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
        "n_train": int(split.x_train.shape[0]),
        "n_test": int(split.x_test.shape[0]),
        "d": int(split.d),
        "n_atoms": int(split.n_atoms),
    }

def _as_plain_numpy_array(value: any_type) -> np.ndarray:
    """Convert tensors and scalar metadata to plain NumPy arrays for NPZ export."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return np.array(arr, copy=True)

def _write_npz_compressed(path: path_type, arrays: dict[str, any_type]) -> None:
    """Write an NPZ file without using np.savez_compressed dispatch."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, value in arrays.items():
            arr = _as_plain_numpy_array(value)
            buf = io.BytesIO()
            np.lib.format.write_array(buf, arr, allow_pickle=False)
            zf.writestr(f"{name}.npy", buf.getvalue())

def save_fair_split(split: md22_split, path: str | path_type, *, seed: int, train_frac: float, test_frac: float, n_train: int | None, n_test: int | None) -> path_type:
    """Export the canonical preprocessed split used by all fair MD22 baselines."""
    path = path_type(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = split_metadata(split, seed=seed, train_frac=train_frac, test_frac=test_frac, n_train=n_train, n_test=n_test)

    arrays = {
        "X_train": split.x_train,
        "y_train": split.y_train,
        "g_train": split.g_train,
        "E_train": split.e_train,
        "F_train": split.f_train,
        "X_test": split.x_test,
        "y_test": split.y_test,
        "g_test": split.g_test,
        "E_test": split.e_test,
        "F_test": split.f_test,
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
