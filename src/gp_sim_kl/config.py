from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

KernelName = Literal["rbf", "matern52"]
MethodName = Literal[
    "Exact GP",
    "Exact dGP",
    "Exact dGP-deRoos",
    "Vecchia GP",
    "Vecchia dGP",
    "Vecchia dGP-deRoos",
    "TERA",
]

METHOD_EXACT_GP: MethodName = "Exact GP"
METHOD_EXACT_DGP: MethodName = "Exact dGP"
METHOD_EXACT_DGP_DEROOS: MethodName = "Exact dGP-deRoos"
METHOD_VECCHIA_GP: MethodName = "Vecchia GP"
METHOD_VECCHIA_DGP: MethodName = "Vecchia dGP"
METHOD_VECCHIA_DGP_DEROOS: MethodName = "Vecchia dGP-deRoos"
METHOD_TERA: MethodName = "TERA"

METHOD_ORDER: tuple[MethodName, ...] = (
    METHOD_EXACT_GP,
    METHOD_EXACT_DGP,
    METHOD_EXACT_DGP_DEROOS,
    METHOD_VECCHIA_GP,
    METHOD_VECCHIA_DGP,
    METHOD_VECCHIA_DGP_DEROOS,
    METHOD_TERA,
)
ALL_METHOD_NAMES = set(METHOD_ORDER)

def validate_methods(methods: list[str]) -> list[MethodName]:
    bad = [m for m in methods if m not in ALL_METHOD_NAMES]
    if bad:
        allowed = ", ".join(METHOD_ORDER)
        raise ValueError(f"Unknown method name(s): {bad}. Allowed methods: {allowed}")
    normalized: list[MethodName] = []
    for method in methods:
        name = method
        if name not in normalized:
            normalized.append(name)
    return normalized

@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    kernel: KernelName
    use_ard: bool
    lengthscale: float | list[float] | None
    outputscale: float
    sigma_f: float
    sigma_g: float
    n_train: int
    n_eval: int
    m: int | str
    d_values: list[int]
    repeats: int
    seed: int
    device: str
    dtype: str
    methods: list[MethodName]
    n_train_values: list[int] | None = None
    m_values: list[int] | None = None
    target_median_correlation: float | None = None
    design: Literal["sobol", "random"] = "sobol"
    sampling: Literal["dense", "deroos", "zeros"] = "dense"
    dense_sampling_max_obs_dim: int = 20000
    deroos_sampling_max_n2: int = 6400
    exact_dense_max_d: int = 100
    exact_dense_max_obs_dim: int = 25000
    exact_deroos_max_n: int = 100
    vecchia_dense_max_local_dim: int = 5000

def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "methods" in raw:
        raw["methods"] = validate_methods(raw["methods"])
    return ExperimentConfig(**raw)
