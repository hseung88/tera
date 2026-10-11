from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as path_type
from typing import Literal as literal_type

import yaml

kernel_name_type = literal_type["rbf", "matern52"]
method_name_type = literal_type[
    'exact gp',
    'exact dgp',
    'exact dgp-deroos',
    'vecchia gp',
    'vecchia dgp',
    'vecchia dgp-deroos',
    'tera',
]

method_exact_gp: method_name_type = 'exact gp'
method_exact_dgp: method_name_type = 'exact dgp'
method_exact_dgp_deroos: method_name_type = 'exact dgp-deroos'
method_vecchia_gp: method_name_type = 'vecchia gp'
method_vecchia_dgp: method_name_type = 'vecchia dgp'
method_vecchia_dgp_deroos: method_name_type = 'vecchia dgp-deroos'
method_tera: method_name_type = 'tera'

method_order: tuple[method_name_type, ...] = (
    method_exact_gp,
    method_exact_dgp,
    method_exact_dgp_deroos,
    method_vecchia_gp,
    method_vecchia_dgp,
    method_vecchia_dgp_deroos,
    method_tera,
)
all_method_names = set(method_order)


def validate_methods(methods: list[str]) -> list[method_name_type]:
    from tera.methods.names import normalize_method_name

    methods = [normalize_method_name(m) for m in methods]
    bad = [m for m in methods if m not in all_method_names]
    if bad:
        allowed = ", ".join(method_order)
        raise ValueError(f"Unknown method name(s): {bad}. Allowed methods: {allowed}")
    normalized: list[method_name_type] = []
    for method in methods:
        name = method
        if name not in normalized:
            normalized.append(name)
    return normalized


@dataclass(slots=True)
class experiment_config:
    experiment_name: str
    kernel: kernel_name_type
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
    methods: list[method_name_type]
    prediction_batch_size: int = 32
    n_train_values: list[int] | None = None
    m_values: list[int] | None = None
    target_median_correlation: float | None = None
    design: literal_type["sobol", "random"] = "sobol"
    sampling: literal_type["dense", "deroos", "zeros"] = "dense"
    dense_sampling_max_obs_dim: int = 20000
    deroos_sampling_max_n2: int = 6400
    exact_dense_max_d: int = 100
    exact_dense_max_obs_dim: int = 25000
    exact_deroos_max_n: int = 100
    vecchia_dense_max_local_dim: int = 5000


def load_config(path: str | path_type) -> experiment_config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "methods" in raw:
        raw["methods"] = validate_methods(raw["methods"])
    return experiment_config(**raw)
