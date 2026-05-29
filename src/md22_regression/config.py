from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

@dataclass(slots=True)
class MD22Config:
    experiment_name: str = "md22_regression"
    data_dir: str = "data/md22"
    datasets: list[str] = field(default_factory=lambda: ["DHA"])
    methods: list[str] = field(default_factory=lambda: ["Exact GP", "DSoftKI", "DDSVGP", "TERA"])
    seeds: list[int] = field(default_factory=lambda: [6535, 8830, 92357])
    train_frac: float = 0.9
    test_frac: float = 0.1
    n_train: int | None = None
    n_test: int | None = None
    device: str = "cuda"
    dtype: str = "float64"
    kernel: str = "rbf"
    outputscale: float = 1.0

    sigma_f: float = 1e-3
    sigma_g: float = 1e-3
    lengthscale: float | list[float] | None = None
    lengthscale_init: str = "median"
    lengthscale_init_max_points: int = 2048
    use_ard: bool = False
    x_scale: float = 3.0
    preprocessing_version: str = "md22_v1_neg_energy_train_energy_std_chain_rule"
    export_fair_data: bool = True
    fair_data_dir: str | None = None
    strict_external_fairness: bool = True
    m: int = 20

    exact_max_train: int = 100000
    external_results_csv: str | None = None
    external_repo: str | None = None
    out_csv_name: str = "results.csv"
    verbose: bool = False
    log_training_curves: bool = False
    curve_log_every: int = 5

    exact_train_epochs: int = 0
    exact_train_steps: int = 50

    exact_lr: float | None = None
    exact_lr_default: float = 0.01
    exact_lr_by_dataset: dict[str, float] = field(default_factory=lambda: {"double-walled-nanotube": 0.005})
    exact_weight_decay: float = 0.0
    exact_learn_lengthscale: bool = True
    exact_learn_outputscale: bool = True
    exact_learn_sigma_f: bool = True
    exact_min_sigma_f: float = 1e-6
    exact_log_every: int = 0

    tera_train_steps: int = 0
    tera_train_epochs: int = 1

    tera_graph_refresh_epochs: int = 0
    tera_batch_size: int = 256
    tera_lr: float = 0.01
    tera_weight_decay: float = 0.0
    tera_learn_lengthscale: bool = True
    tera_learn_outputscale: bool = True
    tera_learn_sigma_f: bool = True
    tera_learn_sigma_g: bool = True
    tera_min_sigma_f: float = 1e-6
    tera_min_sigma_g: float = 0.0
    tera_log_every: int = 0
    tera_gradient_noise_model: str = "iid"

def load_config(path: str | Path) -> MD22Config:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return MD22Config(**raw)
