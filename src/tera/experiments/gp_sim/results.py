from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path as path_type
from typing import Sequence as sequence_type

import pandas as pd


@dataclass(slots=True)
class experiment_row:
    experiment_name: str
    repeat: int
    kernel: str
    d: int
    n_train: int
    n_eval: int
    m: int
    method: str
    avg_marginal_kl: float
    sum_marginal_kl: float
    maxabs_mean_diff_to_exact_dgp: float
    maxabs_var_diff_to_exact_dgp: float
    maxabs_mean_diff_to_vecchia_deroos_reference: float
    maxabs_var_diff_to_vecchia_deroos_reference: float
    build_time_sec: float
    predict_time_sec: float
    peak_mem_gb: float
    status: str = "ok"


def save_rows(rows: sequence_type[experiment_row], path: str | path_type) -> None:
    df = pd.DataFrame([asdict(r) for r in rows])
    path_type(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
