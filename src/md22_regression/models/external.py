from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from md22_regression.data import MD22Split
from md22_regression.models.base import Prediction

class ExternalResultsModel:
    """Adapter for baselines run outside this repository.

    DSoftKI and DDSVGP should be run on the canonical fair split exported by
    this runner. In strict mode, the CSV row must identify the same split and
    preprocessing convention through split_id and preprocessing_version.
    """

    def __init__(
        self,
        *,
        method: str,
        csv_path: str | None,
        seed: int,
        strict_fairness: bool = True,
        preprocessing_version: str | None = None,
        x_scale: float | None = None,
    ) -> None:
        self.name = method
        self.csv_path = csv_path
        self.seed = int(seed)
        self.strict_fairness = bool(strict_fairness)
        self.preprocessing_version = preprocessing_version
        self.x_scale = None if x_scale is None else float(x_scale)
        self.normalized_energy_rmse_per_atom: float | None = None
        self.raw_energy_rmse_per_atom: float | None = None
        self.energy_rmse_per_atom: float | None = None
        self.wall_time_sec: float | None = None
        self.peak_mem_gb: float | None = None

    def _require_columns(self, df: pd.DataFrame, columns: list[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"External result for {self.name} is missing fairness column(s) {missing}. "
                "Run the external baseline on the exported fair split or pass --allow-nonstrict-external."
            )

    def fit(self, split: MD22Split) -> None:
        if not self.csv_path:
            raise RuntimeError(f"{self.name} is external. Set external_results_csv or run it separately.")
        path = Path(self.csv_path)
        if not path.exists():
            raise FileNotFoundError(f"External results CSV not found: {path}")
        df = pd.read_csv(path)
        base_cols = ["dataset", "seed", "method"]
        self._require_columns(df, base_cols)
        if "raw_energy_rmse_per_atom" not in df.columns and "energy_rmse_per_atom" not in df.columns:
            raise RuntimeError("External result is missing raw_energy_rmse_per_atom or energy_rmse_per_atom.")
        mask = (
            (df["dataset"].astype(str) == split.name)
            & (df["seed"].astype(int) == self.seed)
            & (df["method"].astype(str) == self.name)
        )
        if self.strict_fairness:
            fairness_cols = ["split_id", "preprocessing_version", "n_train", "n_test", "d", "x_scale"]
            self._require_columns(df, fairness_cols)
            mask = (
                mask
                & (df["split_id"].astype(str) == split.split_id)
                & (df["preprocessing_version"].astype(str) == split.preprocessing_version)
                & (df["n_train"].astype(int) == int(split.X_train.shape[0]))
                & (df["n_test"].astype(int) == int(split.X_test.shape[0]))
                & (df["d"].astype(int) == int(split.d))
            )

            mask = mask & ((df["x_scale"].astype(float) - float(split.scaler.x_scale)).abs() < 1e-12)
        rows = df.loc[mask]
        if rows.empty:
            raise RuntimeError(
                f"No external result for dataset={split.name}, seed={self.seed}, method={self.name}, "
                f"split_id={split.split_id}."
            )
        row = rows.iloc[0]
        self.normalized_energy_rmse_per_atom = float(row.get("normalized_energy_rmse_per_atom", float("nan")))
        self.raw_energy_rmse_per_atom = float(row.get("raw_energy_rmse_per_atom", row.get("energy_rmse_per_atom")))
        self.energy_rmse_per_atom = self.raw_energy_rmse_per_atom
        self.wall_time_sec = float(row.get("wall_time_sec", row.get("total_time_sec", 0.0)))
        self.peak_mem_gb = float(row.get("peak_mem_gb", 0.0))

    def predict(self, X: torch.Tensor) -> Prediction:
        raise RuntimeError("ExternalResultsModel does not provide predictions. The runner reads stored metrics.")
