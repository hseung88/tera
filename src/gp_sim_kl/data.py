from __future__ import annotations

from dataclasses import dataclass

import torch

@dataclass(slots=True)
class SimulatedDataset:
    X_train: torch.Tensor
    X_train_scaled: torch.Tensor
    X_eval: torch.Tensor
    X_eval_scaled: torch.Tensor
    lengthscale: torch.Tensor
    outputscale: float
    sigma_f: float
    sigma_g: float
    kernel_name: str
    f_train_obs: torch.Tensor
    g_train_obs: torch.Tensor
    z_train_obs: torch.Tensor
    sampling_backend: str

@dataclass(slots=True)
class PredictiveMarginals:
    mean: torch.Tensor
    var: torch.Tensor
    mean_weights: torch.Tensor | None = None
