from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class simulated_dataset:
    x_train: torch.Tensor
    x_train_scaled: torch.Tensor
    x_eval: torch.Tensor
    x_eval_scaled: torch.Tensor
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
class predictive_marginals:
    mean: torch.Tensor
    var: torch.Tensor
    mean_weights: torch.Tensor | None = None
