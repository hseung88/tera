from __future__ import annotations

import torch


def normalized_energy_rmse_per_atom(
    y_pred: torch.Tensor, y_true: torch.Tensor, n_atoms: int
) -> float:

    rmse = torch.sqrt(torch.mean((y_pred.reshape(-1) - y_true.reshape(-1)).pow(2)))
    return float((rmse / float(n_atoms)).detach().cpu())


def energy_rmse_per_atom(e_pred: torch.Tensor, e_true: torch.Tensor, n_atoms: int) -> float:

    e_pred = e_pred.reshape(-1)
    e_true = e_true.to(device=e_pred.device, dtype=e_pred.dtype).reshape(-1)
    rmse = torch.sqrt(torch.mean((e_pred - e_true).pow(2)))
    return float((rmse / float(n_atoms)).detach().cpu())


def raw_energy_rmse_per_atom(
    y_pred: torch.Tensor,
    e_true: torch.Tensor,
    *,
    energy_mean: torch.Tensor | float,
    energy_std: torch.Tensor | float,
    n_atoms: int,
) -> float:

    mean = torch.as_tensor(energy_mean, device=y_pred.device, dtype=y_pred.dtype)
    std = torch.as_tensor(energy_std, device=y_pred.device, dtype=y_pred.dtype)
    e_pred = mean - std * y_pred.reshape(-1)
    e_true = e_true.to(device=y_pred.device, dtype=y_pred.dtype).reshape(-1)
    rmse = torch.sqrt(torch.mean((e_pred - e_true).pow(2)))
    return float((rmse / float(n_atoms)).detach().cpu())
