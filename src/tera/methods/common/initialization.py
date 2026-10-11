from __future__ import annotations

import torch

from tera.methods.common.deroos import function_covariance
from tera.methods.common.utils import cholesky_with_jitter, resolve_lengthscale


def median_pairwise_lengthscale(inputs: torch.Tensor, *, max_points: int = 2048) -> torch.Tensor:

    n = int(inputs.shape[0])
    if n < 2:
        return torch.ones(1, device=inputs.device, dtype=inputs.dtype)
    max_points = int(max_points)
    if max_points > 0 and n > max_points:
        idx = torch.linspace(0, n - 1, steps=max_points, device=inputs.device).round().long()
        x_ref = inputs[idx].contiguous()
    else:
        x_ref = inputs
    with torch.no_grad():
        dist = torch.pdist(x_ref, p=2.0)
        positive = dist[dist > 0]
        if positive.numel() == 0:
            return torch.ones(1, device=inputs.device, dtype=inputs.dtype)
        ell = torch.median(positive).clamp_min(torch.finfo(inputs.dtype).eps)
    return ell.reshape(1).to(device=inputs.device, dtype=inputs.dtype)


def resolve_kernel_lengthscale(
    inputs: torch.Tensor,
    *,
    lengthscale: float | list[float] | None,
    lengthscale_init: str = "median",
    lengthscale_init_max_points: int = 2048,
    use_ard: bool = False,
) -> torch.Tensor:
    if lengthscale is not None:
        ell = resolve_lengthscale(lengthscale, inputs.shape[1], use_ard, device=inputs.device, dtype=inputs.dtype)
        return ell.repeat(inputs.shape[1]) if use_ard and ell.numel() == 1 else ell
    if lengthscale_init == "median":
        ell = median_pairwise_lengthscale(inputs, max_points=lengthscale_init_max_points)
    elif lengthscale_init == "one":
        ell = torch.ones(1, device=inputs.device, dtype=inputs.dtype)
    else:
        raise ValueError(
            f"Unknown lengthscale_init={lengthscale_init!r}. Supported values are 'median' and 'one'."
        )
    if use_ard:
        return ell.repeat(inputs.shape[1]).contiguous()
    return ell


def kff_cholesky(
    inputs: torch.Tensor, lengthscale: torch.Tensor, outputscale, sigma_f, kernel: str
) -> torch.Tensor:
    covariance = function_covariance(inputs, inputs, lengthscale, outputscale, kernel)
    covariance = 0.5 * (covariance + covariance.T)
    sf = torch.as_tensor(sigma_f, device=inputs.device, dtype=inputs.dtype)
    if bool((sf > 0.0).detach().cpu().item()):
        covariance = covariance + sf * torch.eye(inputs.shape[0], device=inputs.device, dtype=inputs.dtype)
    return cholesky_with_jitter(covariance)


__all__ = [
    'function_covariance',
    'median_pairwise_lengthscale',
    'resolve_kernel_lengthscale',
    'kff_cholesky',
]
