from __future__ import annotations

from typing import Dict as dict_type, Tuple as tuple_type

import torch

from tera.experiments.gp_sim.data import simulated_dataset
from tera.experiments.gp_sim.deroos import function_covariance
from tera.experiments.gp_sim.utils import cholesky_with_jitter

neighborhood_key_type = tuple_type[int, ...]


def neighborhood_key(idx: torch.Tensor) -> neighborhood_key_type:
    """Return a stable cache key for a one-dimensional integer index tensor."""
    return tuple(int(i) for i in idx.detach().cpu().tolist())


class function_conditioning_cache:
    """Cache local function-value Cholesky factors for repeated neighborhoods."""

    def __init__(self) -> None:
        self.data: simulated_dataset | None = None
        self._cache: dict_type[neighborhood_key_type, tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self, data: simulated_dataset) -> None:
        self.data = data
        self._cache.clear()

    def get(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(L_ff, alpha_y)`` for ``p(f_* | f_idx)`` computations."""
        if self.data is None:
            raise RuntimeError("FunctionConditioningCache.reset() must be called before get().")

        key = neighborhood_key(idx)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        data = self.data
        xc = data.x_train[idx]
        n_local = int(xc.shape[0])
        kff = function_covariance(
            xc,
            xc,
            data.lengthscale,
            data.outputscale,
            data.kernel_name,
        )
        kff = 0.5 * (kff + kff.T)
        if data.sigma_f > 0.0:
            kff = kff + (data.sigma_f**2) * torch.eye(n_local, device=xc.device, dtype=xc.dtype)

        lff = cholesky_with_jitter(kff)
        alpha_y = torch.cholesky_solve(data.f_train_obs[idx].unsqueeze(-1), lff).squeeze(-1)
        cached = (lff, alpha_y)
        self._cache[key] = cached
        return cached
