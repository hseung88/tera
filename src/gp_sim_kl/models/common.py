from __future__ import annotations

from typing import Dict, Tuple

import torch

from gp_sim_kl.data import SimulatedDataset
from gp_sim_kl.deroos import function_covariance
from gp_sim_kl.utils import cholesky_with_jitter

NeighborhoodKey = Tuple[int, ...]

def neighborhood_key(idx: torch.Tensor) -> NeighborhoodKey:
    """Return a stable cache key for a one-dimensional integer index tensor."""
    return tuple(int(i) for i in idx.detach().cpu().tolist())

class FunctionConditioningCache:
    """Cache local function-value Cholesky factors for repeated neighborhoods."""

    def __init__(self) -> None:
        self.data: SimulatedDataset | None = None
        self._cache: Dict[NeighborhoodKey, tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self, data: SimulatedDataset) -> None:
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
        Xc = data.X_train[idx]
        n_local = int(Xc.shape[0])
        Kff = function_covariance(
            Xc,
            Xc,
            data.lengthscale,
            data.outputscale,
            data.kernel_name,
        )
        Kff = 0.5 * (Kff + Kff.T)
        if data.sigma_f > 0.0:
            Kff = Kff + (data.sigma_f ** 2) * torch.eye(n_local, device=Xc.device, dtype=Xc.dtype)

        Lff = cholesky_with_jitter(Kff)
        alpha_y = torch.cholesky_solve(data.f_train_obs[idx].unsqueeze(-1), Lff).squeeze(-1)
        cached = (Lff, alpha_y)
        self._cache[key] = cached
        return cached
