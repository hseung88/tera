from __future__ import annotations

from typing import Dict, Tuple

import torch

from gp_sim_kl.data import PredictiveMarginals, SimulatedDataset
from gp_sim_kl.deroos import prepare_value_grad_deroos_conditioner, predict_one_value_grad_deroos
from gp_sim_kl.models.base import MarginalPredictor
from gp_sim_kl.ordering import knn_to_eval
from gp_sim_kl.utils import scale_inputs

class VecchiaDGPDeroosPredictor(MarginalPredictor):
    def __init__(self, m: int) -> None:
        self.m = m
        self.data: SimulatedDataset | None = None
        self._cache: Dict[Tuple[int, ...], object] = {}

    def build(self, data: SimulatedDataset) -> None:
        self.data = data
        self._cache.clear()

    def _get_conditioner(self, idx: torch.Tensor):
        assert self.data is not None
        key = tuple(int(i) for i in idx.detach().cpu().tolist())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        cond = prepare_value_grad_deroos_conditioner(
            self.data.X_train[idx],
            self.data.f_train_obs[idx],
            self.data.g_train_obs[idx],
            lengthscale=self.data.lengthscale,
            outputscale=self.data.outputscale,
            kernel=self.data.kernel_name,
            sigma_f=self.data.sigma_f,
            sigma_g=self.data.sigma_g,
        )
        self._cache[key] = cond
        return cond

    def predict_f_marginals(self, X_eval: torch.Tensor) -> PredictiveMarginals:
        data = self.data
        assert data is not None
        with torch.no_grad():
            X_eval_scaled = scale_inputs(X_eval, data.lengthscale)
            neighborhoods = knn_to_eval(data.X_train_scaled, X_eval_scaled, self.m)
            means: list[torch.Tensor] = []
            vars_: list[torch.Tensor] = []
            for j, idx in enumerate(neighborhoods):
                cond = self._get_conditioner(idx)
                mu, var = predict_one_value_grad_deroos(cond, X_eval[j:j + 1])
                means.append(mu)
                vars_.append(var)
            return PredictiveMarginals(mean=torch.stack(means).contiguous(), var=torch.stack(vars_).contiguous())
