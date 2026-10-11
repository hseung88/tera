from __future__ import annotations

from typing import Dict as dict_type, Tuple as tuple_type

import torch

from tera.experiments.gp_sim.data import predictive_marginals, simulated_dataset
from tera.experiments.gp_sim.deroos import (
    predict_one_value_grad_deroos,
    prepare_value_grad_deroos_conditioner,
)
from tera.experiments.gp_sim.models.base import marginal_predictor
from tera.experiments.gp_sim.ordering import knn_to_eval
from tera.experiments.gp_sim.utils import scale_inputs


class vecchia_dgp_deroos_predictor(marginal_predictor):
    def __init__(self, m: int) -> None:
        self.m = m
        self.data: simulated_dataset | None = None
        self._cache: dict_type[tuple_type[int, ...], object] = {}

    def build(self, data: simulated_dataset) -> None:
        self.data = data
        self._cache.clear()

    def _get_conditioner(self, idx: torch.Tensor):
        assert self.data is not None
        key = tuple(int(i) for i in idx.detach().cpu().tolist())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        cond = prepare_value_grad_deroos_conditioner(
            self.data.x_train[idx],
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

    def predict_f_marginals(self, x_eval: torch.Tensor) -> predictive_marginals:
        data = self.data
        assert data is not None
        with torch.no_grad():
            x_eval_scaled = scale_inputs(x_eval, data.lengthscale)
            neighborhoods = knn_to_eval(data.x_train_scaled, x_eval_scaled, self.m)
            means: list[torch.Tensor] = []
            vars_: list[torch.Tensor] = []
            for j, idx in enumerate(neighborhoods):
                cond = self._get_conditioner(idx)
                mu, var = predict_one_value_grad_deroos(cond, x_eval[j : j + 1])
                means.append(mu)
                vars_.append(var)
            return predictive_marginals(
                mean=torch.stack(means).contiguous(), var=torch.stack(vars_).contiguous()
            )
