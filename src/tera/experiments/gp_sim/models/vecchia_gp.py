from __future__ import annotations

from typing import List as list_type

import torch

from tera.experiments.gp_sim.data import predictive_marginals, simulated_dataset
from tera.experiments.gp_sim.deroos import function_covariance
from tera.experiments.gp_sim.models.base import marginal_predictor
from tera.experiments.gp_sim.models.common import function_conditioning_cache
from tera.experiments.gp_sim.ordering import knn_to_eval
from tera.experiments.gp_sim.utils import scale_inputs


class vecchia_gp_predictor(marginal_predictor):
    """Function-only vecchia gp predictor using nearest observed neighbors."""

    def __init__(self, m: int) -> None:
        self.m = m
        self.data: simulated_dataset | None = None
        self._function_cache = function_conditioning_cache()

    def build(self, data: simulated_dataset) -> None:
        self.data = data
        self._function_cache.reset(data)

    def predict_f_marginals(self, x_eval: torch.Tensor) -> predictive_marginals:
        if self.data is None:
            raise RuntimeError("build() must be called before prediction.")

        data = self.data
        x_eval_scaled = scale_inputs(x_eval, data.lengthscale)
        neighborhoods = knn_to_eval(data.x_train_scaled, x_eval_scaled, self.m)

        means: list_type[torch.Tensor] = []
        variances: list_type[torch.Tensor] = []
        prior_var = torch.as_tensor(data.outputscale, device=x_eval.device, dtype=x_eval.dtype)

        with torch.no_grad():
            for j, idx in enumerate(neighborhoods):
                if idx.numel() == 0:
                    means.append(x_eval.new_zeros(()))
                    variances.append(prior_var)
                    continue

                lff, alpha_y = self._function_cache.get(idx)
                xc = data.x_train[idx]
                k_cx = function_covariance(
                    xc,
                    x_eval[j : j + 1],
                    data.lengthscale,
                    data.outputscale,
                    data.kernel_name,
                ).squeeze(-1)

                mean = torch.dot(k_cx, alpha_y)
                weights = torch.cholesky_solve(k_cx.unsqueeze(-1), lff).squeeze(-1)
                var = torch.clamp(
                    prior_var - torch.dot(k_cx, weights), min=torch.finfo(x_eval.dtype).eps
                )
                means.append(mean)
                variances.append(var)

        return predictive_marginals(
            mean=torch.stack(means).contiguous(),
            var=torch.stack(variances).contiguous(),
        )
