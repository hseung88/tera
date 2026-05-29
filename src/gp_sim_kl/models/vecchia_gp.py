from __future__ import annotations

from typing import List

import torch

from gp_sim_kl.data import PredictiveMarginals, SimulatedDataset
from gp_sim_kl.deroos import function_covariance
from gp_sim_kl.models.base import MarginalPredictor
from gp_sim_kl.models.common import FunctionConditioningCache
from gp_sim_kl.ordering import knn_to_eval
from gp_sim_kl.utils import scale_inputs

class VecchiaGPPredictor(MarginalPredictor):
    """Function-only Vecchia GP predictor using nearest observed neighbors."""

    def __init__(self, m: int) -> None:
        self.m = m
        self.data: SimulatedDataset | None = None
        self._function_cache = FunctionConditioningCache()

    def build(self, data: SimulatedDataset) -> None:
        self.data = data
        self._function_cache.reset(data)

    def predict_f_marginals(self, X_eval: torch.Tensor) -> PredictiveMarginals:
        if self.data is None:
            raise RuntimeError("build() must be called before prediction.")

        data = self.data
        X_eval_scaled = scale_inputs(X_eval, data.lengthscale)
        neighborhoods = knn_to_eval(data.X_train_scaled, X_eval_scaled, self.m)

        means: List[torch.Tensor] = []
        variances: List[torch.Tensor] = []
        prior_var = torch.as_tensor(data.outputscale, device=X_eval.device, dtype=X_eval.dtype)

        with torch.no_grad():
            for j, idx in enumerate(neighborhoods):
                if idx.numel() == 0:
                    means.append(X_eval.new_zeros(()))
                    variances.append(prior_var)
                    continue

                Lff, alpha_y = self._function_cache.get(idx)
                Xc = data.X_train[idx]
                k_cx = function_covariance(
                    Xc,
                    X_eval[j : j + 1],
                    data.lengthscale,
                    data.outputscale,
                    data.kernel_name,
                ).squeeze(-1)

                mean = torch.dot(k_cx, alpha_y)
                weights = torch.cholesky_solve(k_cx.unsqueeze(-1), Lff).squeeze(-1)
                var = torch.clamp(prior_var - torch.dot(k_cx, weights), min=torch.finfo(X_eval.dtype).eps)
                means.append(mean)
                variances.append(var)

        return PredictiveMarginals(
            mean=torch.stack(means).contiguous(),
            var=torch.stack(variances).contiguous(),
        )
