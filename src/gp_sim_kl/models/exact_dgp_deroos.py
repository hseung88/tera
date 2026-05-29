from __future__ import annotations

import torch

from gp_sim_kl.data import PredictiveMarginals, SimulatedDataset
from gp_sim_kl.deroos import prepare_value_grad_deroos_conditioner, predict_all_value_grad_deroos
from gp_sim_kl.models.base import MarginalPredictor

class ExactDerivativeGPDeroosPredictor(MarginalPredictor):
    """Exact derivative-GP predictor using the de Roos stationary-kernel algebra."""

    def __init__(self) -> None:
        self.data: SimulatedDataset | None = None
        self.cond = None

    def build(self, data: SimulatedDataset) -> None:
        with torch.no_grad():
            self.data = data
            self.cond = prepare_value_grad_deroos_conditioner(
                data.X_train,
                data.f_train_obs,
                data.g_train_obs,
                lengthscale=data.lengthscale,
                outputscale=data.outputscale,
                kernel=data.kernel_name,
                sigma_f=data.sigma_f,
                sigma_g=data.sigma_g,
            )

    def predict_f_marginals(self, X_eval: torch.Tensor) -> PredictiveMarginals:
        with torch.no_grad():
            if self.cond is None:
                raise RuntimeError("build() must be called before prediction.")
            mean, var = predict_all_value_grad_deroos(self.cond, X_eval)
            return PredictiveMarginals(mean=mean.contiguous(), var=var.contiguous())
