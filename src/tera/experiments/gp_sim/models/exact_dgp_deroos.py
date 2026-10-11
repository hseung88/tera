from __future__ import annotations

import torch

from tera.experiments.gp_sim.data import predictive_marginals, simulated_dataset
from tera.experiments.gp_sim.deroos import (
    predict_all_value_grad_deroos,
    prepare_value_grad_deroos_conditioner,
)
from tera.experiments.gp_sim.models.base import marginal_predictor


class exact_derivative_gp_deroos_predictor(marginal_predictor):
    """Exact derivative-GP predictor using the de Roos stationary-kernel algebra."""

    def __init__(self) -> None:
        self.data: simulated_dataset | None = None
        self.cond = None

    def build(self, data: simulated_dataset) -> None:
        with torch.no_grad():
            self.data = data
            self.cond = prepare_value_grad_deroos_conditioner(
                data.x_train,
                data.f_train_obs,
                data.g_train_obs,
                lengthscale=data.lengthscale,
                outputscale=data.outputscale,
                kernel=data.kernel_name,
                sigma_f=data.sigma_f,
                sigma_g=data.sigma_g,
            )

    def predict_f_marginals(self, x_eval: torch.Tensor) -> predictive_marginals:
        with torch.no_grad():
            if self.cond is None:
                raise RuntimeError("build() must be called before prediction.")
            mean, var = predict_all_value_grad_deroos(self.cond, x_eval)
            return predictive_marginals(mean=mean.contiguous(), var=var.contiguous())
