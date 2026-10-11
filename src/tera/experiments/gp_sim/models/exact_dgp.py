from __future__ import annotations

import torch

from tera.experiments.gp_sim.data import predictive_marginals, simulated_dataset
from tera.experiments.gp_sim.deroos import full_value_grad_covariance, value_grad_cross_blocks_many
from tera.experiments.gp_sim.models.base import marginal_predictor
from tera.experiments.gp_sim.utils import cholesky_with_jitter


class exact_derivative_gp_predictor(marginal_predictor):
    """Dense exact derivative-GP predictor for function marginals."""

    def __init__(self) -> None:
        self.data: simulated_dataset | None = None
        self.l: torch.Tensor | None = None
        self.kzz_obs: torch.Tensor | None = None
        self.alpha_z: torch.Tensor | None = None

    def build(self, data: simulated_dataset) -> None:
        with torch.no_grad():
            self.data = data
            kzz = full_value_grad_covariance(
                data.x_train,
                data.lengthscale,
                data.outputscale,
                data.kernel_name,
                sigma_f=data.sigma_f,
                sigma_g=data.sigma_g,
            )
            self.kzz_obs = kzz
            self.l = cholesky_with_jitter(kzz)
            self.alpha_z = torch.cholesky_solve(data.z_train_obs.unsqueeze(-1), self.l).squeeze(-1)

    def predict_f_marginals(self, x_eval: torch.Tensor) -> predictive_marginals:
        with torch.no_grad():
            assert self.data is not None and self.l is not None and self.alpha_z is not None
            n, d = self.data.x_train.shape
            k_f, k_g, k_xx = value_grad_cross_blocks_many(
                self.data.x_train,
                x_eval,
                self.data.lengthscale,
                self.data.outputscale,
                self.data.kernel_name,
            )
            k_z = torch.cat([k_f[:, None, :], k_g.reshape(n, d, x_eval.shape[0])], dim=1)
            k_z = k_z.reshape(n * (d + 1), x_eval.shape[0]).contiguous()
            mean = k_z.T @ self.alpha_z
            weights = torch.cholesky_solve(k_z, self.l)
            var = k_xx - (k_z * weights).sum(dim=0)
            var = torch.clamp(var, min=torch.finfo(x_eval.dtype).eps)
            return predictive_marginals(mean=mean.contiguous(), var=var.contiguous())
