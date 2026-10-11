from __future__ import annotations

import torch

from tera.experiments.gp_sim.data import predictive_marginals, simulated_dataset
from tera.experiments.gp_sim.deroos import function_covariance
from tera.experiments.gp_sim.models.base import marginal_predictor
from tera.experiments.gp_sim.utils import cholesky_with_jitter


class exact_gp_predictor(marginal_predictor):
    def __init__(self) -> None:
        self.data: simulated_dataset | None = None
        self.l: torch.Tensor | None = None
        self.alpha_y: torch.Tensor | None = None

    def build(self, data: simulated_dataset) -> None:
        with torch.no_grad():
            self.data = data
            kff = function_covariance(
                data.x_train, data.x_train, data.lengthscale, data.outputscale, data.kernel_name
            )
            kff = 0.5 * (kff + kff.T)
            if data.sigma_f > 0.0:
                kff = kff + (data.sigma_f**2) * torch.eye(
                    kff.shape[0], device=kff.device, dtype=kff.dtype
                )
            self.l = cholesky_with_jitter(kff)
            self.alpha_y = torch.cholesky_solve(data.f_train_obs.unsqueeze(-1), self.l).squeeze(-1)

    def predict_f_marginals(self, x_eval: torch.Tensor) -> predictive_marginals:
        with torch.no_grad():
            assert self.data is not None and self.l is not None and self.alpha_y is not None
            k_xt = function_covariance(
                x_eval,
                self.data.x_train,
                self.data.lengthscale,
                self.data.outputscale,
                self.data.kernel_name,
            )
            mean = k_xt @ self.alpha_y
            v = torch.cholesky_solve(k_xt.T, self.l)
            k_xx_diag = torch.full(
                (x_eval.shape[0],),
                float(self.data.outputscale),
                device=x_eval.device,
                dtype=x_eval.dtype,
            )
            var = torch.clamp(
                k_xx_diag - torch.sum(k_xt * v.T, dim=1), min=torch.finfo(x_eval.dtype).eps
            )
            return predictive_marginals(mean=mean.contiguous(), var=var.contiguous())
