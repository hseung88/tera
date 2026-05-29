from __future__ import annotations

import torch

from gp_sim_kl.data import PredictiveMarginals, SimulatedDataset
from gp_sim_kl.deroos import function_covariance
from gp_sim_kl.models.base import MarginalPredictor
from gp_sim_kl.utils import cholesky_with_jitter

class ExactGPPredictor(MarginalPredictor):
    def __init__(self) -> None:
        self.data: SimulatedDataset | None = None
        self.L: torch.Tensor | None = None
        self.alpha_y: torch.Tensor | None = None

    def build(self, data: SimulatedDataset) -> None:
        with torch.no_grad():
            self.data = data
            Kff = function_covariance(data.X_train, data.X_train, data.lengthscale, data.outputscale, data.kernel_name)
            Kff = 0.5 * (Kff + Kff.T)
            if data.sigma_f > 0.0:
                Kff = Kff + (data.sigma_f ** 2) * torch.eye(Kff.shape[0], device=Kff.device, dtype=Kff.dtype)
            self.L = cholesky_with_jitter(Kff)
            self.alpha_y = torch.cholesky_solve(data.f_train_obs.unsqueeze(-1), self.L).squeeze(-1)

    def predict_f_marginals(self, X_eval: torch.Tensor) -> PredictiveMarginals:
        with torch.no_grad():
            assert self.data is not None and self.L is not None and self.alpha_y is not None
            K_xt = function_covariance(X_eval, self.data.X_train, self.data.lengthscale, self.data.outputscale, self.data.kernel_name)
            mean = K_xt @ self.alpha_y
            v = torch.cholesky_solve(K_xt.T, self.L)
            K_xx_diag = torch.full((X_eval.shape[0],), float(self.data.outputscale), device=X_eval.device, dtype=X_eval.dtype)
            var = torch.clamp(K_xx_diag - torch.sum(K_xt * v.T, dim=1), min=torch.finfo(X_eval.dtype).eps)
            return PredictiveMarginals(mean=mean.contiguous(), var=var.contiguous())
