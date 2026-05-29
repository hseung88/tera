from __future__ import annotations

import torch

from gp_sim_kl.data import PredictiveMarginals, SimulatedDataset
from gp_sim_kl.deroos import full_value_grad_covariance, value_grad_cross_blocks_many
from gp_sim_kl.models.base import MarginalPredictor
from gp_sim_kl.utils import cholesky_with_jitter

class ExactDerivativeGPPredictor(MarginalPredictor):
    """Dense exact derivative-GP predictor for function marginals."""

    def __init__(self) -> None:
        self.data: SimulatedDataset | None = None
        self.L: torch.Tensor | None = None
        self.Kzz_obs: torch.Tensor | None = None
        self.alpha_z: torch.Tensor | None = None

    def build(self, data: SimulatedDataset) -> None:
        with torch.no_grad():
            self.data = data
            Kzz = full_value_grad_covariance(
                data.X_train,
                data.lengthscale,
                data.outputscale,
                data.kernel_name,
                sigma_f=data.sigma_f,
                sigma_g=data.sigma_g,
            )
            self.Kzz_obs = Kzz
            self.L = cholesky_with_jitter(Kzz)
            self.alpha_z = torch.cholesky_solve(data.z_train_obs.unsqueeze(-1), self.L).squeeze(-1)

    def predict_f_marginals(self, X_eval: torch.Tensor) -> PredictiveMarginals:
        with torch.no_grad():
            assert self.data is not None and self.L is not None and self.alpha_z is not None
            n, d = self.data.X_train.shape
            K_f, K_g, k_xx = value_grad_cross_blocks_many(
                self.data.X_train,
                X_eval,
                self.data.lengthscale,
                self.data.outputscale,
                self.data.kernel_name,
            )
            K_z = torch.cat([K_f[:, None, :], K_g.reshape(n, d, X_eval.shape[0])], dim=1)
            K_z = K_z.reshape(n * (d + 1), X_eval.shape[0]).contiguous()
            mean = K_z.T @ self.alpha_z
            weights = torch.cholesky_solve(K_z, self.L)
            var = k_xx - (K_z * weights).sum(dim=0)
            var = torch.clamp(var, min=torch.finfo(X_eval.dtype).eps)
            return PredictiveMarginals(mean=mean.contiguous(), var=var.contiguous())
