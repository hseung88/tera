from __future__ import annotations

import math
import torch

from gp_sim_kl.data import PredictiveMarginals, SimulatedDataset
from gp_sim_kl.deroos import function_covariance
from gp_sim_kl.models.base import MarginalPredictor
from gp_sim_kl.models.common import FunctionConditioningCache
from gp_sim_kl.ordering import knn_to_eval
from gp_sim_kl.utils import cholesky_with_jitter, scale_inputs

class TERAPredictor(MarginalPredictor):
    def __init__(self, m: int) -> None:
        self.m = m
        self.data: SimulatedDataset | None = None
        self._function_cache = FunctionConditioningCache()

    def build(self, data: SimulatedDataset) -> None:
        self.data = data
        self._function_cache.reset(data)

    def predict_f_marginals(self, X_eval: torch.Tensor) -> PredictiveMarginals:
        if self.data is None:
            raise RuntimeError('build() must be called before prediction.')

        data = self.data
        X_eval_scaled = scale_inputs(X_eval, data.lengthscale)
        neighborhoods = knn_to_eval(data.X_train_scaled, X_eval_scaled, self.m)

        means: list[torch.Tensor] = []
        vars_: list[torch.Tensor] = []
        with torch.no_grad():
            for j, idx in enumerate(neighborhoods):
                mean_j, var_j = self._predict_one(
                    x_eval=X_eval[j:j + 1],
                    x_eval_scaled=X_eval_scaled[j:j + 1],
                    idx=idx,
                )
                means.append(mean_j)
                vars_.append(var_j)
        return PredictiveMarginals(mean=torch.stack(means).contiguous(), var=torch.stack(vars_).contiguous())

    def _predict_one(self, *, x_eval: torch.Tensor, x_eval_scaled: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.data is not None
        data = self.data
        device = x_eval.device
        dtype = x_eval.dtype

        m_local = int(idx.numel())
        k_xx = function_covariance(x_eval, x_eval, data.lengthscale, data.outputscale, data.kernel_name).reshape(()).to(dtype=dtype)
        if m_local == 0:
            return x_eval.new_zeros(()), k_xx

        Xc = data.X_train[idx]
        Xc_scaled = data.X_train_scaled[idx]
        Lff, _ = self._function_cache.get(idx)

        k_fc = function_covariance(Xc, x_eval, data.lengthscale, data.outputscale, data.kernel_name).squeeze(-1)

        delta = (Xc - x_eval).T.contiguous()
        delta_scaled = (Xc_scaled - x_eval_scaled).T.contiguous()
        H = delta_scaled.T @ delta_scaled
        cols = H.T.contiguous()
        q = cols[:, None, :] - cols[None, :, :]

        r_i = torch.diagonal(H, 0)
        alpha_i = _alpha_from_r(r_i, data.kernel_name, data.outputscale)

        delta_cc = Xc_scaled[:, None, :] - Xc_scaled[None, :, :]
        r_cc = (delta_cc * delta_cc).sum(dim=-1)
        alpha_cc = _alpha_from_r(r_cc, data.kernel_name, data.outputscale)
        beta_cc = _beta_from_r(r_cc, data.kernel_name, data.outputscale)

        bar_k = ((-alpha_i[:, None]) * cols).reshape(m_local * m_local).contiguous()
        Q = ((-alpha_cc[:, :, None]) * q).permute(0, 2, 1).reshape(m_local * m_local, m_local).contiguous()

        G0_blocks = alpha_cc[:, :, None, None] * H.view(1, 1, m_local, m_local)
        G0_blocks = G0_blocks + beta_cc[:, :, None, None] * (q[:, :, :, None] * q[:, :, None, :])
        if data.sigma_g > 0.0:
            ell2 = data.lengthscale * data.lengthscale
            if ell2.numel() == 1:
                delta_noise = delta * ell2.reshape(1, 1)
            else:
                delta_noise = delta * ell2.view(-1, 1)
            R = delta.T @ delta_noise
            diag_idx = torch.arange(m_local, device=device)
            G0_blocks[diag_idx, diag_idx] = G0_blocks[diag_idx, diag_idx] + (data.sigma_g ** 2) * R
        G0 = G0_blocks.permute(0, 2, 1, 3).reshape(m_local * m_local, m_local * m_local).contiguous()

        rhs = torch.cat([k_fc[:, None], Q.T], dim=1)
        sol = torch.cholesky_solve(rhs, Lff)
        beta_star = sol[:, 0]
        QKinv = sol[:, 1:]

        sigma2 = k_xx - torch.dot(k_fc, beta_star)
        k_delta = bar_k - Q @ beta_star
        K_delta = G0 - Q @ QKinv
        K_delta = 0.5 * (K_delta + K_delta.T)
        w_star = _solve_psd(K_delta, k_delta)

        reduction = torch.dot(w_star, k_delta)
        var = torch.clamp(sigma2 - reduction, min=torch.finfo(dtype).eps)

        qtw = Q.T @ w_star
        f_weights_local = beta_star - torch.cholesky_solve(qtw.unsqueeze(-1), Lff).squeeze(-1)
        q_obs = data.g_train_obs[idx] @ delta
        mean = torch.dot(f_weights_local, data.f_train_obs[idx]) + torch.dot(w_star, q_obs.reshape(-1))
        return mean, var

def _solve_psd(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    A = 0.5 * (A + A.T)
    try:
        L = cholesky_with_jitter(A)
        return torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
    except Exception:
        return torch.linalg.pinv(A) @ b

def _alpha_from_r(r: torch.Tensor, kernel_name: str, outputscale: float) -> torch.Tensor:
    os = outputscale
    if kernel_name == 'rbf':
        return os * torch.exp(-0.5 * r)
    if kernel_name == 'matern52':
        a = math.sqrt(5.0) * torch.sqrt(torch.clamp(r, min=1e-12))
        return (5.0 / 3.0) * os * (1.0 + a) * torch.exp(-a)
    raise ValueError(f'Unknown kernel name: {kernel_name}')

def _beta_from_r(r: torch.Tensor, kernel_name: str, outputscale: float) -> torch.Tensor:
    os = outputscale
    if kernel_name == 'rbf':
        return -os * torch.exp(-0.5 * r)
    if kernel_name == 'matern52':
        a = math.sqrt(5.0) * torch.sqrt(torch.clamp(r, min=1e-12))
        return -(25.0 / 3.0) * os * torch.exp(-a)
    raise ValueError(f'Unknown kernel name: {kernel_name}')
