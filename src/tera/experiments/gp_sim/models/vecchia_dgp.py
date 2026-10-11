from __future__ import annotations

import math

import torch

from tera.experiments.gp_sim.data import predictive_marginals, simulated_dataset
from tera.experiments.gp_sim.models.base import marginal_predictor
from tera.experiments.gp_sim.ordering import knn_to_eval
from tera.experiments.gp_sim.utils import (
    cholesky_with_jitter,
    function_output_indices,
    scale_inputs,
)


class vecchia_dgp_predictor(marginal_predictor):
    def __init__(self, m: int) -> None:
        self.m = m
        self.data: simulated_dataset | None = None

    def build(self, data: simulated_dataset) -> None:
        self.data = data

    def _inv_ls2(self, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        assert self.data is not None
        ls = self.data.lengthscale
        if ls.numel() == 1:
            return (1.0 / (ls * ls)).repeat(d).to(device=device, dtype=dtype)
        return (1.0 / (ls * ls)).to(device=device, dtype=dtype)

    def _local_full_block(self, xc: torch.Tensor) -> torch.Tensor:
        assert self.data is not None
        name = self.data.kernel_name
        outputscale = float(self.data.outputscale)
        device, dtype = xc.device, xc.dtype
        m, d = xc.shape
        w = self._inv_ls2(d, device, dtype)

        delta = xc[:, None, :] - xc[None, :, :]
        v = delta * w.view(1, 1, d)
        sq = (delta * v).sum(dim=-1)

        if name == "rbf":
            kff = outputscale * torch.exp(-0.5 * sq)
            kfg = kff[..., None] * v
            kgf = -kfg
            diagw = torch.diag(w)
            outer = v[..., :, None] * v[..., None, :]
            kgg = kff[..., None, None] * (diagw.view(1, 1, d, d) - outer)
        elif name == "matern52":
            sqrt5 = math.sqrt(5.0)
            r = torch.sqrt(torch.clamp(sq, min=1e-12))
            exp_term = torch.exp(-sqrt5 * r)
            kff = outputscale * (1.0 + sqrt5 * r + (5.0 / 3.0) * sq) * exp_term
            coeff = outputscale * (5.0 / 3.0) * (1.0 + sqrt5 * r) * exp_term
            kfg = coeff[..., None] * v
            kgf = -kfg
            diagw = torch.diag(w)
            outer = v[..., :, None] * v[..., None, :]
            kgg = (
                -outputscale
                * (5.0 / 3.0)
                * exp_term[..., None, None]
                * (5.0 * outer - diagw.view(1, 1, d, d) * (1.0 + sqrt5 * r)[..., None, None])
            )
        else:
            raise ValueError(f"Unknown kernel name: {name}")

        blk = torch.zeros((m, m, d + 1, d + 1), device=device, dtype=dtype)
        blk[:, :, 0, 0] = kff
        blk[:, :, 0, 1:] = kfg
        blk[:, :, 1:, 0] = kgf
        blk[:, :, 1:, 1:] = kgg
        covariance = blk.permute(0, 2, 1, 3).reshape(m * (d + 1), m * (d + 1)).contiguous()
        return 0.5 * (covariance + covariance.T)

    def _cross_value_to_local(self, x: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        assert self.data is not None
        name = self.data.kernel_name
        outputscale = float(self.data.outputscale)
        device, dtype = xc.device, xc.dtype
        m, d = xc.shape
        w = self._inv_ls2(d, device, dtype)

        delta = x.view(1, 1, d) - xc.view(1, m, d)
        v = delta * w.view(1, 1, d)
        sq = (delta * v).sum(dim=-1).squeeze(0)

        if name == "rbf":
            kff = outputscale * torch.exp(-0.5 * sq)
            kfg = kff[:, None] * v.squeeze(0)
        elif name == "matern52":
            sqrt5 = math.sqrt(5.0)
            r = torch.sqrt(torch.clamp(sq, min=1e-12))
            exp_term = torch.exp(-sqrt5 * r)
            kff = outputscale * (1.0 + sqrt5 * r + (5.0 / 3.0) * sq) * exp_term
            coeff = outputscale * (5.0 / 3.0) * (1.0 + sqrt5 * r) * exp_term
            kfg = coeff[:, None] * v.squeeze(0)
        else:
            raise ValueError(f"Unknown kernel name: {name}")

        return torch.cat([kff[:, None], kfg], dim=1).reshape(1, m * (d + 1)).contiguous()

    @staticmethod
    def _local_z(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return torch.cat([f.unsqueeze(-1), g], dim=1).reshape(-1)

    def predict_f_marginals(self, x_eval: torch.Tensor) -> predictive_marginals:
        data = self.data
        assert data is not None
        with torch.no_grad():
            x_eval_scaled = scale_inputs(x_eval, data.lengthscale)
            neighborhoods = knn_to_eval(data.x_train_scaled, x_eval_scaled, self.m)
            d = data.x_train.shape[1]
            means: list[torch.Tensor] = []
            vars_: list[torch.Tensor] = []
            k_xx = torch.as_tensor(data.outputscale, device=x_eval.device, dtype=x_eval.dtype)

            for j, idx in enumerate(neighborhoods):
                xc = data.x_train[idx]
                kcc = self._local_full_block(xc)
                n_local = xc.shape[0]
                idx_f = function_output_indices(n_local, d, xc.device)
                if data.sigma_f > 0.0:
                    kcc[idx_f, idx_f] += data.sigma_f**2
                if data.sigma_g > 0.0:
                    ar = torch.arange(1, d + 1, device=xc.device, dtype=torch.long)
                    idx_g = (idx_f[:, None] + ar[None, :]).reshape(-1)
                    kcc[idx_g, idx_g] += data.sigma_g**2
                k_fz = self._cross_value_to_local(x_eval[j : j + 1], xc)
                lcc = cholesky_with_jitter(kcc)
                local_w = torch.cholesky_solve(k_fz.T, lcc).squeeze(-1)
                z_local = self._local_z(data.f_train_obs[idx], data.g_train_obs[idx])
                mean = torch.dot(local_w, z_local)
                var = torch.clamp(
                    k_xx - torch.dot(k_fz.squeeze(0), local_w), min=torch.finfo(x_eval.dtype).eps
                )
                means.append(mean)
                vars_.append(var)
            return predictive_marginals(
                mean=torch.stack(means).contiguous(), var=torch.stack(vars_).contiguous()
            )
