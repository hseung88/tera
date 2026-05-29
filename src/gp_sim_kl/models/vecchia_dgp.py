from __future__ import annotations

import math
import torch

from gp_sim_kl.data import PredictiveMarginals, SimulatedDataset
from gp_sim_kl.models.base import MarginalPredictor
from gp_sim_kl.ordering import knn_to_eval
from gp_sim_kl.utils import cholesky_with_jitter, function_output_indices, scale_inputs

class VecchiaDGPPredictor(MarginalPredictor):
    def __init__(self, m: int) -> None:
        self.m = m
        self.data: SimulatedDataset | None = None

    def build(self, data: SimulatedDataset) -> None:
        self.data = data

    def _inv_ls2(self, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        assert self.data is not None
        ls = self.data.lengthscale
        if ls.numel() == 1:
            return (1.0 / (ls * ls)).repeat(d).to(device=device, dtype=dtype)
        return (1.0 / (ls * ls)).to(device=device, dtype=dtype)

    def _local_full_block(self, Xc: torch.Tensor) -> torch.Tensor:
        assert self.data is not None
        name = self.data.kernel_name
        outputscale = float(self.data.outputscale)
        device, dtype = Xc.device, Xc.dtype
        m, d = Xc.shape
        w = self._inv_ls2(d, device, dtype)

        delta = Xc[:, None, :] - Xc[None, :, :]
        v = delta * w.view(1, 1, d)
        sq = (delta * v).sum(dim=-1)

        if name == 'rbf':
            Kff = outputscale * torch.exp(-0.5 * sq)
            Kfg = Kff[..., None] * v
            Kgf = -Kfg
            diagW = torch.diag(w)
            outer = v[..., :, None] * v[..., None, :]
            Kgg = Kff[..., None, None] * (diagW.view(1, 1, d, d) - outer)
        elif name == 'matern52':
            sqrt5 = math.sqrt(5.0)
            r = torch.sqrt(torch.clamp(sq, min=1e-12))
            exp_term = torch.exp(-sqrt5 * r)
            Kff = outputscale * (1.0 + sqrt5 * r + (5.0 / 3.0) * sq) * exp_term
            coeff = outputscale * (5.0 / 3.0) * (1.0 + sqrt5 * r) * exp_term
            Kfg = coeff[..., None] * v
            Kgf = -Kfg
            diagW = torch.diag(w)
            outer = v[..., :, None] * v[..., None, :]
            Kgg = -outputscale * (5.0 / 3.0) * exp_term[..., None, None] * (
                5.0 * outer - diagW.view(1, 1, d, d) * (1.0 + sqrt5 * r)[..., None, None]
            )
        else:
            raise ValueError(f'Unknown kernel name: {name}')

        blk = torch.zeros((m, m, d + 1, d + 1), device=device, dtype=dtype)
        blk[:, :, 0, 0] = Kff
        blk[:, :, 0, 1:] = Kfg
        blk[:, :, 1:, 0] = Kgf
        blk[:, :, 1:, 1:] = Kgg
        K = blk.permute(0, 2, 1, 3).reshape(m * (d + 1), m * (d + 1)).contiguous()
        return 0.5 * (K + K.T)

    def _cross_value_to_local(self, x: torch.Tensor, Xc: torch.Tensor) -> torch.Tensor:
        assert self.data is not None
        name = self.data.kernel_name
        outputscale = float(self.data.outputscale)
        device, dtype = Xc.device, Xc.dtype
        m, d = Xc.shape
        w = self._inv_ls2(d, device, dtype)

        delta = x.view(1, 1, d) - Xc.view(1, m, d)
        v = delta * w.view(1, 1, d)
        sq = (delta * v).sum(dim=-1).squeeze(0)

        if name == 'rbf':
            kff = outputscale * torch.exp(-0.5 * sq)
            kfg = kff[:, None] * v.squeeze(0)
        elif name == 'matern52':
            sqrt5 = math.sqrt(5.0)
            r = torch.sqrt(torch.clamp(sq, min=1e-12))
            exp_term = torch.exp(-sqrt5 * r)
            kff = outputscale * (1.0 + sqrt5 * r + (5.0 / 3.0) * sq) * exp_term
            coeff = outputscale * (5.0 / 3.0) * (1.0 + sqrt5 * r) * exp_term
            kfg = coeff[:, None] * v.squeeze(0)
        else:
            raise ValueError(f'Unknown kernel name: {name}')

        return torch.cat([kff[:, None], kfg], dim=1).reshape(1, m * (d + 1)).contiguous()

    @staticmethod
    def _local_z(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return torch.cat([f.unsqueeze(-1), g], dim=1).reshape(-1)

    def predict_f_marginals(self, X_eval: torch.Tensor) -> PredictiveMarginals:
        data = self.data
        assert data is not None
        with torch.no_grad():
            X_eval_scaled = scale_inputs(X_eval, data.lengthscale)
            neighborhoods = knn_to_eval(data.X_train_scaled, X_eval_scaled, self.m)
            d = data.X_train.shape[1]
            means: list[torch.Tensor] = []
            vars_: list[torch.Tensor] = []
            k_xx = torch.as_tensor(data.outputscale, device=X_eval.device, dtype=X_eval.dtype)

            for j, idx in enumerate(neighborhoods):
                Xc = data.X_train[idx]
                Kcc = self._local_full_block(Xc)
                n_local = Xc.shape[0]
                idx_f = function_output_indices(n_local, d, Xc.device)
                if data.sigma_f > 0.0:
                    Kcc[idx_f, idx_f] += data.sigma_f ** 2
                if data.sigma_g > 0.0:
                    ar = torch.arange(1, d + 1, device=Xc.device, dtype=torch.long)
                    idx_g = (idx_f[:, None] + ar[None, :]).reshape(-1)
                    Kcc[idx_g, idx_g] += data.sigma_g ** 2
                k_fz = self._cross_value_to_local(X_eval[j:j + 1], Xc)
                Lcc = cholesky_with_jitter(Kcc)
                local_w = torch.cholesky_solve(k_fz.T, Lcc).squeeze(-1)
                z_local = self._local_z(data.f_train_obs[idx], data.g_train_obs[idx])
                mean = torch.dot(local_w, z_local)
                var = torch.clamp(k_xx - torch.dot(k_fz.squeeze(0), local_w), min=torch.finfo(X_eval.dtype).eps)
                means.append(mean)
                vars_.append(var)
            return PredictiveMarginals(mean=torch.stack(means).contiguous(), var=torch.stack(vars_).contiguous())
