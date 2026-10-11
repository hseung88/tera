"""Batched tera prediction from the joint reduced covariance."""

from __future__ import annotations

import torch

from tera.methods.common.data import predictive_marginals
from tera.methods.common.deroos import k_kp_kpp_from_r
from tera.methods.common.utils import scale_inputs
from tera.methods.tera.model import _alpha_from_r, _beta_from_r


def solve_joint_batch(matrix, rhs):
    """Use the scalar tera jitter schedule independently for each target."""
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    # Failed Cholesky factors can produce NaNs in backward even with zero gradients.
    factors, info = torch.linalg.cholesky_ex(matrix + 1e-8 * eye)
    if not info.any():
        return torch.cholesky_solve(rhs, factors)
    probe = factors.detach()
    del factors
    pending = torch.arange(len(matrix), device=matrix.device)
    result = torch.zeros_like(rhs)
    jitter = 1e-8
    while pending.numel():
        good = info == 0
        if good.any():
            ids = pending[good]
            # Failed factors must stay outside the backward graph.
            factors = (
                torch.linalg.cholesky(matrix[ids] + jitter * eye)
                if torch.is_grad_enabled() and matrix.requires_grad
                else probe[good]
            )
            result = result.index_copy(0, ids, torch.cholesky_solve(rhs[ids], factors))
        pending = pending[~good]
        jitter *= 10.0
        if not pending.numel() or jitter > 1e-1:
            break
        with torch.no_grad():
            probe, info = torch.linalg.cholesky_ex(matrix[pending] + jitter * eye)
    if pending.numel():
        solved = torch.linalg.pinv(matrix[pending]) @ rhs[pending]
        result = result.index_copy(0, pending, solved)
    return result


def function_covariance_batch(inputs1, inputs2, ell, outputscale, kernel, *, centered=True):
    if not centered:
        r = ((inputs1.unsqueeze(-2) - inputs2.unsqueeze(-3)) / ell).square().sum(-1)
        return k_kp_kpp_from_r(r, kernel, outputscale)[0]
    center = 0.5 * (inputs1.mean(dim=-2, keepdim=True) + inputs2.mean(dim=-2, keepdim=True))
    x1, x2 = (inputs1 - center) / ell, (inputs2 - center) / ell
    r = x1.square().sum(-1, keepdim=True) + x2.square().sum(-1).unsqueeze(-2)
    r = (r - 2.0 * (x1 @ x2.transpose(-1, -2))).clamp_min(0)
    return k_kp_kpp_from_r(r, kernel, outputscale)[0]


class batched_tera_predictor:
    def __init__(self, m, gradient_noise_model, prediction_batch_size):
        if gradient_noise_model not in {"iid", "scaled"}:
            raise ValueError("gradient_noise_model must be either 'iid' or 'scaled'.")
        self.m = int(m)
        self.gradient_noise_model = gradient_noise_model
        self.data = None
        if prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        self.prediction_batch_size = prediction_batch_size

    def build(self, data):
        self.data = data

    @torch.no_grad()
    def predict_f_marginals(self, x_eval, *, neighborhoods=None):
        if self.data is None:
            raise RuntimeError("build() must be called before prediction.")
        if not len(x_eval):
            return predictive_marginals(
                mean=x_eval.new_empty(0),
                var=x_eval.new_empty(0),
            )
        data = self.data
        if neighborhoods is not None and len(neighborhoods) != len(x_eval):
            raise ValueError("neighborhood count must match target count")
        k = max(0, min(self.m, len(data.x_train)))
        distance_mode = (
            "use_mm_for_euclid_dist"
            if max(len(x_eval), len(data.x_train)) > 25
            else "donot_use_mm_for_euclid_dist"
        )
        means, variances = [], []
        for start in range(0, len(x_eval), self.prediction_batch_size):
            stop = start + self.prediction_batch_size
            targets = x_eval[start:stop]
            scaled = scale_inputs(targets, data.lengthscale)
            if neighborhoods is None:
                if k == 0:
                    ids = torch.empty((len(targets), 0), device=targets.device, dtype=torch.long)
                else:
                    distances = torch.cdist(scaled, data.x_train_scaled, compute_mode=distance_mode)
                    ids = distances.topk(k, largest=False).indices
                    del distances
            else:
                selected = neighborhoods[start:stop]
                ids = selected if isinstance(selected, torch.Tensor) else torch.stack(selected)
            mean, var = self._predict_batch(targets, scaled, ids)
            means.append(mean)
            variances.append(var)
        return predictive_marginals(
            mean=torch.cat(means),
            var=torch.cat(variances),
        )

    def _predict_batch(self, targets, targets_scaled, ids, *, centered=True):
        data = self.data
        batch, m = ids.shape
        os = targets.new_tensor(data.outputscale)
        if m == 0:
            result = (targets.sum(-1) * 0.0, os.expand(batch) + targets.sum(-1) * 0.0)
            return result
        xc = data.x_train[ids]
        xcs = data.x_train_scaled[ids]
        delta = (xc - targets[:, None]).transpose(-1, -2)
        delta_scaled = (xcs - targets_scaled[:, None]).transpose(-1, -2)
        h = delta_scaled.transpose(-1, -2) @ delta_scaled
        cols = h.transpose(-1, -2)
        q = cols[:, :, None, :] - cols[:, None, :, :]
        r_i = h.diagonal(dim1=-2, dim2=-1)
        r_cc = (xcs[:, :, None] - xcs[:, None, :]).square().sum(-1)
        alpha_i = _alpha_from_r(r_i, data.kernel_name, os)
        alpha_cc = _alpha_from_r(r_cc, data.kernel_name, os)
        beta_cc = _beta_from_r(r_cc, data.kernel_name, os)

        ell = data.lengthscale
        kff = function_covariance_batch(xc, xc, ell, os, data.kernel_name, centered=centered)
        k_fc = function_covariance_batch(
            xc, targets[:, None], ell, os, data.kernel_name, centered=centered
        ).squeeze(-1)
        kff = kff + data.sigma_f**2 * torch.eye(m, dtype=targets.dtype, device=targets.device)
        bar_k = (-alpha_i[:, :, None] * cols).reshape(batch, m * m)
        q_matrix = (-alpha_cc[:, :, :, None] * q).permute(0, 1, 3, 2).reshape(batch, m * m, m)
        g = alpha_cc[:, :, :, None, None] * h[:, None, None]
        g = g + beta_cc[:, :, :, None, None] * q.unsqueeze(-1) * q.unsqueeze(-2)
        if self.gradient_noise_model == "iid":
            r_matrix = delta.transpose(-1, -2) @ delta
        else:
            delta_noise = delta * ell.square().reciprocal().reshape(1, -1, 1)
            r_matrix = delta.transpose(-1, -2) @ delta_noise
        r_matrix = 0.5 * (r_matrix + r_matrix.transpose(-1, -2))
        site = torch.arange(m, device=targets.device)
        g[:, site, site] = g[:, site, site] + data.sigma_g * r_matrix[:, None]
        g = g.permute(0, 1, 3, 2, 4).reshape(batch, m * m, m * m)
        observed = (data.g_train_obs[ids] @ delta).reshape(batch, m * m)
        joint = torch.cat(
            (torch.cat((kff, q_matrix.transpose(-1, -2)), dim=-1), torch.cat((q_matrix, g), dim=-1)),
            dim=-2,
        )
        del kff, q_matrix, g
        joint = 0.5 * (joint + joint.transpose(-1, -2))
        obs = torch.cat((data.f_train_obs[ids], observed), dim=-1)
        cross = torch.cat((k_fc, bar_k), dim=-1)
        solved = solve_joint_batch(joint, torch.stack((obs, cross), dim=-1))
        mean = (cross * solved[..., 0]).sum(-1)
        var = os - (cross * solved[..., 1]).sum(-1)
        result = (mean, var.clamp_min(torch.finfo(targets.dtype).eps))
        return result
