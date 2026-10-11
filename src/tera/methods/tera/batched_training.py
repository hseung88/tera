"""Vectorized tera conditional likelihood, preserving the scalar objective."""

from __future__ import annotations

import math

import torch

from .batched_prediction import solve_joint_batch
from .model import _alpha_from_r, _beta_from_r


def _function_covariance(inputs1, inputs2, ell, outputscale, kernel):
    # Use the training path's direct distances and Matern clamp, rather than
    # the prediction path's centered Gram distances.
    r = ((inputs1.unsqueeze(-2) - inputs2.unsqueeze(-3)) / ell).square().sum(-1)
    if kernel == "rbf":
        return outputscale * torch.exp(-0.5 * r)
    if kernel == "matern52":
        a = math.sqrt(5.0) * r.clamp_min(1e-12).sqrt()
        return outputscale * (1.0 + a + a.square() / 3.0) * torch.exp(-a)
    raise ValueError(f"Unknown kernel name: {kernel}")


def _local_observed_y_factors(
    *,
    state,
    positions,
    m,
    lengthscale,
    outputscale,
    sigma_f,
    sigma_g,
    kernel,
    gradient_noise_model,
    gram_distances,
):
    targets = state.inputs[positions]
    batch = len(positions)
    prior_var = outputscale + sigma_f
    if m == 0:
        return targets.new_zeros(batch), prior_var.clamp_min(torch.finfo(targets.dtype).eps).expand(
            batch
        )
    ids = state.neighbors[positions, :m]
    xc, yc, gc = state.inputs[ids], state.y[ids], state.g[ids]
    delta = (xc - targets[:, None]).transpose(-1, -2)
    xcs, xs = xc / lengthscale, targets / lengthscale
    delta_scaled = (xcs - xs[:, None]).transpose(-1, -2)
    h = delta_scaled.transpose(-1, -2) @ delta_scaled
    r_i = h.diagonal(dim1=-2, dim2=-1)
    r_cc = (
        (r_i.unsqueeze(-1) + r_i.unsqueeze(-2) - 2.0 * h).clamp_min(0)
        if gram_distances
        else (xcs[:, :, None] - xcs[:, None, :]).square().sum(-1)
    )
    alpha_i = _alpha_from_r(r_i, kernel, outputscale)
    alpha_cc = _alpha_from_r(r_cc, kernel, outputscale)
    beta_cc = _beta_from_r(r_cc, kernel, outputscale)
    eye = torch.eye(m, device=targets.device, dtype=targets.dtype)
    kff = _function_covariance(xc, xc, lengthscale, outputscale, kernel)
    kff = 0.5 * (kff + kff.transpose(-1, -2)) + sigma_f * eye
    k_fc = _function_covariance(xc, targets[:, None], lengthscale, outputscale, kernel).squeeze(-1)

    cols = h.transpose(-1, -2)
    q = cols[:, :, None, :] - cols[:, None, :, :]
    bar_k = (-alpha_i[:, :, None] * cols).reshape(batch, m * m)
    q_matrix = (-alpha_cc[:, :, :, None] * q).permute(0, 1, 3, 2).reshape(batch, m * m, m)
    g = alpha_cc[:, :, :, None, None] * h[:, None, None]
    g = g + beta_cc[:, :, :, None, None] * (q.unsqueeze(-1) * q.unsqueeze(-2))
    if gradient_noise_model == "iid":
        r_matrix = delta.transpose(-1, -2) @ delta
    else:
        delta_noise = delta * lengthscale.square().reciprocal().reshape(1, -1, 1)
        r_matrix = delta.transpose(-1, -2) @ delta_noise
    r_matrix = 0.5 * (r_matrix + r_matrix.transpose(-1, -2))
    site = torch.arange(m, device=targets.device)
    g[:, site, site] = g[:, site, site] + sigma_g * r_matrix[:, None]
    g = g.permute(0, 1, 3, 2, 4).reshape(batch, m * m, m * m)
    observed = (gc @ delta).reshape(batch, m * m)
    joint = torch.cat(
        (torch.cat((kff, q_matrix.transpose(-1, -2)), dim=-1), torch.cat((q_matrix, g), dim=-1)),
        dim=-2,
    )
    del kff, q_matrix, g
    joint = 0.5 * (joint + joint.transpose(-1, -2))
    obs = torch.cat((yc, observed), dim=-1)
    cross = torch.cat((k_fc, bar_k), dim=-1)
    solved = solve_joint_batch(joint, torch.stack((obs, cross), dim=-1))
    mean = (cross * solved[..., 0]).sum(-1)
    var = (prior_var - (cross * solved[..., 1]).sum(-1)).clamp_min(torch.finfo(targets.dtype).eps)
    return mean, var


def batch_nll(
    *,
    state,
    target_positions,
    lengthscale,
    outputscale,
    sigma_f,
    sigma_g,
    kernel,
    gradient_noise_model,
    gram_distances=False,
):
    """Average conditional NLL over targets grouped by neighborhood size."""
    if gradient_noise_model not in {"iid", "scaled"}:
        raise ValueError("gradient_noise_model must be either 'iid' or 'scaled'.")
    positions = target_positions.to(device="cpu", dtype=torch.long)
    if positions.numel() == 0:
        raise ValueError("tera training batch must contain at least one target")
    counts = positions.clamp_max(state.neighbors.shape[1])
    total = state.inputs.new_zeros(())
    for m in counts.unique().tolist():
        pos = positions[counts == m].to(state.inputs.device)
        mean, var = _local_observed_y_factors(
            state=state,
            positions=pos,
            m=m,
            lengthscale=lengthscale,
            outputscale=outputscale,
            sigma_f=sigma_f,
            sigma_g=sigma_g,
            kernel=kernel,
            gradient_noise_model=gradient_noise_model,
            gram_distances=gram_distances,
        )
        total = (
            total
            + 0.5
            * (var.log() + (state.y[pos] - mean).square() / var + math.log(2.0 * math.pi)).sum()
        )
    return total / positions.numel()
