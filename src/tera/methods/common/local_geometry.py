"""Radial kernels and batched nearest-neighbor search."""

import math

import torch


def _radial(r, outputscale, kind):
    if kind == "rbf":
        k = outputscale * torch.exp(-0.5 * r)
        return k, k, -k
    if kind == "matern52":
        a = math.sqrt(5.0) * r.clamp_min(1e-12).sqrt()
        exp = outputscale * torch.exp(-a)
        return exp * (1 + a + a.square() / 3), exp * (5 / 3) * (1 + a), exp * (-25 / 3)
    raise ValueError(f"Unknown kernel: {kind}")


def nearest_neighbors(train_scaled, targets_scaled, m, chunk_size=4096):

    k = min(max(int(m), 0), len(train_scaled))
    batch = len(targets_scaled)
    best = targets_scaled.new_empty(batch, 0)
    indices = torch.empty(batch, 0, dtype=torch.long, device=targets_scaled.device)
    if k == 0:
        return indices
    for start in range(0, len(train_scaled), chunk_size):
        distances = torch.cdist(targets_scaled, train_scaled[start : start + chunk_size])
        candidate_ids = torch.arange(
            start, start + distances.shape[-1], device=indices.device
        ).expand(batch, -1)
        candidates = torch.cat((best, distances), dim=-1)
        ids = torch.cat((indices, candidate_ids), dim=-1)
        best, chosen = candidates.topk(min(k, candidates.shape[-1]), largest=False, dim=-1)
        indices = ids.gather(-1, chosen)
    return indices
