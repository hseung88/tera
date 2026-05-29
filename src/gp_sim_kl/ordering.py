from __future__ import annotations

import torch

def maximin_ordering(x_scaled: torch.Tensor) -> torch.Tensor:
    n = x_scaled.shape[0]
    order = torch.empty(n, dtype=torch.long, device=x_scaled.device)
    order[0] = 0
    dist0 = torch.cdist(x_scaled, x_scaled[0:1]).squeeze(1)
    min_dist = dist0.clone()
    min_dist[0] = -1.0
    for t in range(1, n):
        idx = torch.argmax(min_dist)
        order[t] = idx
        d_new = torch.cdist(x_scaled, x_scaled[idx:idx + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, d_new)
        min_dist[idx] = -1.0
    return order

def knn_to_eval(x_train_scaled: torch.Tensor, x_eval_scaled: torch.Tensor, m: int) -> list[torch.Tensor]:
    if m <= 0:
        return [torch.empty(0, dtype=torch.long, device=x_train_scaled.device) for _ in range(x_eval_scaled.shape[0])]
    dists = torch.cdist(x_eval_scaled, x_train_scaled)
    k = min(m, x_train_scaled.shape[0])
    nn = torch.topk(dists, k=k, largest=False).indices
    return [nn[i].contiguous() for i in range(nn.shape[0])]
