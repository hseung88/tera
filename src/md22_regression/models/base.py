from __future__ import annotations

from dataclasses import dataclass
import torch

from md22_regression.data import MD22Split

@dataclass(slots=True)
class Prediction:
    y_mean: torch.Tensor
    y_var: torch.Tensor | None = None
