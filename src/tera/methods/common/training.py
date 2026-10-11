from dataclasses import dataclass

import torch


@dataclass
class regression_data:
    x_train: torch.Tensor
    y_train: torch.Tensor
    g_train: torch.Tensor | None = None
    x_test: torch.Tensor | None = None

    def __post_init__(self):
        if self.x_test is None:
            self.x_test = self.x_train[:0]
