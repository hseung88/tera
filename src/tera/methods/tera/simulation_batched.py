from __future__ import annotations

from dataclasses import replace

from tera.methods.common.data import simulated_dataset
from tera.methods.tera.batched_prediction import batched_tera_predictor


class batched_tera_simulation_predictor(batched_tera_predictor):
    def __init__(self, m: int, *, prediction_batch_size: int = 256) -> None:
        super().__init__(
            m=m, gradient_noise_model="iid", prediction_batch_size=prediction_batch_size
        )

    def build(self, data: simulated_dataset) -> None:
        super().build(replace(data, sigma_g=data.sigma_g**2))
