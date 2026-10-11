from __future__ import annotations

from abc import ABC as abc_type, abstractmethod

from tera.experiments.gp_sim.data import predictive_marginals, simulated_dataset


class marginal_predictor(abc_type):
    @abstractmethod
    def build(self, data: simulated_dataset) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_f_marginals(self, x_eval) -> predictive_marginals:
        raise NotImplementedError
