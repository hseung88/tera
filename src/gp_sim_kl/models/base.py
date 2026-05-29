from __future__ import annotations

from abc import ABC, abstractmethod

from gp_sim_kl.data import PredictiveMarginals, SimulatedDataset

class MarginalPredictor(ABC):
    @abstractmethod
    def build(self, data: SimulatedDataset) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_f_marginals(self, X_eval) -> PredictiveMarginals:
        raise NotImplementedError
