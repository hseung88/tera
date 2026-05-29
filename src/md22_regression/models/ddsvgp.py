from __future__ import annotations

from md22_regression.models.external import ExternalResultsModel

class DDSVGPModel(ExternalResultsModel):
    def __init__(
        self,
        *,
        csv_path: str | None,
        seed: int,
        strict_fairness: bool = True,
        preprocessing_version: str | None = None,
        x_scale: float | None = None,
    ) -> None:
        super().__init__(
            method="DDSVGP",
            csv_path=csv_path,
            seed=seed,
            strict_fairness=strict_fairness,
            preprocessing_version=preprocessing_version,
            x_scale=x_scale,
        )
