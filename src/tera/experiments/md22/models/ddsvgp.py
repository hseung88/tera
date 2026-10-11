from tera.experiments.md22.models.derivative_baselines import derivative_baseline_model


class ddsvgp_model(derivative_baseline_model):
    def __init__(self, *, config, seed):
        super().__init__(name="ddsvgp", config=config, seed=seed)
