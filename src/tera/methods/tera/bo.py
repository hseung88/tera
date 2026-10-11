from __future__ import annotations

from dataclasses import dataclass
from typing import Any as any_type

import torch

from tera.methods.common.bo import initial_lengthscale
from tera.methods.common.training import regression_data
from tera.methods.common.utils import scale_inputs
from tera.methods.tera.bo_training import tera_model


class differentiable_tera_botorch_model:
    def __init__(self, model: tera_model, *, prediction_batch_size: int) -> None:
        if prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        from botorch.models.model import Model as model_type

        class __wrapped(model_type):
            _num_outputs = 1

            def __init__(self, tera_model: tera_model) -> None:
                super().__init__()
                self.tera_model = tera_model
                self.prediction_batch_size = prediction_batch_size

            @property
            def _data(self):
                data = self.tera_model.predictor.data
                if data is None:
                    raise RuntimeError("tera model has no fitted predictor data.")
                return data

            @property
            def num_outputs(self) -> int:
                return 1

            def posterior(
                self,
                X: torch.Tensor,
                output_indices: list[int] | None = None,
                observation_noise: bool = False,
                **kwargs: any_type,
            ):
                import gpytorch
                from botorch.posteriors.gpytorch import GPyTorchPosterior as g_py_torch_posterior_type
                from linear_operator.operators import DiagLinearOperator as diag_linear_operator_type

                if output_indices not in (None, [0]):
                    raise ValueError("tera BO wrapper exposes a single scalar output.")
                batch_shape, q, d = X.shape[:-2], X.shape[-2], X.shape[-1]
                flat_x = X.reshape(-1, d)
                if q != 1:
                    raise ValueError(
                        "tera supports q=1; use leading batch dimensions for candidates"
                    )
                means, variances = self._predict_candidates(flat_x)
                mean = means.reshape(*batch_shape, q, 1)
                var = variances.reshape(*batch_shape, q, 1).clamp_min(torch.finfo(X.dtype).eps)
                if observation_noise:
                    var = var + float(self.tera_model.sigma_f)
                covar = diag_linear_operator_type(var.reshape(*batch_shape, q))
                mvn = gpytorch.distributions.MultitaskMultivariateNormal(
                    mean, covar, interleaved=True
                )
                return g_py_torch_posterior_type(mvn)

            def _predict_candidates(self, flat_x):
                data = self._data
                means, variances = [], []
                k = max(0, min(int(self.tera_model.m), len(data.x_train)))
                for chunk in flat_x.split(self.prediction_batch_size):
                    scaled = scale_inputs(chunk, data.lengthscale)
                    with torch.no_grad():
                        distances = torch.cdist(
                            scaled.detach(),
                            data.x_train_scaled,
                            compute_mode="donot_use_mm_for_euclid_dist",
                        )
                        ids = distances.topk(k, largest=False).indices
                    # Call the differentiable core, not the MD22 no_grad wrapper.
                    mean, var = self.tera_model.predictor._predict_batch(
                        chunk, scaled, ids, centered=False
                    )
                    means.append(mean)
                    variances.append(var)
                return torch.cat(means), torch.cat(variances)

        self.model = __wrapped(model).eval()


@dataclass
class terabo_state:
    lengthscale_cfg: float | list[float] | None = None
    outputscale: float | None = None
    sigma_f: float | None = None
    sigma_g: float | None = None
    num_calls: int = 0
    num_refits: int = 0


def _lengthscale_to_cfg(lengthscale: torch.Tensor, use_ard: bool) -> float | list[float]:
    flat = lengthscale.detach().cpu().reshape(-1)
    if use_ard:
        return [float(x) for x in flat]
    return float(flat[0])


def _should_refit_tera(state: terabo_state | None, cfg: any_type) -> bool:
    if state is None or state.lengthscale_cfg is None:
        return True
    every = max(1, int(cfg.tera_refit_every))
    return (int(state.num_calls) % every) == 0


def fit_tera_surrogate(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_g: torch.Tensor,
    cfg: any_type,
    *,
    seed: int,
    state: terabo_state | None = None,
):
    first_fit = state is None or state.lengthscale_cfg is None
    do_refit = _should_refit_tera(state, cfg)

    if first_fit:
        ell = initial_lengthscale(
            train_x.shape[-1], cfg, device=train_x.device, dtype=train_x.dtype, method="tera"
        )
        lengthscale_cfg: float | list[float] = _lengthscale_to_cfg(ell, cfg.use_ard)
        outputscale = float(cfg.outputscale_init)
        sigma_f = float(cfg.tera_sigma_f)
        sigma_g = float(cfg.tera_sigma_g)
        train_steps = int(cfg.tera_initial_train_steps if do_refit else 0)
    else:
        assert state is not None
        lengthscale_cfg = state.lengthscale_cfg
        outputscale = float(
            state.outputscale if state.outputscale is not None else cfg.outputscale_init
        )
        sigma_f = float(state.sigma_f if state.sigma_f is not None else cfg.tera_sigma_f)
        sigma_g = float(state.sigma_g if state.sigma_g is not None else cfg.tera_sigma_g)
        train_steps = int(cfg.tera_update_train_steps if do_refit else 0)

    split = regression_data(
        x_train=train_x, y_train=train_y.reshape(-1), g_train=train_g, x_test=train_x[:0]
    )
    model = tera_model(
        m=cfg.tera_m,
        kernel=cfg.kernel,
        outputscale=outputscale,
        sigma_f=sigma_f,
        sigma_g=sigma_g,
        lengthscale=lengthscale_cfg,
        lengthscale_init="one",
        lengthscale_init_max_points=0,
        use_ard=cfg.use_ard,
        seed=seed,
        train_steps=train_steps,
        train_epochs=0,
        graph_refresh_epochs=0,
        train_batch_size=cfg.tera_train_batch_size,
        lr=cfg.tera_lr,
        weight_decay=cfg.gp_weight_decay,
        learn_lengthscale=cfg.tera_learn_lengthscale and train_steps > 0,
        learn_outputscale=cfg.tera_learn_outputscale and train_steps > 0,
        learn_sigma_f=cfg.tera_learn_sigma_f and train_steps > 0,
        learn_sigma_g=cfg.tera_learn_sigma_g and train_steps > 0,
        min_sigma_f=cfg.min_noise_var,
        min_sigma_g=0.0,
        lengthscale_min=cfg.tera_lengthscale_min,
        lengthscale_max=cfg.tera_lengthscale_max,
        log_every=0,
        gradient_noise_model=cfg.tera_gradient_noise_model,
    )
    model.fit(split)

    if state is not None:
        state.lengthscale_cfg = _lengthscale_to_cfg(model.lengthscale, cfg.use_ard)
        state.outputscale = float(model.outputscale)
        state.sigma_f = float(model.sigma_f)
        state.sigma_g = float(model.sigma_g)
        state.num_calls += 1
        if train_steps > 0:
            state.num_refits += 1

    return differentiable_tera_botorch_model(
        model, prediction_batch_size=cfg.tera_prediction_batch_size
    ).model
