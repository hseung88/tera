from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tera.methods.common.batching import resolve_batch_size
from tera.methods.common.data import simulated_dataset
from tera.methods.common.initialization import resolve_kernel_lengthscale
from tera.methods.common.ordering import maximin_ordering, predecessor_neighbors
from tera.methods.common.prediction import prediction
from tera.methods.common.training import regression_data
from tera.methods.common.utils import scale_inputs


@dataclass(slots=True)
class vecchia_training_state:
    inputs: torch.Tensor
    y: torch.Tensor
    g: torch.Tensor
    neighbors: torch.Tensor
    sample_positions: torch.Tensor


class tera_model:
    name = 'tera'

    def __init__(
        self,
        *,
        m: int,
        kernel: str,
        outputscale: float,
        sigma_f: float,
        sigma_g: float,
        lengthscale,
        lengthscale_init: str,
        lengthscale_init_max_points: int,
        use_ard: bool,
        seed: int = 0,
        train_steps: int = 0,
        train_epochs: int = 0,
        graph_refresh_epochs: int = 0,
        train_batch_size: int | None = None,
        batch_size: int | None = None,
        prediction_batch_size: int = 32,
        lr: float = 5e-2,
        weight_decay: float = 0.0,
        learn_lengthscale: bool = True,
        learn_outputscale: bool = True,
        learn_sigma_f: bool = True,
        learn_sigma_g: bool = False,
        min_sigma_f: float = 1e-6,
        min_sigma_g: float = 0.0,
        lengthscale_min: float | None = None,
        lengthscale_max: float | None = None,
        log_every: int = 0,
        gradient_noise_model: str | None = None,
        validation_callback=None,
    ) -> None:
        if prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        self.prediction_batch_size = prediction_batch_size
        self.m = int(m)
        self.kernel = kernel
        self.outputscale_init = float(outputscale)
        self.sigma_f_init = float(sigma_f)
        self.sigma_g_init = float(sigma_g)
        self.lengthscale_cfg = lengthscale
        self.lengthscale_init = lengthscale_init
        self.lengthscale_init_max_points = int(lengthscale_init_max_points)
        self.use_ard = bool(use_ard)
        self.seed = int(seed)
        self.train_steps = int(train_steps)
        self.train_epochs = int(train_epochs)
        self.graph_refresh_epochs = int(graph_refresh_epochs)
        self.train_batch_size = resolve_batch_size(
            train_batch_size, batch_size, default=64, name="train_batch_size"
        )
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.learn_lengthscale = bool(learn_lengthscale)
        self.learn_outputscale = bool(learn_outputscale)
        self.learn_sigma_f = bool(learn_sigma_f)
        self.learn_sigma_g = bool(learn_sigma_g)
        self.min_sigma_f = float(min_sigma_f)
        self.min_sigma_g = float(min_sigma_g)
        self.lengthscale_min = None if lengthscale_min is None else float(lengthscale_min)
        self.lengthscale_max = None if lengthscale_max is None else float(lengthscale_max)
        if self.lengthscale_min is not None and self.lengthscale_min <= 0.0:
            raise ValueError("lengthscale_min must be positive when provided.")
        if self.lengthscale_max is not None and self.lengthscale_max <= 0.0:
            raise ValueError("lengthscale_max must be positive when provided.")
        if (
            self.lengthscale_min is not None
            and self.lengthscale_max is not None
            and self.lengthscale_min > self.lengthscale_max
        ):
            raise ValueError("lengthscale_min must be no larger than lengthscale_max.")
        self.log_every = int(log_every)
        if gradient_noise_model is None:
            gradient_noise_model = "scaled" if self.use_ard else "iid"
        if gradient_noise_model not in {"iid", "scaled"}:
            raise ValueError("gradient_noise_model must be either 'iid' or 'scaled'.")
        self.gradient_noise_model = gradient_noise_model
        self.validation_callback = validation_callback

        self.predictor = self._make_predictor()
        self.lengthscale: torch.Tensor | None = None
        self.outputscale: float = self.outputscale_init
        self.sigma_f: float = self.sigma_f_init
        self.sigma_g: float = self.sigma_g_init
        self.training_history: list[dict[str, float]] = []

    def _make_predictor(self):
        from .batched_prediction import batched_tera_predictor

        return batched_tera_predictor(
            self.m, self.gradient_noise_model, self.prediction_batch_size
        )

    def _batch_nll(self, **kwargs):
        return _batch_nll(**kwargs)

    def _likelihood_lengthscale(self, lengthscale):
        return self._apply_lengthscale_bounds(lengthscale)

    def _apply_lengthscale_bounds(self, lengthscale: torch.Tensor) -> torch.Tensor:
        if self.lengthscale_min is not None:
            lengthscale = lengthscale.clamp_min(float(self.lengthscale_min))
        if self.lengthscale_max is not None:
            lengthscale = lengthscale.clamp_max(float(self.lengthscale_max))
        return lengthscale

    def _project_log_lengthscale_(self, log_lengthscale: torch.Tensor) -> None:
        lower = math.log(float(self.lengthscale_min)) if self.lengthscale_min is not None else None
        upper = math.log(float(self.lengthscale_max)) if self.lengthscale_max is not None else None
        if lower is None and upper is None:
            return
        with torch.no_grad():
            log_lengthscale.clamp_(min=lower, max=upper)

    def fit(self, split: regression_data) -> None:
        init_lengthscale = resolve_kernel_lengthscale(
            split.x_train,
            lengthscale=self.lengthscale_cfg,
            lengthscale_init=self.lengthscale_init,
            lengthscale_init_max_points=self.lengthscale_init_max_points,
            use_ard=self.use_ard,
        )
        init_lengthscale = self._apply_lengthscale_bounds(init_lengthscale)
        self.training_history.clear()

        if self.train_steps > 0 or self.train_epochs > 0:
            if self._use_outer_graph_refresh():
                self.lengthscale, self.outputscale, self.sigma_f, self.sigma_g = (
                    self._train_likelihood_with_graph_refresh(
                        split,
                        init_lengthscale,
                    )
                )
            else:
                state = build_training_state(split, init_lengthscale, self.m)
                self.lengthscale, self.outputscale, self.sigma_f, self.sigma_g = (
                    self._train_likelihood(state, split, init_lengthscale)
                )
        else:
            self.lengthscale = init_lengthscale.detach().clone()
            self.outputscale = self.outputscale_init
            self.sigma_f = self.sigma_f_init
            self.sigma_g = self.sigma_g_init

        x_train_scaled = scale_inputs(split.x_train, self.lengthscale)
        dummy_eval = split.x_test[:0]
        data = simulated_dataset(
            x_train=split.x_train,
            x_train_scaled=x_train_scaled,
            x_eval=dummy_eval,
            x_eval_scaled=dummy_eval,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            sigma_f=math.sqrt(max(float(self.sigma_f), 0.0)),
            sigma_g=self.sigma_g,
            kernel_name=self.kernel,
            f_train_obs=split.y_train,
            g_train_obs=split.g_train,
            z_train_obs=torch.cat([split.y_train.unsqueeze(-1), split.g_train], dim=1).reshape(-1),
            sampling_backend="training",
        )
        self.predictor.build(data)

    def _use_outer_graph_refresh(self) -> bool:
        return (
            self.train_epochs > 0
            and self.graph_refresh_epochs > 0
            and self.use_ard
            and self.learn_lengthscale
        )

    def _train_likelihood_with_graph_refresh(
        self,
        split: regression_data,
        init_lengthscale: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, float]:
        remaining_epochs = int(self.train_epochs)
        refresh_epochs = max(1, int(self.graph_refresh_epochs))
        current_lengthscale = init_lengthscale.detach().clone()
        current_outputscale = self.outputscale_init
        current_sigma_f = self.sigma_f_init
        current_sigma_g = self.sigma_g_init
        step_offset = 0

        while remaining_epochs > 0:
            stage_epochs = min(refresh_epochs, remaining_epochs)
            state = build_training_state(split, current_lengthscale, self.m)
            current_lengthscale, current_outputscale, current_sigma_f, current_sigma_g = (
                self._train_likelihood(
                    state,
                    split,
                    current_lengthscale,
                    start_outputscale=current_outputscale,
                    start_sigma_f=current_sigma_f,
                    start_sigma_g=current_sigma_g,
                    train_epochs_override=stage_epochs,
                    train_steps_override=0,
                    step_offset=step_offset,
                )
            )
            step_offset += stage_epochs * math.ceil(
                int(state.sample_positions.numel())
                / max(1, min(self.train_batch_size, int(state.sample_positions.numel())))
            )
            remaining_epochs -= stage_epochs
        return current_lengthscale, current_outputscale, current_sigma_f, current_sigma_g

    def predict(self, inputs: torch.Tensor) -> prediction:
        pred = self.predictor.predict_f_marginals(inputs)
        return prediction(y_mean=pred.mean, y_var=pred.var)

    def _train_likelihood(
        self,
        state: vecchia_training_state,
        split: regression_data,
        init_lengthscale: torch.Tensor,
        *,
        start_outputscale: float | None = None,
        start_sigma_f: float | None = None,
        start_sigma_g: float | None = None,
        train_epochs_override: int | None = None,
        train_steps_override: int | None = None,
        step_offset: int = 0,
    ) -> tuple[torch.Tensor, float, float, float]:
        device, dtype = state.inputs.device, state.inputs.dtype
        eps = torch.finfo(dtype).eps

        init_lengthscale = self._apply_lengthscale_bounds(init_lengthscale)
        log_lengthscale = torch.nn.Parameter(
            torch.log(init_lengthscale.detach().clone().clamp_min(eps))
        )
        self._project_log_lengthscale_(log_lengthscale)
        outputscale0 = (
            self.outputscale_init if start_outputscale is None else float(start_outputscale)
        )
        sigma_f0 = self.sigma_f_init if start_sigma_f is None else float(start_sigma_f)
        sigma_g0 = self.sigma_g_init if start_sigma_g is None else float(start_sigma_g)
        log_outputscale = torch.nn.Parameter(
            torch.log(torch.tensor(outputscale0, device=device, dtype=dtype).clamp_min(eps))
        )
        log_sigma_f_raw = torch.nn.Parameter(
            torch.log(
                torch.tensor(max(sigma_f0 - self.min_sigma_f, eps), device=device, dtype=dtype)
            )
        )
        log_sigma_g_raw = torch.nn.Parameter(
            torch.log(
                torch.tensor(max(sigma_g0 - self.min_sigma_g, eps), device=device, dtype=dtype)
            )
        )

        params: list[torch.nn.Parameter] = []
        if self.learn_lengthscale:
            params.append(log_lengthscale)
        if self.learn_outputscale:
            params.append(log_outputscale)
        if self.learn_sigma_f:
            params.append(log_sigma_f_raw)
        if self.learn_sigma_g:
            params.append(log_sigma_g_raw)
        if not params:
            return init_lengthscale.detach().clone(), outputscale0, sigma_f0, sigma_g0

        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        sample_positions = state.sample_positions.cpu()
        n_sample = int(sample_positions.numel())
        train_batch_size = max(1, min(self.train_batch_size, n_sample))
        steps_per_epoch = math.ceil(n_sample / train_batch_size)
        effective_epochs = (
            self.train_epochs if train_epochs_override is None else int(train_epochs_override)
        )
        effective_steps = (
            self.train_steps if train_steps_override is None else int(train_steps_override)
        )
        total_steps = (
            effective_epochs * steps_per_epoch if effective_epochs > 0 else effective_steps
        )
        perm = torch.empty(0, dtype=torch.long)
        cursor = n_sample

        for step in range(1, total_steps + 1):
            if effective_epochs > 0:
                if perm.numel() == 0 or cursor >= n_sample:
                    perm = torch.randperm(n_sample, generator=gen)
                    cursor = 0
                draw_cpu = perm[cursor : min(cursor + train_batch_size, n_sample)]
                cursor += int(draw_cpu.numel())
                idx = sample_positions[draw_cpu]
            else:
                draw = torch.randint(0, n_sample, (train_batch_size,), generator=gen, device="cpu")
                idx = sample_positions[draw]
            lengthscale = (
                torch.exp(log_lengthscale) if self.learn_lengthscale else init_lengthscale.detach()
            )
            lengthscale = self._likelihood_lengthscale(lengthscale)
            outputscale = (
                torch.exp(log_outputscale)
                if self.learn_outputscale
                else torch.tensor(outputscale0, device=device, dtype=dtype)
            )
            sigma_f = (
                self.min_sigma_f + torch.exp(log_sigma_f_raw)
                if self.learn_sigma_f
                else torch.tensor(sigma_f0, device=device, dtype=dtype)
            )
            sigma_g = (
                self.min_sigma_g + torch.exp(log_sigma_g_raw)
                if self.learn_sigma_g
                else torch.tensor(sigma_g0, device=device, dtype=dtype)
            )

            loss = self._batch_nll(
                state=state,
                target_positions=idx,
                lengthscale=lengthscale,
                outputscale=outputscale,
                sigma_f=sigma_f,
                sigma_g=sigma_g,
                kernel=self.kernel,
                gradient_noise_model=self.gradient_noise_model,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
            opt.step()
            if self.learn_lengthscale:
                self._project_log_lengthscale_(log_lengthscale)

            if self.log_every > 0 and (
                step == 1 or step % self.log_every == 0 or step == total_steps
            ):
                cur_lengthscale = (
                    torch.exp(log_lengthscale)
                    if self.learn_lengthscale
                    else init_lengthscale.detach()
                )
                cur_lengthscale = self._apply_lengthscale_bounds(cur_lengthscale).detach().clone()
                cur_outputscale = float(
                    (
                        torch.exp(log_outputscale)
                        if self.learn_outputscale
                        else torch.tensor(outputscale0, device=device, dtype=dtype)
                    )
                    .detach()
                    .cpu()
                )
                cur_sigma_f = float(
                    (
                        (self.min_sigma_f + torch.exp(log_sigma_f_raw))
                        if self.learn_sigma_f
                        else torch.tensor(sigma_f0, device=device, dtype=dtype)
                    )
                    .detach()
                    .cpu()
                )
                cur_sigma_g = float(
                    (
                        (self.min_sigma_g + torch.exp(log_sigma_g_raw))
                        if self.learn_sigma_g
                        else torch.tensor(sigma_g0, device=device, dtype=dtype)
                    )
                    .detach()
                    .cpu()
                )
                metrics = {}
                if self.validation_callback is not None:
                    with torch.no_grad():
                        predictor = self._make_predictor()
                        x_train_scaled = scale_inputs(split.x_train, cur_lengthscale)
                        dummy_eval = split.x_test[:0]
                        data = simulated_dataset(
                            x_train=split.x_train,
                            x_train_scaled=x_train_scaled,
                            x_eval=dummy_eval,
                            x_eval_scaled=dummy_eval,
                            lengthscale=cur_lengthscale,
                            outputscale=cur_outputscale,
                            sigma_f=math.sqrt(max(float(cur_sigma_f), 0.0)),
                            sigma_g=cur_sigma_g,
                            kernel_name=self.kernel,
                            f_train_obs=split.y_train,
                            g_train_obs=split.g_train,
                            z_train_obs=torch.cat(
                                [split.y_train.unsqueeze(-1), split.g_train], dim=1
                            ).reshape(-1),
                            sampling_backend="training",
                        )
                        predictor.build(data)
                        pred = predictor.predict_f_marginals(split.x_test)
                        metrics = self.validation_callback(pred.mean)
                rec = {
                    "step": float(step_offset + step),
                    "loss": float(loss.detach().cpu()),
                    "lengthscale": float(cur_lengthscale.reshape(-1)[0].cpu()),
                    "outputscale": cur_outputscale,
                    "sigma_f": cur_sigma_f,
                    "sigma_g": cur_sigma_g,
                    **metrics,
                }
                self.training_history.append(rec)
                print(
                    f"{self.name} train "
                    f"step={int(rec['step'])} loss={rec['loss']:.6g} "
                    f"ell={rec['lengthscale']:.6g} os={rec['outputscale']:.6g} "
                    f"sf={rec['sigma_f']:.3g} sg={rec['sigma_g']:.3g} "
                    f"raw_rmse={rec.get('raw_energy_rmse_per_atom', math.nan):.6g}",
                    flush=True,
                )

        with torch.no_grad():
            final_lengthscale = (
                torch.exp(log_lengthscale) if self.learn_lengthscale else init_lengthscale
            )
            final_lengthscale = self._apply_lengthscale_bounds(final_lengthscale).detach().clone()
            final_outputscale = float(
                (
                    torch.exp(log_outputscale)
                    if self.learn_outputscale
                    else torch.tensor(outputscale0, device=device, dtype=dtype)
                )
                .detach()
                .cpu()
            )
            final_sigma_f = float(
                (
                    (self.min_sigma_f + torch.exp(log_sigma_f_raw))
                    if self.learn_sigma_f
                    else torch.tensor(sigma_f0, device=device, dtype=dtype)
                )
                .detach()
                .cpu()
            )
            final_sigma_g = float(
                (
                    (self.min_sigma_g + torch.exp(log_sigma_g_raw))
                    if self.learn_sigma_g
                    else torch.tensor(sigma_g0, device=device, dtype=dtype)
                )
                .detach()
                .cpu()
            )
        return final_lengthscale, final_outputscale, final_sigma_f, final_sigma_g


def build_training_state(
    split: regression_data, init_lengthscale: torch.Tensor, m: int
) -> vecchia_training_state:
    x_scaled = scale_inputs(split.x_train, init_lengthscale)
    order = maximin_ordering(x_scaled)
    inputs = split.x_train[order].contiguous()
    y = split.y_train[order].contiguous()
    g = split.g_train[order].contiguous()
    xs = x_scaled[order].contiguous()

    return vecchia_training_state(
        inputs=inputs,
        y=y,
        g=g,
        neighbors=predecessor_neighbors(xs, m),
        sample_positions=torch.arange(len(inputs)),
    )


def _batch_nll(
    **kwargs,
) -> torch.Tensor:

    from .batched_training import batch_nll

    return batch_nll(**kwargs)


def _alpha_from_r(r: torch.Tensor, kernel_name: str, outputscale: torch.Tensor) -> torch.Tensor:
    if kernel_name == "rbf":
        return outputscale * torch.exp(-0.5 * r)
    if kernel_name == "matern52":
        a = math.sqrt(5.0) * torch.sqrt(torch.clamp(r, min=1e-12))
        return (5.0 / 3.0) * outputscale * (1.0 + a) * torch.exp(-a)
    raise ValueError(f"Unknown kernel name: {kernel_name}")


def _beta_from_r(r: torch.Tensor, kernel_name: str, outputscale: torch.Tensor) -> torch.Tensor:
    if kernel_name == "rbf":
        return -outputscale * torch.exp(-0.5 * r)
    if kernel_name == "matern52":
        a = math.sqrt(5.0) * torch.sqrt(torch.clamp(r, min=1e-12))
        return -(25.0 / 3.0) * outputscale * torch.exp(-a)
    raise ValueError(f"Unknown kernel name: {kernel_name}")
