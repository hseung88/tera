from __future__ import annotations

import math

import torch

from tera.methods.common.deroos import k_kp_kpp_from_r
from tera.methods.common.initialization import resolve_kernel_lengthscale
from tera.methods.common.prediction import prediction
from tera.methods.common.training import regression_data
from tera.methods.common.utils import cholesky_with_jitter


def _function_covariance(
    inputs1: torch.Tensor,
    inputs2: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float | torch.Tensor,
    kernel: str,
) -> torch.Tensor:

    if lengthscale.numel() == 1:
        x1s = inputs1 / lengthscale.reshape(1)
        x2s = inputs2 / lengthscale.reshape(1)
    else:
        ell = lengthscale.reshape(1, -1)
        x1s = inputs1 / ell
        x2s = inputs2 / ell
    r = (x1s * x1s).sum(dim=1, keepdim=True) + (x2s * x2s).sum(dim=1, keepdim=True).T
    r = r - 2.0 * (x1s @ x2s.T)
    r = r.clamp_min(0.0)
    k, _, _ = k_kp_kpp_from_r(r, kernel, outputscale)
    return k


class exact_gp_model:
    validation_callback = None
    name = 'exact gp'

    def __init__(
        self,
        *,
        kernel: str,
        outputscale: float,
        sigma_f: float,
        lengthscale,
        lengthscale_init: str,
        lengthscale_init_max_points: int,
        use_ard: bool,
        max_train: int,
        train_steps: int = 0,
        train_epochs: int = 0,
        lr: float = 3e-2,
        weight_decay: float = 0.0,
        learn_lengthscale: bool = True,
        learn_outputscale: bool = True,
        learn_sigma_f: bool = True,
        min_sigma_f: float = 1e-6,
        log_every: int = 0,
    ) -> None:
        self.kernel = kernel
        self.outputscale_init = float(outputscale)
        self.sigma_f_init = float(sigma_f)
        self.lengthscale_cfg = lengthscale
        self.lengthscale_init = lengthscale_init
        self.lengthscale_init_max_points = int(lengthscale_init_max_points)
        self.use_ard = bool(use_ard)
        self.max_train = int(max_train)
        self.train_steps = int(train_steps)
        self.train_epochs = int(train_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.learn_lengthscale = bool(learn_lengthscale)
        self.learn_outputscale = bool(learn_outputscale)
        self.learn_sigma_f = bool(learn_sigma_f)
        self.min_sigma_f = float(min_sigma_f)
        self.log_every = int(log_every)
        self.x_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.lengthscale: torch.Tensor | None = None
        self.outputscale: float = self.outputscale_init
        self.sigma_f: float = self.sigma_f_init
        self.l: torch.Tensor | None = None
        self.alpha: torch.Tensor | None = None
        self.training_history: list[dict[str, float]] = []

    def fit(self, split: regression_data) -> None:
        if split.x_train.shape[0] > self.max_train:
            raise RuntimeError(
                f"exact gp skipped because n_train={split.x_train.shape[0]} exceeds standard_gp_max_train={self.max_train}"
            )
        init_lengthscale = resolve_kernel_lengthscale(
            split.x_train,
            lengthscale=self.lengthscale_cfg,
            lengthscale_init=self.lengthscale_init,
            lengthscale_init_max_points=self.lengthscale_init_max_points,
            use_ard=self.use_ard,
        )
        if self.train_steps > 0 or self.train_epochs > 0:
            self.lengthscale, self.outputscale, self.sigma_f = self._train_mll(
                split, init_lengthscale
            )
        else:
            self.lengthscale = init_lengthscale.detach().clone()
            self.outputscale = self.outputscale_init
            self.sigma_f = self.sigma_f_init
        self.x_train = split.x_train
        self.y_train = split.y_train
        covariance = _function_covariance(
            split.x_train, split.x_train, self.lengthscale, self.outputscale, self.kernel
        )
        covariance = 0.5 * (covariance + covariance.T)
        if self.sigma_f > 0.0:
            covariance = covariance + self.sigma_f * torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
        self.l = cholesky_with_jitter(covariance)
        self.alpha = torch.cholesky_solve(split.y_train.unsqueeze(-1), self.l).squeeze(-1)

    def _train_mll(
        self, split: regression_data, init_lengthscale: torch.Tensor
    ) -> tuple[torch.Tensor, float, float]:
        inputs = split.x_train
        y = split.y_train
        device, dtype = inputs.device, inputs.dtype
        eps = torch.finfo(dtype).eps
        log_lengthscale = torch.nn.Parameter(
            torch.log(init_lengthscale.detach().clone().clamp_min(eps))
        )
        log_outputscale = torch.nn.Parameter(
            torch.log(
                torch.tensor(self.outputscale_init, device=device, dtype=dtype).clamp_min(eps)
            )
        )
        log_sigma_f_raw = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    max(self.sigma_f_init - self.min_sigma_f, eps), device=device, dtype=dtype
                )
            )
        )
        params: list[torch.nn.Parameter] = []
        if self.learn_lengthscale:
            params.append(log_lengthscale)
        if self.learn_outputscale:
            params.append(log_outputscale)
        if self.learn_sigma_f:
            params.append(log_sigma_f_raw)
        if not params:
            return init_lengthscale.detach().clone(), self.outputscale_init, self.sigma_f_init
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        n = int(inputs.shape[0])
        self.training_history.clear()
        n_steps = self.train_steps if self.train_epochs <= 0 else self.train_epochs
        identity = torch.eye(n, device=device, dtype=dtype)
        for step in range(1, n_steps + 1):
            lengthscale = (
                torch.exp(log_lengthscale) if self.learn_lengthscale else init_lengthscale.detach()
            )
            outputscale = (
                torch.exp(log_outputscale)
                if self.learn_outputscale
                else torch.tensor(self.outputscale_init, device=device, dtype=dtype)
            )
            sigma_f = (
                self.min_sigma_f + torch.exp(log_sigma_f_raw)
                if self.learn_sigma_f
                else torch.tensor(self.sigma_f_init, device=device, dtype=dtype)
            )
            covariance = _function_covariance(inputs, inputs, lengthscale, outputscale, self.kernel)
            covariance = 0.5 * (covariance + covariance.T) + sigma_f * identity
            l = cholesky_with_jitter(covariance)
            alpha = torch.cholesky_solve(y.unsqueeze(-1), l).squeeze(-1)
            loss = (
                0.5 * torch.dot(y, alpha)
                + torch.sum(torch.log(torch.diagonal(l)))
                + 0.5 * n * math.log(2.0 * math.pi)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
            opt.step()
            if self.log_every > 0 and (step == 1 or step % self.log_every == 0 or step == n_steps):
                cur_lengthscale = (
                    (
                        torch.exp(log_lengthscale)
                        if self.learn_lengthscale
                        else init_lengthscale.detach()
                    )
                    .detach()
                    .clone()
                )
                cur_outputscale = float(
                    (
                        torch.exp(log_outputscale)
                        if self.learn_outputscale
                        else torch.tensor(self.outputscale_init, device=device, dtype=dtype)
                    )
                    .detach()
                    .cpu()
                )
                cur_sigma_f = float(
                    (
                        (self.min_sigma_f + torch.exp(log_sigma_f_raw))
                        if self.learn_sigma_f
                        else torch.tensor(self.sigma_f_init, device=device, dtype=dtype)
                    )
                    .detach()
                    .cpu()
                )
                metrics = {}
                if self.validation_callback is not None:
                    with torch.no_grad():
                        k_cur = _function_covariance(
                            inputs, inputs, cur_lengthscale, cur_outputscale, self.kernel
                        )
                        k_cur = 0.5 * (k_cur + k_cur.T) + cur_sigma_f * identity
                        l_cur = cholesky_with_jitter(k_cur)
                        alpha_cur = torch.cholesky_solve(y.unsqueeze(-1), l_cur).squeeze(-1)
                        y_mean = (
                            _function_covariance(
                                split.x_test, inputs, cur_lengthscale, cur_outputscale, self.kernel
                            )
                            @ alpha_cur
                        )
                        metrics = self.validation_callback(y_mean)
                ell0 = float(cur_lengthscale.reshape(-1)[0].cpu())
                rec = {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "lengthscale0": ell0,
                    "outputscale": cur_outputscale,
                    "sigma_f": cur_sigma_f,
                    **metrics,
                }
                self.training_history.append(rec)
                print(
                    "exact gp train "
                    f"step={int(rec['step'])} loss={rec['loss']:.6g} "
                    f"ell0={rec['lengthscale0']:.6g} os={rec['outputscale']:.6g} sf={rec['sigma_f']:.3g}",
                    flush=True,
                )
        with torch.no_grad():
            final_lengthscale = (
                (torch.exp(log_lengthscale) if self.learn_lengthscale else init_lengthscale)
                .detach()
                .clone()
            )
            final_outputscale = float(
                (
                    torch.exp(log_outputscale)
                    if self.learn_outputscale
                    else torch.tensor(self.outputscale_init, device=device, dtype=dtype)
                )
                .detach()
                .cpu()
            )
            final_sigma_f = float(
                (
                    (self.min_sigma_f + torch.exp(log_sigma_f_raw))
                    if self.learn_sigma_f
                    else torch.tensor(self.sigma_f_init, device=device, dtype=dtype)
                )
                .detach()
                .cpu()
            )
        return final_lengthscale, final_outputscale, final_sigma_f

    def predict(self, inputs: torch.Tensor) -> prediction:
        if self.x_train is None or self.l is None or self.alpha is None or self.lengthscale is None:
            raise RuntimeError("fit() must be called before predict().")
        k_xt = _function_covariance(
            inputs, self.x_train, self.lengthscale, self.outputscale, self.kernel
        )
        y_mean = k_xt @ self.alpha
        v = torch.cholesky_solve(k_xt.T, self.l)
        var = torch.as_tensor(self.outputscale, device=inputs.device, dtype=inputs.dtype) - torch.sum(
            k_xt * v.T, dim=1
        )
        return prediction(
            y_mean=y_mean.contiguous(), y_var=var.clamp_min(torch.finfo(inputs.dtype).eps).contiguous()
        )
