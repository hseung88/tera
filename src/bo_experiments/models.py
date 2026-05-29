from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import warnings

import torch


from bo_experiments.config import BOConfig
from gp_sim_kl.models.common import FunctionConditioningCache
from gp_sim_kl.utils import cholesky_with_jitter, scale_inputs
from bo_experiments.tera import TERAModel, _alpha_from_r, _beta_from_r, _func_covariance, _projected_gradient_noise_gram

def require_botorch() -> None:
    try:
        import botorch
        import gpytorch
    except Exception as exc:
        raise ImportError("BO experiments require BoTorch and GPyTorch. Install with `pip install -e .[bo]`.") from exc


def initial_lengthscale(
    dim: int,
    cfg: BOConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    method: str = "vbo",
) -> torch.Tensor:
    if method == "vbo":
        init = cfg.vbo_lengthscale_init
        base = cfg.vbo_base_lengthscale
        lower = getattr(cfg, "vbo_lengthscale_constraint", 1.0e-4)
        upper = None
    elif method == "tera":
        init = cfg.tera_lengthscale_init
        base = cfg.tera_base_lengthscale
        lower = cfg.tera_lengthscale_min
        upper = cfg.tera_lengthscale_max
    else:
        init = cfg.lengthscale_init
        base = cfg.base_lengthscale
        lower = cfg.min_noise_var
        upper = None

    if init == "d_scaled":
        value = base * math.sqrt(float(dim))
    elif init == "prior_mode":
        if method != "vbo":
            raise ValueError("lengthscale_init='prior_mode' is only defined for VBO.")

        value = math.exp(
            float(cfg.vbo_lengthscale_prior_loc)
            + 0.5 * math.log(float(dim))
            - float(cfg.vbo_lengthscale_prior_scale) ** 2
        )
    elif init == "base":
        value = base
    elif init in {"gpytorch_default", "interval_midpoint"}:
        if upper is None:

            value = float(lower) + math.log(2.0)
        else:

            value = 0.5 * (float(lower) + float(upper))
    else:
        value = float(init)

    shape = (dim,) if cfg.use_ard else (1,)
    return torch.full(shape, float(value), device=device, dtype=dtype)

def _gp_modules(dim: int, cfg: BOConfig, *, device: torch.device, dtype: torch.dtype, method: str = "vbo"):
    require_botorch()
    import gpytorch
    from gpytorch.constraints import GreaterThan
    from gpytorch.priors import LogNormalPrior

    is_vbo = method == "vbo"
    lengthscale_prior = None
    noise_prior = None
    if is_vbo and cfg.vbo_use_priors:

        loc = float(cfg.vbo_lengthscale_prior_loc) + 0.5 * math.log(float(dim))
        lengthscale_prior = LogNormalPrior(loc=loc, scale=float(cfg.vbo_lengthscale_prior_scale))
        noise_prior = LogNormalPrior(loc=float(cfg.vbo_noise_prior_loc), scale=float(cfg.vbo_noise_prior_scale))

    kernel_name = str(cfg.vbo_kernel if is_vbo else cfg.kernel)
    ls_constraint_value = float(cfg.vbo_lengthscale_constraint if is_vbo else cfg.min_noise_var)
    noise_constraint_value = float(cfg.vbo_noise_constraint if is_vbo else cfg.min_noise_var)

    base_kwargs: dict[str, Any] = {
        "ard_num_dims": dim if cfg.use_ard else None,
        "lengthscale_constraint": GreaterThan(ls_constraint_value),
    }
    if lengthscale_prior is not None:
        base_kwargs["lengthscale_prior"] = lengthscale_prior

    if kernel_name == "matern52":
        base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, **base_kwargs).to(device=device, dtype=dtype)
    elif kernel_name == "rbf":
        base_kernel = gpytorch.kernels.RBFKernel(**base_kwargs).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported kernel: {kernel_name}")

    if not (is_vbo and str(cfg.vbo_lengthscale_init) == "gpytorch_default"):
        ell = initial_lengthscale(dim, cfg, device=device, dtype=dtype, method=method)
        base_kernel.initialize(lengthscale=ell.view(1, -1) if cfg.use_ard else ell.view(1, 1))

    if is_vbo and not bool(cfg.vbo_use_outputscale):
        covar_module = base_kernel
    else:
        covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_constraint=GreaterThan(cfg.min_noise_var),
        ).to(device=device, dtype=dtype)
        covar_module.initialize(outputscale=torch.as_tensor(cfg.outputscale_init, device=device, dtype=dtype))
        if is_vbo and cfg.vbo_fix_outputscale:
            covar_module.raw_outputscale.requires_grad_(False)

    like_kwargs: dict[str, Any] = {"noise_constraint": gpytorch.constraints.GreaterThan(noise_constraint_value)}
    if noise_prior is not None:
        like_kwargs["noise_prior"] = noise_prior
    likelihood = gpytorch.likelihoods.GaussianLikelihood(**like_kwargs).to(device=device, dtype=dtype)
    if (not is_vbo) or bool(cfg.vbo_initialize_noise):
        noise_init = float(cfg.vbo_noise_var_init if is_vbo else cfg.noise_var_init)
        noise_lower = float(cfg.vbo_noise_constraint if is_vbo else cfg.min_noise_var)

        noise_init = max(noise_init, 1.01 * noise_lower)
        likelihood.initialize(noise=torch.as_tensor(noise_init, device=device, dtype=dtype))
    return covar_module, likelihood

def _fit_vbo_mll(mll, cfg: BOConfig) -> None:
    """Fit a VBO GP through BoTorch's default MAP optimizer.

    The public VBO code routes model fitting through Ax's BoTorch modular
    interface. That path fits the ExactMarginalLogLikelihood with BoTorch's
    default MAP optimizer rather than a hand-written Adam loop. Keeping this
    helper small avoids baking a particular BoTorch version's optimizer
    signature into the experiment code.
    """
    try:
        from botorch.fit import fit_gpytorch_mll
    except ImportError:
        from botorch.fit import fit_gpytorch_model as fit_gpytorch_mll

    kwargs: dict[str, Any] = {}
    if cfg.vbo_fit_maxiter is not None:
        kwargs["options"] = {"maxiter": int(cfg.vbo_fit_maxiter)}
    try:
        fit_gpytorch_mll(mll, **kwargs)
    except TypeError:

        fit_gpytorch_mll(mll)

def fit_single_task_gp(train_X: torch.Tensor, train_Y: torch.Tensor, cfg: BOConfig, *, method: str = "vbo"):
    require_botorch()
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_Y = train_Y.reshape(-1, 1)
    covar_module, likelihood = _gp_modules(train_X.shape[-1], cfg, device=train_X.device, dtype=train_X.dtype, method=method)
    model = SingleTaskGP(train_X, train_Y, covar_module=covar_module, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train(); mll.train()
    if method == "vbo" and str(cfg.vbo_fit_backend) == "botorch_mll":
        _fit_vbo_mll(mll, cfg)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=cfg.gp_lr, weight_decay=cfg.gp_weight_decay)
        for _ in range(max(0, int(cfg.gp_train_steps))):
            opt.zero_grad(set_to_none=True)
            loss = -mll(model(train_X), train_Y.squeeze(-1)).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
            opt.step()
    return model.eval()

class DifferentiableTERABotorchModel:
    """BoTorch wrapper around TERA with differentiable fixed-neighbor marginals."""

    def __init__(self, model: TERAModel) -> None:
        require_botorch()
        from botorch.models.model import Model

        class _Wrapped(Model):
            _num_outputs = 1

            def __init__(self, tera_model: TERAModel) -> None:
                super().__init__()
                self.tera_model = tera_model
                self._function_cache = FunctionConditioningCache()
                self._function_cache.reset(self._data)

            @property
            def _data(self):
                data = self.tera_model.predictor.data
                if data is None:
                    raise RuntimeError("TERA model has no fitted predictor data.")
                return data

            @property
            def num_outputs(self) -> int:
                return 1

            def posterior(self, X: torch.Tensor, output_indices: list[int] | None = None, observation_noise: bool = False, **kwargs: Any):
                import gpytorch
                from botorch.posteriors.gpytorch import GPyTorchPosterior
                from linear_operator.operators import DiagLinearOperator

                if output_indices not in (None, [0]):
                    raise ValueError("TERA BO wrapper exposes a single scalar output.")
                batch_shape, q, d = X.shape[:-2], X.shape[-2], X.shape[-1]
                flat_X = X.reshape(-1, d)
                data = self._data
                k = min(int(self.tera_model.m), int(data.X_train.shape[0]))
                means: list[torch.Tensor] = []
                vars_: list[torch.Tensor] = []
                for x in flat_X:
                    if k <= 0:
                        idx = torch.empty(0, device=x.device, dtype=torch.long)
                    else:
                        with torch.no_grad():
                            xs = scale_inputs(x.detach().view(1, -1), data.lengthscale)
                            dists = torch.sum((data.X_train_scaled - xs) ** 2, dim=-1)
                            idx = torch.topk(dists, k=k, largest=False).indices.contiguous()
                    mu, var = self._predict_one(x.view(1, -1), idx)
                    means.append(mu); vars_.append(var)
                mean = torch.stack(means).reshape(*batch_shape, q, 1)
                var = torch.stack(vars_).reshape(*batch_shape, q, 1).clamp_min(torch.finfo(X.dtype).eps)
                if observation_noise:
                    var = var + float(self.tera_model.sigma_f)
                covar = DiagLinearOperator(var.reshape(*batch_shape, q))
                mvn = gpytorch.distributions.MultitaskMultivariateNormal(mean, covar, interleaved=True)
                return GPyTorchPosterior(mvn)

            def conditioning_indices(self, x_eval: torch.Tensor, k: int | None = None) -> torch.Tensor:
                data = self._data
                kk = min(int(self.tera_model.m if k is None else k), int(data.X_train.shape[0]))
                if kk <= 0:
                    return torch.empty(0, device=x_eval.device, dtype=torch.long)
                with torch.no_grad():
                    x2 = x_eval.detach().reshape(1, -1)
                    xs = scale_inputs(x2, data.lengthscale)
                    dists = torch.sum((data.X_train_scaled - xs) ** 2, dim=-1)
                    return torch.topk(dists, k=kk, largest=False).indices.contiguous()

            def posterior_fixed_indices(self, X: torch.Tensor, idx: torch.Tensor, observation_noise: bool = False):
                import gpytorch
                from botorch.posteriors.gpytorch import GPyTorchPosterior
                from linear_operator.operators import DiagLinearOperator

                batch_shape, q, d = X.shape[:-2], X.shape[-2], X.shape[-1]
                flat_X = X.reshape(-1, d)
                means: list[torch.Tensor] = []
                vars_: list[torch.Tensor] = []
                for x in flat_X:
                    mu, var = self._predict_one(x.view(1, -1), idx)
                    means.append(mu)
                    vars_.append(var)
                mean = torch.stack(means).reshape(*batch_shape, q, 1)
                var = torch.stack(vars_).reshape(*batch_shape, q, 1).clamp_min(torch.finfo(X.dtype).eps)
                if observation_noise:
                    var = var + float(self.tera_model.sigma_f)
                covar = DiagLinearOperator(var.reshape(*batch_shape, q))
                mvn = gpytorch.distributions.MultitaskMultivariateNormal(mean, covar, interleaved=True)
                return GPyTorchPosterior(mvn)

            def _predict_one(self, x_eval: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                data = self._data
                device, dtype = x_eval.device, x_eval.dtype
                m = int(idx.numel())
                outputscale = torch.as_tensor(data.outputscale, device=device, dtype=dtype)
                k_xx = outputscale.reshape(())
                if m == 0:
                    return x_eval.new_zeros(()), k_xx
                Xc = data.X_train[idx]
                Xcs = data.X_train_scaled[idx]
                xs = scale_inputs(x_eval, data.lengthscale)
                Lff, _ = self._function_cache.get(idx)
                Lff = Lff.to(device=device, dtype=dtype)
                k_fc = _func_covariance(Xc, x_eval, data.lengthscale, outputscale, data.kernel_name).squeeze(-1)
                delta = (Xc - x_eval).T.contiguous()
                delta_s = (Xcs - xs).T.contiguous()
                H = delta_s.T @ delta_s
                cols = H.T.contiguous()
                q_mat = cols[:, None, :] - cols[None, :, :]
                r_i = torch.diagonal(H, 0)
                alpha_i = _alpha_from_r(r_i, data.kernel_name, outputscale)
                delta_cc = Xcs[:, None, :] - Xcs[None, :, :]
                r_cc = (delta_cc * delta_cc).sum(dim=-1)
                alpha_cc = _alpha_from_r(r_cc, data.kernel_name, outputscale)
                beta_cc = _beta_from_r(r_cc, data.kernel_name, outputscale)
                bar_k = ((-alpha_i[:, None]) * cols).reshape(m * m).contiguous()
                Q = ((-alpha_cc[:, :, None]) * q_mat).permute(0, 2, 1).reshape(m * m, m).contiguous()
                G0_blocks = alpha_cc[:, :, None, None] * H.view(1, 1, m, m)
                G0_blocks = G0_blocks + beta_cc[:, :, None, None] * (q_mat[:, :, :, None] * q_mat[:, :, None, :])
                if data.sigma_g > 0.0:
                    R = _projected_gradient_noise_gram(delta, data.lengthscale, self.tera_model.gradient_noise_model)
                    diag_idx = torch.arange(m, device=device)
                    G0_blocks[diag_idx, diag_idx] = G0_blocks[diag_idx, diag_idx] + float(data.sigma_g) * R
                G0 = G0_blocks.permute(0, 2, 1, 3).reshape(m * m, m * m).contiguous()
                sol = torch.cholesky_solve(torch.cat([k_fc[:, None], Q.T], dim=1), Lff)
                beta_star, QKinv = sol[:, 0], sol[:, 1:]
                sigma2 = k_xx - torch.dot(k_fc, beta_star)
                k_delta = bar_k - Q @ beta_star
                K_delta = 0.5 * (G0 - Q @ QKinv + (G0 - Q @ QKinv).T)
                L_delta = cholesky_with_jitter(K_delta)
                w_star = torch.cholesky_solve(k_delta.unsqueeze(-1), L_delta).squeeze(-1)
                var = torch.clamp(sigma2 - torch.dot(w_star, k_delta), min=torch.finfo(dtype).eps)
                qtw = Q.T @ w_star
                f_weights = beta_star - torch.cholesky_solve(qtw.unsqueeze(-1), Lff).squeeze(-1)
                q_obs = data.g_train_obs[idx] @ delta
                mean = torch.dot(f_weights, data.f_train_obs[idx]) + torch.dot(w_star, q_obs.reshape(-1))
                return mean, var

        self.model = _Wrapped(model).eval()

@dataclass
class TERABOState:
    """Mutable BO state for warm-started TERA surrogate fitting."""

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

def _should_refit_tera(state: TERABOState | None, cfg: BOConfig) -> bool:
    if state is None or state.lengthscale_cfg is None:
        return True
    every = max(1, int(cfg.tera_refit_every))
    return (int(state.num_calls) % every) == 0

def fit_tera_surrogate(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    train_g: torch.Tensor,
    cfg: BOConfig,
    *,
    seed: int,
    state: TERABOState | None = None,
):
    first_fit = state is None or state.lengthscale_cfg is None
    do_refit = _should_refit_tera(state, cfg)

    if first_fit:
        ell = initial_lengthscale(train_X.shape[-1], cfg, device=train_X.device, dtype=train_X.dtype, method="tera")
        lengthscale_cfg: float | list[float] = _lengthscale_to_cfg(ell, cfg.use_ard)
        outputscale = float(cfg.outputscale_init)
        sigma_f = float(cfg.tera_sigma_f)
        sigma_g = float(cfg.tera_sigma_g)
        train_steps = int(cfg.tera_initial_train_steps if do_refit else 0)
    else:
        assert state is not None
        lengthscale_cfg = state.lengthscale_cfg
        outputscale = float(state.outputscale if state.outputscale is not None else cfg.outputscale_init)
        sigma_f = float(state.sigma_f if state.sigma_f is not None else cfg.tera_sigma_f)
        sigma_g = float(state.sigma_g if state.sigma_g is not None else cfg.tera_sigma_g)
        train_steps = int(cfg.tera_update_train_steps if do_refit else 0)

    split = SimpleNamespace(X_train=train_X, y_train=train_y.reshape(-1), g_train=train_g, X_test=train_X[:0])
    model = TERAModel(
        m=cfg.tera_m, kernel=cfg.kernel, outputscale=outputscale,
        sigma_f=sigma_f, sigma_g=sigma_g, lengthscale=lengthscale_cfg,
        lengthscale_init="one", lengthscale_init_max_points=0, use_ard=cfg.use_ard, seed=seed,
        train_steps=train_steps, train_epochs=0, graph_refresh_epochs=0,
        batch_size=cfg.tera_batch_size, lr=cfg.tera_lr, weight_decay=cfg.gp_weight_decay,
        learn_lengthscale=cfg.tera_learn_lengthscale and train_steps > 0,
        learn_outputscale=cfg.tera_learn_outputscale and train_steps > 0,
        learn_sigma_f=cfg.tera_learn_sigma_f and train_steps > 0,
        learn_sigma_g=cfg.tera_learn_sigma_g and train_steps > 0,
        min_sigma_f=cfg.min_noise_var, min_sigma_g=0.0,
        lengthscale_min=cfg.tera_lengthscale_min, lengthscale_max=cfg.tera_lengthscale_max,
        log_every=0, gradient_noise_model=cfg.tera_gradient_noise_model,
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

    return DifferentiableTERABotorchModel(model).model

def make_log_ei(model, best_f: torch.Tensor):
    require_botorch()
    try:
        from botorch.acquisition.analytic import LogExpectedImprovement
    except ImportError:
        from botorch.acquisition import LogExpectedImprovement
    return LogExpectedImprovement(model=model, best_f=best_f)

def _make_sobol_qmc_sampler(num_samples: int):
    require_botorch()
    try:
        from botorch.sampling.normal import SobolQMCNormalSampler
    except ImportError:
        from botorch.sampling import SobolQMCNormalSampler
    n = max(1, int(num_samples))
    try:
        return SobolQMCNormalSampler(sample_shape=torch.Size([n]))
    except TypeError:
        return SobolQMCNormalSampler(num_samples=n)

def make_vbo_acquisition(model, best_f: torch.Tensor, cfg: BOConfig):
    """Construct the VBO acquisition function.

    The public Vanilla BO implementation uses qLogNoisyExpectedImprovement.
    We keep analytic LogEI as an explicit ablation through ``vbo_acq: log_ei``.
    """
    acq_name = str(getattr(cfg, "vbo_acq", "log_nei")).lower()
    if acq_name in {"log_ei", "ei"}:
        return make_log_ei(model, best_f)
    if acq_name not in {"log_nei", "qlognei", "nei"}:
        raise ValueError(f"Unsupported VBO acquisition '{acq_name}'. Use 'log_nei' or 'log_ei'.")

    require_botorch()
    try:
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    except ImportError:
        try:
            from botorch.acquisition.monte_carlo import qLogNoisyExpectedImprovement
        except ImportError:
            from botorch.acquisition import qLogNoisyExpectedImprovement

    if not hasattr(model, "train_inputs") or len(model.train_inputs) == 0:
        raise ValueError("qLogNEI requires a fitted model with train_inputs for X_baseline.")
    X_baseline = model.train_inputs[0].detach()
    sampler = _make_sobol_qmc_sampler(int(getattr(cfg, "vbo_nei_mc_samples", 512)))
    base_kwargs: dict[str, Any] = {
        "model": model,
        "X_baseline": X_baseline,
        "sampler": sampler,
    }
    optional_kwargs: dict[str, Any] = {
        "prune_baseline": bool(getattr(cfg, "vbo_nei_prune_baseline", True)),
        "cache_root": bool(getattr(cfg, "vbo_nei_cache_root", True)),
    }
    try:
        return qLogNoisyExpectedImprovement(**base_kwargs, **optional_kwargs)
    except TypeError:
        optional_kwargs.pop("cache_root", None)
        try:
            return qLogNoisyExpectedImprovement(**base_kwargs, **optional_kwargs)
        except TypeError:
            optional_kwargs.pop("prune_baseline", None)
            return qLogNoisyExpectedImprovement(**base_kwargs, **optional_kwargs)

def optimize_log_ei(model, bounds: torch.Tensor, best_f: torch.Tensor, cfg: BOConfig, *, seed: int, method: str = "vbo") -> torch.Tensor:
    require_botorch()
    from botorch.optim import optimize_acqf

    if method == "vbo":
        acqf = make_vbo_acquisition(model, best_f, cfg)
    else:
        acqf = make_log_ei(model, best_f)

    if method == "vbo":

        raw = int(cfg.vbo_acq_raw_samples)
        restarts = int(cfg.vbo_acq_restarts)
        maxiter = int(cfg.vbo_acq_maxiter)
        options: dict[str, Any] = {
            "nonnegative": False,
            "sample_around_best": bool(cfg.vbo_acq_sample_around_best),
            "sample_around_best_sigma": float(cfg.vbo_acq_sample_around_best_sigma),
            "maxiter": maxiter,
            "batch_limit": int(cfg.vbo_acq_batch_limit),
        }
        retry = bool(cfg.vbo_acq_retry_on_optimization_warning)
    elif method == "tera":

        raw = int(cfg.tera_acq_raw_samples)
        restarts = int(cfg.tera_acq_restarts)
        maxiter = int(cfg.tera_acq_maxiter)
        options = {"maxiter": maxiter}
        retry = True
    else:
        raw = int(cfg.acq_raw_samples)
        restarts = int(cfg.acq_restarts)
        maxiter = int(cfg.acq_maxiter)
        options = {"maxiter": maxiter}
        retry = True

    def _run_optimize_acqf():
        kwargs = dict(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=restarts,
            raw_samples=raw,
            options=options,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=r"Optimization failed in `gen_candidates_scipy`.*",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=r"Optimization failed on the second try.*",
            )

            try:
                return optimize_acqf(
                    **kwargs,
                    sequential=True,
                    retry_on_optimization_warning=retry,
                )
            except TypeError:
                try:
                    return optimize_acqf(**kwargs, sequential=True)
                except TypeError:
                    return optimize_acqf(**kwargs)

    candidate, _ = _run_optimize_acqf()
    return candidate.detach().clamp(bounds[0], bounds[1])
