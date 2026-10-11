from __future__ import annotations

import torch

from tera.experiments.gp_sim.config import experiment_config
from tera.experiments.gp_sim.data import simulated_dataset
from tera.experiments.gp_sim.deroos import (
    full_value_grad_covariance,
    sample_value_grad_observations_deroos,
)
from tera.experiments.gp_sim.ordering import maximin_ordering
from tera.experiments.gp_sim.utils import (
    calibrate_isotropic_lengthscale_from_inputs,
    cholesky_with_jitter,
    dtype_from_name,
    resolve_lengthscale,
    scale_inputs,
    split_value_grad,
)
from tera.methods.common.random import random_stream, seed_for


def _make_design(
    n: int, d: int, *, design: str, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if design == "sobol":
        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        return eng.draw(n).to(device=device, dtype=dtype).contiguous()
    if design == "random":
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return torch.rand((n, d), generator=gen, device=device, dtype=dtype)
    raise ValueError(f"Unknown design: {design}")


def _sample_observations(
    cfg: experiment_config, x_train: torch.Tensor, lengthscale: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    n, d = x_train.shape
    obs_dim = n * (d + 1)
    gen = torch.Generator(device=x_train.device)
    gen.manual_seed(seed)

    if cfg.sampling == "zeros":
        z = torch.zeros(obs_dim, device=x_train.device, dtype=x_train.dtype)
        f, g = split_value_grad(z, d)
        return f, g, z, "zeros"

    if cfg.sampling == "deroos":
        if cfg.sigma_g != 0.0:
            raise RuntimeError("deRoos exact sampling currently requires sigma_g=0.")
        n2 = n * n
        if n2 > cfg.deroos_sampling_max_n2:
            raise RuntimeError(
                f"deRoos exact sampling requires an n^2 by n^2 square-root update with n^2={n2}. "
                f"Increase deroos_sampling_max_n2 or lower n_train."
            )
        f, g, z = sample_value_grad_observations_deroos(
            x_train,
            lengthscale,
            cfg.outputscale,
            cfg.kernel,
            sigma_f=cfg.sigma_f,
            sigma_g=cfg.sigma_g,
            generator=gen,
        )
        return f, g, z, "deroos"

    if cfg.sampling != "dense":
        raise ValueError(f"Unknown sampling backend: {cfg.sampling}")

    if obs_dim > cfg.dense_sampling_max_obs_dim:
        raise RuntimeError(
            f"Dense exact sampling would require an {obs_dim} x {obs_dim} covariance. "
            f"Use sampling: deroos for exact high-dimensional derivative-GP simulation, "
            f"increase dense_sampling_max_obs_dim, or lower n/d."
        )
    kzz = full_value_grad_covariance(
        x_train,
        lengthscale,
        cfg.outputscale,
        cfg.kernel,
        sigma_f=cfg.sigma_f,
        sigma_g=cfg.sigma_g,
    )
    l = cholesky_with_jitter(kzz)
    eps = torch.randn((obs_dim,), generator=gen, device=x_train.device, dtype=x_train.dtype)
    z = l @ eps
    f, g = split_value_grad(z, d)
    return f, g, z, "dense"


def simulate_dataset(cfg: experiment_config, d: int, *, repeat: int) -> simulated_dataset:
    with torch.no_grad():
        device = torch.device(cfg.device)
        dtype = dtype_from_name(cfg.dtype)
        train_seed = seed_for(cfg.seed, random_stream.gp_train_inputs, repeat, d)
        eval_seed = seed_for(cfg.seed, random_stream.gp_eval_inputs, repeat, d)
        observation_seed = seed_for(cfg.seed, random_stream.gp_observations, repeat, d)

        x_train_raw = _make_design(
            cfg.n_train, d, design=cfg.design, seed=train_seed, device=device, dtype=dtype
        )
        x_eval = _make_design(
            cfg.n_eval, d, design=cfg.design, seed=eval_seed, device=device, dtype=dtype
        )

        if cfg.target_median_correlation is not None:
            if cfg.use_ard:
                raise ValueError(
                    "target_median_correlation calibration currently supports isotropic metrics only."
                )
            probe = torch.cat([x_train_raw, x_eval], dim=0)
            lengthscale = calibrate_isotropic_lengthscale_from_inputs(
                probe, cfg.kernel, cfg.target_median_correlation
            ).to(device=device, dtype=dtype)
        else:
            lengthscale = resolve_lengthscale(
                cfg.lengthscale, d, cfg.use_ard, device=device, dtype=dtype
            )

        x_train_scaled_raw = scale_inputs(x_train_raw, lengthscale)
        order = maximin_ordering(x_train_scaled_raw)
        x_train = x_train_raw[order].contiguous()
        x_train_scaled = x_train_scaled_raw[order].contiguous()
        x_eval_scaled = scale_inputs(x_eval, lengthscale).contiguous()

        f_obs, g_obs, z_obs, backend = _sample_observations(
            cfg, x_train, lengthscale, seed=observation_seed
        )

        return simulated_dataset(
            x_train=x_train,
            x_train_scaled=x_train_scaled,
            x_eval=x_eval.contiguous(),
            x_eval_scaled=x_eval_scaled,
            lengthscale=lengthscale,
            outputscale=cfg.outputscale,
            sigma_f=cfg.sigma_f,
            sigma_g=cfg.sigma_g,
            kernel_name=cfg.kernel,
            f_train_obs=f_obs.contiguous(),
            g_train_obs=g_obs.contiguous(),
            z_train_obs=z_obs.contiguous(),
            sampling_backend=backend,
        )
