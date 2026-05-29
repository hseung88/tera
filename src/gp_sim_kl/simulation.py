from __future__ import annotations

import torch

from gp_sim_kl.config import ExperimentConfig
from gp_sim_kl.data import SimulatedDataset
from gp_sim_kl.deroos import full_value_grad_covariance, sample_value_grad_observations_deroos
from gp_sim_kl.ordering import maximin_ordering
from gp_sim_kl.utils import (
    calibrate_isotropic_lengthscale_from_inputs,
    cholesky_with_jitter,
    dtype_from_name,
    resolve_lengthscale,
    scale_inputs,
    split_value_grad,
)

def _make_design(n: int, d: int, *, design: str, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if design == 'sobol':

        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        return eng.draw(n).to(device=device, dtype=dtype).contiguous()
    if design == 'random':
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return torch.rand((n, d), generator=gen, device=device, dtype=dtype)
    raise ValueError(f'Unknown design: {design}')

def _sample_observations(cfg: ExperimentConfig, X_train: torch.Tensor, lengthscale: torch.Tensor, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    n, d = X_train.shape
    obs_dim = n * (d + 1)
    gen = torch.Generator(device=X_train.device)
    gen.manual_seed(seed)

    if cfg.sampling == 'zeros':
        z = torch.zeros(obs_dim, device=X_train.device, dtype=X_train.dtype)
        f, g = split_value_grad(z, d)
        return f, g, z, 'zeros'

    if cfg.sampling == 'deroos':
        if cfg.sigma_g != 0.0:
            raise RuntimeError('deRoos exact sampling currently requires sigma_g=0.')
        n2 = n * n
        if n2 > cfg.deroos_sampling_max_n2:
            raise RuntimeError(
                f'deRoos exact sampling requires an n^2 by n^2 square-root update with n^2={n2}. '
                f'Increase deroos_sampling_max_n2 or lower n_train.'
            )
        f, g, z = sample_value_grad_observations_deroos(
            X_train,
            lengthscale,
            cfg.outputscale,
            cfg.kernel,
            sigma_f=cfg.sigma_f,
            sigma_g=cfg.sigma_g,
            generator=gen,
        )
        return f, g, z, 'deroos'

    if cfg.sampling != 'dense':
        raise ValueError(f'Unknown sampling backend: {cfg.sampling}')

    if obs_dim > cfg.dense_sampling_max_obs_dim:
        raise RuntimeError(
            f'Dense exact sampling would require an {obs_dim} x {obs_dim} covariance. '
            f'Use sampling: deroos for exact high-dimensional derivative-GP simulation, '
            f'increase dense_sampling_max_obs_dim, or lower n/d.'
        )
    Kzz = full_value_grad_covariance(
        X_train,
        lengthscale,
        cfg.outputscale,
        cfg.kernel,
        sigma_f=cfg.sigma_f,
        sigma_g=cfg.sigma_g,
    )
    L = cholesky_with_jitter(Kzz)
    eps = torch.randn((obs_dim,), generator=gen, device=X_train.device, dtype=X_train.dtype)
    z = L @ eps
    f, g = split_value_grad(z, d)
    return f, g, z, 'dense'

def simulate_dataset(cfg: ExperimentConfig, d: int, *, repeat: int) -> SimulatedDataset:
    with torch.no_grad():
        device = torch.device(cfg.device)
        dtype = dtype_from_name(cfg.dtype)
        base_seed = int(cfg.seed + 1000003 * repeat + 7919 * d)

        X_train_raw = _make_design(cfg.n_train, d, design=cfg.design, seed=base_seed, device=device, dtype=dtype)
        X_eval = _make_design(cfg.n_eval, d, design=cfg.design, seed=base_seed + 104729, device=device, dtype=dtype)

        if cfg.target_median_correlation is not None:
            if cfg.use_ard:
                raise ValueError('target_median_correlation calibration currently supports isotropic metrics only.')
            probe = torch.cat([X_train_raw, X_eval], dim=0)
            lengthscale = calibrate_isotropic_lengthscale_from_inputs(probe, cfg.kernel, cfg.target_median_correlation).to(device=device, dtype=dtype)
        else:
            lengthscale = resolve_lengthscale(cfg.lengthscale, d, cfg.use_ard, device=device, dtype=dtype)

        X_train_scaled_raw = scale_inputs(X_train_raw, lengthscale)
        order = maximin_ordering(X_train_scaled_raw)
        X_train = X_train_raw[order].contiguous()
        X_train_scaled = X_train_scaled_raw[order].contiguous()
        X_eval_scaled = scale_inputs(X_eval, lengthscale).contiguous()

        f_obs, g_obs, z_obs, backend = _sample_observations(cfg, X_train, lengthscale, seed=base_seed + 65537)

        return SimulatedDataset(
            X_train=X_train,
            X_train_scaled=X_train_scaled,
            X_eval=X_eval.contiguous(),
            X_eval_scaled=X_eval_scaled,
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
