from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gp_sim_kl.data import PredictiveMarginals, SimulatedDataset
from gp_sim_kl.deroos import function_covariance
from gp_sim_kl.models.common import FunctionConditioningCache
from gp_sim_kl.ordering import knn_to_eval, maximin_ordering
from gp_sim_kl.utils import cholesky_with_jitter, scale_inputs
from md22_regression.data import MD22Split
from md22_regression.metrics import normalized_energy_rmse_per_atom, raw_energy_rmse_per_atom
from md22_regression.models.base import Prediction
from md22_regression.models.common import resolve_kernel_lengthscale

@dataclass(slots=True)
class _VecchiaTrainingState:
    X: torch.Tensor
    y: torch.Tensor
    g: torch.Tensor
    cond_sets: list[torch.Tensor]
    sample_positions: torch.Tensor

class _MD22TERAPredictor:
    def __init__(self, m: int, gradient_noise_model: str) -> None:
        if gradient_noise_model not in {"iid", "scaled"}:
            raise ValueError("gradient_noise_model must be either 'iid' or 'scaled'.")
        self.m = int(m)
        self.gradient_noise_model = gradient_noise_model
        self.data: SimulatedDataset | None = None
        self._function_cache = FunctionConditioningCache()

    def build(self, data: SimulatedDataset) -> None:
        self.data = data
        self._function_cache.reset(data)

    def predict_f_marginals(self, X_eval: torch.Tensor) -> PredictiveMarginals:
        if self.data is None:
            raise RuntimeError("build() must be called before prediction.")
        data = self.data
        X_eval_scaled = scale_inputs(X_eval, data.lengthscale)
        neighborhoods = knn_to_eval(data.X_train_scaled, X_eval_scaled, self.m)
        means: list[torch.Tensor] = []
        vars_: list[torch.Tensor] = []
        with torch.no_grad():
            for j, idx in enumerate(neighborhoods):
                mean_j, var_j = self._predict_one(
                    x_eval=X_eval[j:j + 1],
                    x_eval_scaled=X_eval_scaled[j:j + 1],
                    idx=idx,
                )
                means.append(mean_j)
                vars_.append(var_j)
        return PredictiveMarginals(mean=torch.stack(means).contiguous(), var=torch.stack(vars_).contiguous())

    def _predict_one(self, *, x_eval: torch.Tensor, x_eval_scaled: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.data is not None
        data = self.data
        device = x_eval.device
        dtype = x_eval.dtype

        m_local = int(idx.numel())
        k_xx = function_covariance(x_eval, x_eval, data.lengthscale, data.outputscale, data.kernel_name).reshape(()).to(dtype=dtype)
        if m_local == 0:
            return x_eval.new_zeros(()), k_xx

        Xc = data.X_train[idx]
        Xc_scaled = data.X_train_scaled[idx]
        Lff, _ = self._function_cache.get(idx)

        k_fc = function_covariance(Xc, x_eval, data.lengthscale, data.outputscale, data.kernel_name).squeeze(-1)

        delta = (Xc - x_eval).T.contiguous()
        delta_scaled = (Xc_scaled - x_eval_scaled).T.contiguous()
        H = delta_scaled.T @ delta_scaled
        cols = H.T.contiguous()
        q = cols[:, None, :] - cols[None, :, :]

        r_i = torch.diagonal(H, 0)
        alpha_i = _alpha_from_r(r_i, data.kernel_name, torch.as_tensor(data.outputscale, device=device, dtype=dtype))

        delta_cc = Xc_scaled[:, None, :] - Xc_scaled[None, :, :]
        r_cc = (delta_cc * delta_cc).sum(dim=-1)
        outputscale = torch.as_tensor(data.outputscale, device=device, dtype=dtype)
        alpha_cc = _alpha_from_r(r_cc, data.kernel_name, outputscale)
        beta_cc = _beta_from_r(r_cc, data.kernel_name, outputscale)

        bar_k = ((-alpha_i[:, None]) * cols).reshape(m_local * m_local).contiguous()
        Q = ((-alpha_cc[:, :, None]) * q).permute(0, 2, 1).reshape(m_local * m_local, m_local).contiguous()

        G0_blocks = alpha_cc[:, :, None, None] * H.view(1, 1, m_local, m_local)
        G0_blocks = G0_blocks + beta_cc[:, :, None, None] * (q[:, :, :, None] * q[:, :, None, :])
        if data.sigma_g > 0.0:
            R = _projected_gradient_noise_gram(delta, data.lengthscale, self.gradient_noise_model)
            diag_idx = torch.arange(m_local, device=device)
            G0_blocks[diag_idx, diag_idx] = G0_blocks[diag_idx, diag_idx] + (data.sigma_g) * R
        G0 = G0_blocks.permute(0, 2, 1, 3).reshape(m_local * m_local, m_local * m_local).contiguous()

        rhs = torch.cat([k_fc[:, None], Q.T], dim=1)
        sol = torch.cholesky_solve(rhs, Lff)
        beta_star = sol[:, 0]
        QKinv = sol[:, 1:]

        sigma2 = k_xx - torch.dot(k_fc, beta_star)
        k_delta = bar_k - Q @ beta_star
        K_delta = G0 - Q @ QKinv
        K_delta = 0.5 * (K_delta + K_delta.T)
        L_delta = cholesky_with_jitter(K_delta)
        w_star = torch.cholesky_solve(k_delta.unsqueeze(-1), L_delta).squeeze(-1)

        reduction = torch.dot(w_star, k_delta)
        var = torch.clamp(sigma2 - reduction, min=torch.finfo(dtype).eps)

        qtw = Q.T @ w_star
        f_weights_local = beta_star - torch.cholesky_solve(qtw.unsqueeze(-1), Lff).squeeze(-1)
        q_obs = data.g_train_obs[idx] @ delta
        mean = torch.dot(f_weights_local, data.f_train_obs[idx]) + torch.dot(w_star, q_obs.reshape(-1))
        return mean, var

class TERAModel:
    r"""TERA with minibatched projected conditional Vecchia likelihood training.

    Training follows the manuscript's current scalar-factor objective:
    each ordered target y_i is scored by p_theta(y_i | y_c(i), q_c(i)), where
    q_c(i) = (I_m \otimes D_i^T) g_c(i).  Conditioning sets are fixed during
    the inner optimization stage.  Prediction uses the single-target scheme only.
    """

    name = "TERA"

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
        batch_size: int = 64,
        lr: float = 5e-2,
        weight_decay: float = 0.0,
        learn_lengthscale: bool = True,
        learn_outputscale: bool = True,
        learn_sigma_f: bool = True,
        learn_sigma_g: bool = False,
        min_sigma_f: float = 1e-6,
        min_sigma_g: float = 0.0,
        log_every: int = 0,
        gradient_noise_model: str = "iid",
    ) -> None:
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
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.learn_lengthscale = bool(learn_lengthscale)
        self.learn_outputscale = bool(learn_outputscale)
        self.learn_sigma_f = bool(learn_sigma_f)
        self.learn_sigma_g = bool(learn_sigma_g)
        self.min_sigma_f = float(min_sigma_f)
        self.min_sigma_g = float(min_sigma_g)
        self.log_every = int(log_every)
        if gradient_noise_model not in {"iid", "scaled"}:
            raise ValueError("gradient_noise_model must be either 'iid' or 'scaled'.")
        self.gradient_noise_model = gradient_noise_model

        self.predictor = _MD22TERAPredictor(m=self.m, gradient_noise_model=self.gradient_noise_model)
        self.lengthscale: torch.Tensor | None = None
        self.outputscale: float = self.outputscale_init
        self.sigma_f: float = self.sigma_f_init
        self.sigma_g: float = self.sigma_g_init
        self.training_history: list[dict[str, float]] = []

    def fit(self, split: MD22Split) -> None:
        init_lengthscale = resolve_kernel_lengthscale(
            split.X_train,
            lengthscale=self.lengthscale_cfg,
            lengthscale_init=self.lengthscale_init,
            lengthscale_init_max_points=self.lengthscale_init_max_points,
            use_ard=self.use_ard,
        )
        self.training_history.clear()

        if self.train_steps > 0 or self.train_epochs > 0:
            if self._use_outer_graph_refresh():
                self.lengthscale, self.outputscale, self.sigma_f, self.sigma_g = self._train_likelihood_with_graph_refresh(
                    split,
                    init_lengthscale,
                )
            else:
                state = _build_fixed_vecchia_state(split, init_lengthscale, self.m)
                self.lengthscale, self.outputscale, self.sigma_f, self.sigma_g = self._train_likelihood(state, split, init_lengthscale)
        else:
            self.lengthscale = init_lengthscale.detach().clone()
            self.outputscale = self.outputscale_init
            self.sigma_f = self.sigma_f_init
            self.sigma_g = self.sigma_g_init

        X_train_scaled = scale_inputs(split.X_train, self.lengthscale)
        dummy_eval = split.X_test[:0]
        data = SimulatedDataset(
            X_train=split.X_train,
            X_train_scaled=X_train_scaled,
            X_eval=dummy_eval,
            X_eval_scaled=dummy_eval,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            sigma_f=math.sqrt(max(float(self.sigma_f), 0.0)),
            sigma_g=self.sigma_g,
            kernel_name=self.kernel,
            f_train_obs=split.y_train,
            g_train_obs=split.g_train,
            z_train_obs=torch.cat([split.y_train.unsqueeze(-1), split.g_train], dim=1).reshape(-1),
            sampling_backend="md22",
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
        split: MD22Split,
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
            state = _build_fixed_vecchia_state(split, current_lengthscale, self.m)
            current_lengthscale, current_outputscale, current_sigma_f, current_sigma_g = self._train_likelihood(
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
            step_offset += stage_epochs * math.ceil(int(state.sample_positions.numel()) / max(1, min(self.batch_size, int(state.sample_positions.numel()))))
            remaining_epochs -= stage_epochs
        return current_lengthscale, current_outputscale, current_sigma_f, current_sigma_g

    def predict(self, X: torch.Tensor) -> Prediction:
        pred = self.predictor.predict_f_marginals(X)
        return Prediction(y_mean=pred.mean, y_var=pred.var)

    def _train_likelihood(
        self,
        state: _VecchiaTrainingState,
        split: MD22Split,
        init_lengthscale: torch.Tensor,
        *,
        start_outputscale: float | None = None,
        start_sigma_f: float | None = None,
        start_sigma_g: float | None = None,
        train_epochs_override: int | None = None,
        train_steps_override: int | None = None,
        step_offset: int = 0,
    ) -> tuple[torch.Tensor, float, float, float]:
        device, dtype = state.X.device, state.X.dtype
        eps = torch.finfo(dtype).eps

        log_lengthscale = torch.nn.Parameter(torch.log(init_lengthscale.detach().clone().clamp_min(eps)))
        outputscale0 = self.outputscale_init if start_outputscale is None else float(start_outputscale)
        sigma_f0 = self.sigma_f_init if start_sigma_f is None else float(start_sigma_f)
        sigma_g0 = self.sigma_g_init if start_sigma_g is None else float(start_sigma_g)
        log_outputscale = torch.nn.Parameter(torch.log(torch.tensor(outputscale0, device=device, dtype=dtype).clamp_min(eps)))
        log_sigma_f_raw = torch.nn.Parameter(torch.log(torch.tensor(max(sigma_f0 - self.min_sigma_f, eps), device=device, dtype=dtype)))
        log_sigma_g_raw = torch.nn.Parameter(torch.log(torch.tensor(max(sigma_g0 - self.min_sigma_g, eps), device=device, dtype=dtype)))

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
        sample_positions = state.sample_positions.to(device=device)
        n_sample = int(sample_positions.numel())
        batch_size = max(1, min(self.batch_size, n_sample))
        steps_per_epoch = math.ceil(n_sample / batch_size)
        effective_epochs = self.train_epochs if train_epochs_override is None else int(train_epochs_override)
        effective_steps = self.train_steps if train_steps_override is None else int(train_steps_override)
        total_steps = effective_epochs * steps_per_epoch if effective_epochs > 0 else effective_steps
        perm = torch.empty(0, dtype=torch.long)
        cursor = n_sample

        for step in range(1, total_steps + 1):
            if effective_epochs > 0:
                if perm.numel() == 0 or cursor >= n_sample:
                    perm = torch.randperm(n_sample, generator=gen)
                    cursor = 0
                draw_cpu = perm[cursor: min(cursor + batch_size, n_sample)]
                cursor += int(draw_cpu.numel())
                idx = sample_positions[draw_cpu.to(device=device)]
            else:
                draw = torch.randint(0, n_sample, (batch_size,), generator=gen, device="cpu").to(device=device)
                idx = sample_positions[draw]
            lengthscale = torch.exp(log_lengthscale) if self.learn_lengthscale else init_lengthscale.detach()
            outputscale = torch.exp(log_outputscale) if self.learn_outputscale else torch.tensor(outputscale0, device=device, dtype=dtype)
            sigma_f = self.min_sigma_f + torch.exp(log_sigma_f_raw) if self.learn_sigma_f else torch.tensor(sigma_f0, device=device, dtype=dtype)
            sigma_g = self.min_sigma_g + torch.exp(log_sigma_g_raw) if self.learn_sigma_g else torch.tensor(sigma_g0, device=device, dtype=dtype)

            loss = _batch_nll(
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

            if self.log_every > 0 and (step == 1 or step % self.log_every == 0 or step == total_steps):
                cur_lengthscale = (torch.exp(log_lengthscale) if self.learn_lengthscale else init_lengthscale.detach()).detach().clone()
                cur_outputscale = float((torch.exp(log_outputscale) if self.learn_outputscale else torch.tensor(outputscale0, device=device, dtype=dtype)).detach().cpu())
                cur_sigma_f = float(((self.min_sigma_f + torch.exp(log_sigma_f_raw)) if self.learn_sigma_f else torch.tensor(sigma_f0, device=device, dtype=dtype)).detach().cpu())
                cur_sigma_g = float(((self.min_sigma_g + torch.exp(log_sigma_g_raw)) if self.learn_sigma_g else torch.tensor(sigma_g0, device=device, dtype=dtype)).detach().cpu())
                with torch.no_grad():
                    predictor = _MD22TERAPredictor(m=self.m, gradient_noise_model=self.gradient_noise_model)
                    X_train_scaled = scale_inputs(split.X_train, cur_lengthscale)
                    dummy_eval = split.X_test[:0]
                    data = SimulatedDataset(
                        X_train=split.X_train,
                        X_train_scaled=X_train_scaled,
                        X_eval=dummy_eval,
                        X_eval_scaled=dummy_eval,
                        lengthscale=cur_lengthscale,
                        outputscale=cur_outputscale,
                        sigma_f=math.sqrt(max(float(cur_sigma_f), 0.0)),
                        sigma_g=cur_sigma_g,
                        kernel_name=self.kernel,
                        f_train_obs=split.y_train,
                        g_train_obs=split.g_train,
                        z_train_obs=torch.cat([split.y_train.unsqueeze(-1), split.g_train], dim=1).reshape(-1),
                        sampling_backend="md22",
                    )
                    predictor.build(data)
                    pred = predictor.predict_f_marginals(split.X_test)
                    norm_rmse = normalized_energy_rmse_per_atom(pred.mean, split.y_test, split.n_atoms)
                    raw_rmse = raw_energy_rmse_per_atom(
                        pred.mean,
                        split.E_test,
                        energy_mean=split.scaler.energy_mean,
                        energy_std=split.scaler.energy_std,
                        n_atoms=split.n_atoms,
                    )
                rec = {
                    "step": float(step_offset + step),
                    "loss": float(loss.detach().cpu()),
                    "lengthscale": float(cur_lengthscale.reshape(-1)[0].cpu()),
                    "outputscale": cur_outputscale,
                    "sigma_f": cur_sigma_f,
                    "sigma_g": cur_sigma_g,
                    "normalized_energy_rmse_per_atom": norm_rmse,
                    "raw_energy_rmse_per_atom": raw_rmse,
                }
                self.training_history.append(rec)
                print(
                    "TERA train "
                    f"step={int(rec['step'])} loss={rec['loss']:.6g} "
                    f"ell={rec['lengthscale']:.6g} os={rec['outputscale']:.6g} "
                    f"sf={rec['sigma_f']:.3g} sg={rec['sigma_g']:.3g} raw_rmse={rec['raw_energy_rmse_per_atom']:.6g}",
                    flush=True,
                )

        with torch.no_grad():
            final_lengthscale = (torch.exp(log_lengthscale) if self.learn_lengthscale else init_lengthscale).detach().clone()
            final_outputscale = float((torch.exp(log_outputscale) if self.learn_outputscale else torch.tensor(outputscale0, device=device, dtype=dtype)).detach().cpu())
            final_sigma_f = float(((self.min_sigma_f + torch.exp(log_sigma_f_raw)) if self.learn_sigma_f else torch.tensor(sigma_f0, device=device, dtype=dtype)).detach().cpu())
            final_sigma_g = float(((self.min_sigma_g + torch.exp(log_sigma_g_raw)) if self.learn_sigma_g else torch.tensor(sigma_g0, device=device, dtype=dtype)).detach().cpu())
        return final_lengthscale, final_outputscale, final_sigma_f, final_sigma_g

def _build_fixed_vecchia_state(split: MD22Split, init_lengthscale: torch.Tensor, m: int) -> _VecchiaTrainingState:
    X_scaled = scale_inputs(split.X_train, init_lengthscale)
    order = maximin_ordering(X_scaled)
    X = split.X_train[order].contiguous()
    y = split.y_train[order].contiguous()
    g = split.g_train[order].contiguous()
    Xs = X_scaled[order].contiguous()

    cond_sets: list[torch.Tensor] = []
    for pos in range(X.shape[0]):
        if pos == 0 or m <= 0:
            cond_sets.append(torch.empty(0, dtype=torch.long, device=X.device))
            continue
        prev = torch.arange(pos, device=X.device, dtype=torch.long)
        k = min(int(m), int(pos))
        dists = torch.sum((Xs[prev] - Xs[pos:pos + 1]) ** 2, dim=1)
        nn = torch.topk(dists, k=k, largest=False).indices
        cond_sets.append(prev[nn].contiguous())
    sample_positions = torch.arange(X.shape[0], device=X.device, dtype=torch.long)
    return _VecchiaTrainingState(X=X, y=y, g=g, cond_sets=cond_sets, sample_positions=sample_positions)

def _batch_nll(
    *,
    state: _VecchiaTrainingState,
    target_positions: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
    sigma_f: torch.Tensor,
    sigma_g: torch.Tensor,
    kernel: str,
    gradient_noise_model: str,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for pos_t in target_positions:
        pos = int(pos_t.detach().item())
        mean, var = _local_observed_y_factor(
            X=state.X,
            y=state.y,
            g=state.g,
            target_pos=pos,
            cond_idx=state.cond_sets[pos],
            lengthscale=lengthscale,
            outputscale=outputscale,
            sigma_f=sigma_f,
            sigma_g=sigma_g,
            kernel=kernel,
            gradient_noise_model=gradient_noise_model,
        )
        resid = state.y[pos] - mean
        terms.append(0.5 * (torch.log(var) + resid * resid / var + math.log(2.0 * math.pi)))
    return torch.stack(terms).mean()

def _local_observed_y_factor(
    *,
    X: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    target_pos: int,
    cond_idx: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
    sigma_f: torch.Tensor,
    sigma_g: torch.Tensor,
    kernel: str,
    gradient_noise_model: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = X.device, X.dtype
    x = X[target_pos:target_pos + 1]
    y_target_var = outputscale + sigma_f
    m_local = int(cond_idx.numel())
    if m_local == 0:
        return X.new_zeros(()), y_target_var.clamp_min(torch.finfo(dtype).eps)

    Xc = X[cond_idx]
    yc = y[cond_idx]
    gc = g[cond_idx]
    x_scaled = scale_inputs(x, lengthscale)
    Xc_scaled = scale_inputs(Xc, lengthscale)

    Kff = _func_covariance(Xc, Xc, lengthscale, outputscale, kernel)
    Kff = 0.5 * (Kff + Kff.T)
    Kff = Kff + sigma_f * torch.eye(m_local, device=device, dtype=dtype)
    Lff = cholesky_with_jitter(Kff)

    k_fc = _func_covariance(Xc, x, lengthscale, outputscale, kernel).squeeze(-1)

    delta = (Xc - x).T.contiguous()
    delta_scaled = (Xc_scaled - x_scaled).T.contiguous()
    H = delta_scaled.T @ delta_scaled
    cols = H.T.contiguous()
    q = cols[:, None, :] - cols[None, :, :]

    r_i = torch.diagonal(H, 0)
    alpha_i = _alpha_from_r(r_i, kernel, outputscale)

    delta_cc = Xc_scaled[:, None, :] - Xc_scaled[None, :, :]
    r_cc = (delta_cc * delta_cc).sum(dim=-1)
    alpha_cc = _alpha_from_r(r_cc, kernel, outputscale)
    beta_cc = _beta_from_r(r_cc, kernel, outputscale)

    bar_k = ((-alpha_i[:, None]) * cols).reshape(m_local * m_local).contiguous()
    Q = ((-alpha_cc[:, :, None]) * q).permute(0, 2, 1).reshape(m_local * m_local, m_local).contiguous()

    G0_blocks = alpha_cc[:, :, None, None] * H.view(1, 1, m_local, m_local)
    G0_blocks = G0_blocks + beta_cc[:, :, None, None] * (q[:, :, :, None] * q[:, :, None, :])
    if bool((sigma_g > 0).detach().cpu().item()):
        R = _projected_gradient_noise_gram(delta, lengthscale, gradient_noise_model)
        diag_idx = torch.arange(m_local, device=device)
        G0_blocks[diag_idx, diag_idx] = G0_blocks[diag_idx, diag_idx] + sigma_g * R
    G0 = G0_blocks.permute(0, 2, 1, 3).reshape(m_local * m_local, m_local * m_local).contiguous()

    rhs = torch.cat([k_fc[:, None], Q.T], dim=1)
    sol = torch.cholesky_solve(rhs, Lff)
    beta_star = sol[:, 0]
    QKinv = sol[:, 1:]

    sigma2 = y_target_var - torch.dot(k_fc, beta_star)
    k_delta = bar_k - Q @ beta_star
    K_delta = G0 - Q @ QKinv
    K_delta = 0.5 * (K_delta + K_delta.T)
    L_delta = cholesky_with_jitter(K_delta)
    w_star = torch.cholesky_solve(k_delta.unsqueeze(-1), L_delta).squeeze(-1)

    var = torch.clamp(sigma2 - torch.dot(w_star, k_delta), min=torch.finfo(dtype).eps)
    qtw = Q.T @ w_star
    f_weights_local = beta_star - torch.cholesky_solve(qtw.unsqueeze(-1), Lff).squeeze(-1)
    q_obs = gc @ delta
    mean = torch.dot(f_weights_local, yc) + torch.dot(w_star, q_obs.reshape(-1))
    return mean, var

def _projected_gradient_noise_gram(delta: torch.Tensor, lengthscale: torch.Tensor, gradient_noise_model: str) -> torch.Tensor:
    if gradient_noise_model == "iid":
        R = delta.T @ delta
    elif gradient_noise_model == "scaled":
        ell2 = lengthscale * lengthscale
        if ell2.numel() == 1:
            delta_noise = delta * ell2.reshape(1, 1)
        else:
            delta_noise = delta * ell2.view(-1, 1)
        R = delta.T @ delta_noise
    else:
        raise ValueError("gradient_noise_model must be either 'iid' or 'scaled'.")
    return 0.5 * (R + R.T)

def _func_covariance(X1: torch.Tensor, X2: torch.Tensor, lengthscale: torch.Tensor, outputscale: torch.Tensor, kernel_name: str) -> torch.Tensor:
    if lengthscale.numel() == 1:
        delta_scaled = (X1[:, None, :] - X2[None, :, :]) / lengthscale.reshape(1, 1, 1)
    else:
        delta_scaled = (X1[:, None, :] - X2[None, :, :]) / lengthscale.view(1, 1, -1)

    if kernel_name == "rbf":
        r = (delta_scaled * delta_scaled).sum(dim=-1)
        return outputscale * torch.exp(-0.5 * r)
    if kernel_name == "matern52":
        r2 = (delta_scaled * delta_scaled).sum(dim=-1)
        r = torch.sqrt(torch.clamp(r2, min=1e-12))
        a = math.sqrt(5.0) * r
        return outputscale * (1.0 + a + (a * a) / 3.0) * torch.exp(-a)
    raise ValueError(f"Unknown kernel name: {kernel_name}")

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
