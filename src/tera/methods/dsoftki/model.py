import logging
from typing import Callable as callable_type

import gpytorch
import gpytorch.constraints
import linear_operator
import numpy as np
import torch
import torch.nn as nn
from gpytorch.kernels import ScaleKernel as scale_kernel_type
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError as not_psd_error_type
from torch.utils.data import DataLoader as data_loader_type, TensorDataset as tensor_dataset_type

from tera.methods.common.linear_solver.cg import linear_cg
from tera.methods.dsoftki.mll import hutchinson_pseudo_loss

logger = logging.getLogger(__name__)


class vector_to_sequence(nn.Module):
    def __init__(self, input_dim, sequence_length, embedding_dim):

        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, sequence_length * embedding_dim),
            nn.SiLU(),
        )

    def forward(self, x):

        b = x.size(0)
        out = self.proj(x)
        out = out.view(b, self.sequence_length, self.embedding_dim)
        return out


class dsoftki(torch.nn.Module):
    def __init__(
        self,
        kernel: callable_type,
        interp_points: torch.Tensor,
        dim: int,
        noise=1e-4,
        deriv_noise=1e-2,
        learn_noise=False,
        use_scale=False,
        learn_t=True,
        min_t=5e-5,
        device="cpu",
        fit_device="cpu",
        dtype=torch.float32,
        solver="solve",
        max_cg_iter=50,
        cg_tolerance=0.1,
        mll_approx="hutchinson",
        fit_chunk_size=1024,
        use_qr=False,
        grad_only=False,
        embed_dim=-1,
        hidden_dim=64,
        use_dot=False,
        per_interp_t=True,
    ) -> None:

        methods = ["solve", "cholesky", "cg"]
        if solver not in methods:
            raise ValueError(f"Method {solver} should be in {methods} ...")

        assert (
            isinstance(max_cg_iter, int)
            and isinstance(fit_chunk_size, int)
            and isinstance(embed_dim, int)
        ), f"{type(max_cg_iter)}{type(fit_chunk_size)}{type(embed_dim)}"
        assert (
            isinstance(noise, float)
            and isinstance(deriv_noise, float)
            and isinstance(cg_tolerance, float)
        ), f"{type(noise)}{type(deriv_noise)}{type(cg_tolerance)}"
        assert (
            isinstance(use_qr, bool)
            and isinstance(grad_only, bool)
            and isinstance(use_dot, bool)
            and isinstance(per_interp_t, bool)
        )

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += ["cuda"]
            for i in range(torch.cuda.device_count()):
                devices += [f"cuda:{i}"]
        if device not in devices:
            raise ValueError(f"Device {device} should be in {devices} ...")

        super(dsoftki, self).__init__()

        self.device = device
        self.fit_device = fit_device
        self.dtype = dtype

        self.solve_method = solver
        self.mll_approx = mll_approx

        self.use_qr = use_qr
        self.fit_chunk_size = fit_chunk_size

        self.fit_device = fit_device

        self.noise_constraint = gpytorch.constraints.Positive()
        noise = torch.tensor([noise], dtype=self.dtype, device=self.device)
        noise = self.noise_constraint.inverse_transform(noise)
        if learn_noise:
            self.register_parameter("raw_noise", torch.nn.Parameter(noise))
        else:
            self.raw_noise = noise

        self.deriv_noise_constraint = gpytorch.constraints.Positive()
        deriv_noise = torch.tensor([deriv_noise], dtype=self.dtype, device=self.device)
        deriv_noise = self.deriv_noise_constraint.inverse_transform(deriv_noise)
        if learn_noise:
            self.register_parameter("raw_deriv_noise", torch.nn.Parameter(deriv_noise))
        else:
            self.raw_deriv_noise = deriv_noise

        self.use_scale = use_scale
        if use_scale:
            self.kernel = scale_kernel_type(kernel).to(self.device)
            if hasattr(kernel, "has_lengthscale") and kernel.has_lengthscale:
                self.kernel.base_kernel.register_constraint(
                    "raw_lengthscale", gpytorch.constraints.Interval(0.1, 5.0)
                )
        else:
            self.kernel = kernel.to(self.device)

        m = len(interp_points)
        self.embed_dim = embed_dim
        self.use_dot = use_dot
        if embed_dim != -1:
            self.min_t = min_t
            self.t_constraint = gpytorch.constraints.GreaterThan(min_t)
            t = torch.full((m,), 1.0, dtype=self.dtype, device=self.device)
            t = self.t_constraint.inverse_transform(t)
            self.register_parameter('raw_t', torch.nn.Parameter(t))
            self.learn_t = learn_t

            self.hidden_dim = hidden_dim

            self.embedding_query = nn.Sequential(
                nn.Linear(in_features=dim, out_features=self.hidden_dim),
                nn.Tanh(),
                nn.Linear(in_features=self.hidden_dim, out_features=embed_dim),
                nn.Tanh(),
            ).to(device=self.device)

            if use_dot:
                self.embedding_key = nn.Sequential(
                    nn.Linear(in_features=embed_dim, out_features=self.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(in_features=self.hidden_dim, out_features=embed_dim),
                    nn.Tanh(),
                ).to(device=self.device)
            else:
                self.embedding_key = None

            self.register_parameter("interp_points", torch.nn.Parameter(interp_points))

            self._interp = self._embed_interp
        else:
            self.embedding_query = None
            self.register_parameter("interp_points", torch.nn.Parameter(interp_points))
            d = interp_points.shape[-1]
            self.t_constraint = gpytorch.constraints.GreaterThan(5e-5)
            t = 1
            self.learn_t = learn_t
            if per_interp_t:
                t = torch.full(
                    (
                        m,
                        d,
                    ),
                    t,
                    dtype=self.dtype,
                    device=self.device,
                )
            else:
                t = torch.full((d,), t, dtype=self.dtype, device=self.device)
            t = self.t_constraint.inverse_transform(t)
            if learn_t:
                self.register_parameter('raw_t', torch.nn.Parameter(t))
            else:
                self.raw_t = t
            self._interp = self._basic_interp

        self.u_zz = torch.zeros((m, m), dtype=self.dtype, device=self.device)
        self.k_zz_alpha = torch.zeros(m, dtype=self.dtype, device=self.device)
        if True:
            self.alpha = torch.zeros((m, 1), dtype=self.dtype, device=fit_device)
        else:
            self.alpha = torch.zeros((m, 1), dtype=self.dtype, device=self.device)

        self.store_k_hat_xz = True
        self.store_k_hat_xz_val = True

        self.grad_only = grad_only

        self.recompute_fit_artifact = False
        self.fit_buffer = None
        self.fit_b = None
        self.q_matrix = None
        self.r_matrix = None

        self.max_cg_iter = max_cg_iter
        self.cg_tol = cg_tolerance
        self.x0 = None

    @property
    def noise(self):
        return self.noise_constraint.transform(self.raw_noise)

    @property
    def deriv_noise(self):
        return self.deriv_noise_constraint.transform(self.raw_deriv_noise)

    @property
    def t(self):
        return self.t_constraint.transform(self.raw_t)

    def get_lengthscale(self) -> float:
        if self.use_scale:
            kernel = self.kernel.base_kernel
            if hasattr(kernel, "has_lengthscale") and kernel.has_lengthscale:
                return self.kernel.base_kernel.lengthscale.cpu()
            else:
                return 1.0
        else:
            kernel = self.kernel
            if hasattr(kernel, "has_lengthscale") and kernel.has_lengthscale:
                return self.kernel.lengthscale.cpu()
            else:
                return 1.0

    def get_outputscale(self) -> float:
        if self.use_scale:
            return self.kernel.outputscale.cpu()
        else:
            return 1.0

    def _mk_cov(self, z: torch.Tensor) -> torch.Tensor:
        return self.kernel(z, z).evaluate()

    def _basic_interp(self, x: torch.Tensor, val=True, grad=True) -> torch.Tensor:

        assert self.interp_points.dtype == torch.float32
        z = self.interp_points
        b = x.shape[0]
        m, d = z.shape

        x_expanded = x.unsqueeze(1).expand(-1, z.shape[0], -1)

        diff = x_expanded / self.t - z

        distances = torch.linalg.vector_norm(diff, ord=2, dim=-1)

        neg_distances = -distances

        w_xz = torch.softmax(neg_distances, dim=-1)

        if val and not grad:
            return w_xz

        dist_deriv = (diff / (distances + 1e-6).unsqueeze(-1)) / self.t

        acc = []
        for j in range(len(z)):
            w_ij = w_xz[:, j]

            delta_jk = torch.zeros(len(z), dtype=self.dtype, device=self.device)
            delta_jk[j] = 1.0

            w_ik = w_xz

            w_xz_deriv = (w_ij.unsqueeze(-1) * (delta_jk.unsqueeze(0) - w_ik)).unsqueeze(-1)

            w_xz_deriv = -w_xz_deriv * dist_deriv

            w_xz_deriv = torch.sum(w_xz_deriv, dim=1)

            acc += [w_xz_deriv.reshape(-1)]

        w_xz_deriv = torch.stack(acc, dim=1)
        assert w_xz_deriv.shape[0] == b * d and w_xz_deriv.shape[1] == m

        if self.grad_only:
            return w_xz_deriv
        else:
            assert w_xz.shape[0] == b and w_xz.shape[1] == m
            w_xz_full = torch.cat([w_xz, w_xz_deriv], dim=0)
            assert w_xz_full.shape[0] == b * (d + 1) and w_xz_full.shape[1] == m
            return w_xz_full

    def _embed_interp(self, x: torch.Tensor, val=True, grad=True):

        b, d = x.shape
        z = self.interp_points
        m, d_p = z.shape

        x = x.clone().detach().requires_grad_(True)

        if self.embed_dim == -1:

            def wrapped(x_in):
                f_x_ = x_in
                diffs_ = f_x_[:, None, :] / self.t - z[None, :, :]
                norms_ = torch.linalg.vector_norm(diffs_, ord=2, dim=-1)
                w_xz = torch.softmax(-norms_, dim=-1)
                return w_xz
        elif self.use_dot:

            def wrapped(x_in):
                f_x_ = self.embedding_query(x_in)
                z_ = self.embedding_key(z)
                dot_products = torch.matmul(f_x_, z_.T)
                w_xz = torch.softmax(-dot_products / self.t, dim=-1)
                return w_xz
        else:

            def wrapped(x_in):
                f_x_ = self.embedding_query(x_in)
                diffs_ = f_x_[:, None, :] - z[None, :, :]
                norms_ = torch.linalg.vector_norm(diffs_, ord=2, dim=-1)
                w_xz = torch.softmax(-norms_ / self.t, dim=-1)
                return w_xz

        w = wrapped(x)

        if val and not grad:
            return w

        _, vjp_fn = torch.func.vjp(wrapped, x)
        eye = torch.eye(m, device=x.device).unsqueeze(0).expand(b, -1, -1)

        dw = torch.stack([vjp_fn(eye[:, k])[0] for k in range(m)], dim=1)
        dw = dw.permute(0, 2, 1).reshape(-1, m)

        if not val and grad:
            return dw
        else:
            w_xz_full = torch.cat([w, dw], dim=0)
            return w_xz_full

    def _solve_system(
        self,
        kxx: linear_operator.operators.LinearOperator,
        full_rhs: torch.Tensor,
        forwards_matmul: callable_type = None,
        return_pinv: bool = False,
    ) -> torch.Tensor:
        use_pinv = False
        with torch.no_grad():
            try:
                if self.solve_method == "solve":
                    solve = torch.linalg.solve(kxx, full_rhs)
                elif self.solve_method == "cholesky":
                    l = torch.linalg.cholesky(kxx)
                    solve = torch.cholesky_solve(full_rhs, l)
                elif self.solve_method == "cg":
                    solve = linear_cg(
                        forwards_matmul,
                        full_rhs,
                        max_iter=self.max_cg_iter,
                        tolerance=self.cg_tol,
                    )
                else:
                    raise ValueError(f"Unknown method: {self.solve_method}")
            except (torch.linalg.LinAlgError, not_psd_error_type) as e:
                logger.warning("Linear solve failed; using pseudoinverse: %s", e)
                solve = torch.linalg.pinv(kxx.evaluate()) @ full_rhs
                use_pinv = True

        solve = torch.nan_to_num(solve)
        return (solve, use_pinv) if return_pinv else solve

    def mll(self, inputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        k_zz = self._mk_cov(self.interp_points)
        w_xz = self._interp(inputs)

        if torch.isnan(w_xz).any():
            raise ValueError("NaN detected in W_xz ...")

        try:
            l = psd_safe_cholesky(k_zz)
            if torch.isnan(l).any():
                logger.warning("Non-finite Cholesky factor; retrying in float64")
                l = psd_safe_cholesky(k_zz.double()).float()
                if torch.isnan(l).any():
                    raise torch.linalg.LinAlgError("Non-finite Cholesky factor")
        except (torch.linalg.LinAlgError, not_psd_error_type):
            logger.warning("Reinitializing interpolation points after Cholesky failure")
            m = k_zz.shape[0]
            d = inputs.shape[1]
            self.interp_points.data = torch.randn((m, d), device=self.device, dtype=self.dtype)
            k_zz = self._mk_cov(self.interp_points)
            w_xz = self._interp(inputs)
            l = psd_safe_cholesky(k_zz)

        def exact():

            mean = torch.zeros(len(inputs) * (inputs.shape[1] + 1), dtype=self.dtype, device=self.device)

            lk = (w_xz @ l).to(device=self.device)
            noise = torch.cat(
                [
                    torch.ones(len(inputs), dtype=self.dtype, device=self.device) * self.noise,
                    torch.ones(lk.shape[0] - len(inputs), dtype=self.dtype, device=self.device)
                    * self.deriv_noise,
                ]
            )

            normal_dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                mean, lk, noise, validate_args=None
            )

            ys = y[:, 0]
            d_ys = y[:, 1:]
            y_p = torch.cat([ys, d_ys.reshape(-1)])
            return normal_dist.log_prob(y_p)

        def hutch():

            lk = (w_xz @ l).to(device=self.device)

            hutchinson_mll = hutchinson_pseudo_loss(self, dim=lk.shape[0], num_trace_samples=10)
            if self.grad_only:
                mean = torch.zeros(len(inputs) * (inputs.shape[-1]), dtype=self.dtype, device=self.device)
                noise = (
                    torch.ones(lk.shape[0], dtype=self.dtype, device=self.device) * self.deriv_noise
                )
                y_p = y.reshape(-1)
            else:
                mean = torch.zeros(len(inputs) * (1 + inputs.shape[-1]), dtype=self.dtype, device=self.device)
                noise = torch.cat(
                    [
                        torch.ones(len(inputs), dtype=self.dtype, device=self.device) * self.noise,
                        torch.ones(lk.shape[0] - len(inputs), dtype=self.dtype, device=self.device)
                        * self.deriv_noise,
                    ]
                )
                ys = y[:, 0]
                d_ys = y[:, 1:]
                y_p = torch.cat([ys, d_ys.reshape(-1)])
            return hutchinson_mll(mean, lk, noise, y_p)

        if self.mll_approx == "exact":
            return exact()
        elif self.mll_approx == "hutchinson_fallback":
            try:
                return exact()
            except (torch.linalg.LinAlgError, not_psd_error_type) as e:
                logger.warning("Exact likelihood failed; using Hutchinson estimate: %s", e)
                return hutch()
        elif self.mll_approx == "hutchinson":
            return hutch()
        else:
            raise ValueError(f"Unknown MLL approximation method: {self.mll_approx}")

    def _qr_solve_fit(
        self, m: int, n: int, d: int, inputs: torch.Tensor, y: torch.Tensor, k_zz: torch.Tensor
    ) -> bool:
        device = self.fit_device

        d_p = d if self.grad_only else d + 1
        self.d_p = d_p

        if self.fit_buffer is None:
            self.fit_buffer = torch.zeros((n * d_p + m, m), dtype=self.dtype, device=device)
            self.fit_b = torch.zeros(n * d_p, dtype=self.dtype, device=device)
            self.fit_y = torch.zeros(n * d_p, dtype=self.dtype, device=device)
            if self.store_k_hat_xz:
                self.hat_k_xz = torch.zeros((n * d_p, m), dtype=self.dtype, device=device)
            if self.store_k_hat_xz_val:
                self.hat_k_xz_val = torch.zeros((n, m), dtype=self.dtype, device=device)

        with torch.no_grad():
            fit_chunk_size = self.fit_chunk_size
            batches = int(np.floor(n / fit_chunk_size)) + int(n % fit_chunk_size > 0)
            for i in range(batches):
                start_i = i * fit_chunk_size
                end_i = min((i + 1) * fit_chunk_size, n)
                cs = end_i - start_i

                start = start_i * d_p
                end = end_i * d_p

                x_batch = inputs[start_i:end_i, :]
                w_xz = self._interp(x_batch)
                self.fit_buffer[start:end, :] = (w_xz @ k_zz).to(device)
                if self.store_k_hat_xz:
                    self.hat_k_xz[start:end, :] = self.fit_buffer[start:end, :]
                if self.store_k_hat_xz_val:
                    self.hat_k_xz_val[start_i:end_i, :] = self.fit_buffer[start : start + cs, :]

                self.fit_buffer[start : start + cs, :] /= torch.sqrt(self.noise).to(device)
                self.fit_buffer[start + cs : end, :] /= torch.sqrt(self.deriv_noise).to(device)

                if self.grad_only:
                    d_ys = y[start_i:end_i, 1:]
                    self.fit_y[start:end] = d_ys.reshape(-1).to(device) / torch.sqrt(
                        self.deriv_noise
                    ).to(device)
                else:
                    ys = y[start_i:end_i, 0]
                    d_ys = y[start_i:end_i, 1:]
                    self.fit_y[start:end] = torch.cat(
                        [
                            ys.to(device) / torch.sqrt(self.noise).to(device),
                            d_ys.reshape(-1).to(device) / torch.sqrt(self.deriv_noise).to(device),
                        ]
                    )

        with torch.no_grad():
            psd_safe_cholesky(k_zz, out=self.u_zz, upper=True, max_tries=10)
            if torch.isnan(self.u_zz).any():
                logger.warning("Non-finite Cholesky factor; retrying in float64")
                tmp = psd_safe_cholesky(k_zz.double(), upper=True, max_tries=10)
                self.u_zz[:, :] = tmp.to(self.device)
                if torch.isnan(self.u_zz).any():
                    logger.warning("Reinitializing interpolation points after Cholesky failure")
                    self.interp_points.data = torch.randn(
                        (m, d), device=self.device, dtype=self.dtype
                    )
            self.fit_buffer[n * d_p :, :] = self.u_zz.to(device)

            if self.q_matrix is None:
                self.q_matrix = torch.zeros((n * d_p + m, m), dtype=self.dtype, device=device)
                self.r_matrix = torch.zeros((m, m), dtype=self.dtype, device=device)

            torch.linalg.qr(self.fit_buffer, out=(self.q_matrix, self.r_matrix))

            torch.linalg.solve_triangular(
                self.r_matrix,
                (self.q_matrix.T[:, 0 : n * d_p] @ self.fit_y).unsqueeze(1),
                upper=True,
                out=self.alpha,
            ).squeeze(1)

            torch.matmul(k_zz, self.alpha.to(self.device).squeeze(-1), out=self.k_zz_alpha)

        self.recompute_fit_artifact = True

        return False

    def fit(self, inputs: torch.Tensor, y: torch.Tensor) -> bool:

        n = len(inputs)
        m = len(self.interp_points)
        d = inputs.shape[-1]
        inputs = inputs.to(self.device, dtype=self.dtype)
        y = y.to(self.device, dtype=self.dtype)

        k_zz = self._mk_cov(self.interp_points)

        self.n = n
        self.k_zz = k_zz

        return self._qr_solve_fit(m, n, d, inputs, y, k_zz)

    def pred(self, x_star: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            w_star_z = self._interp(x_star)
            return torch.matmul(w_star_z, self.k_zz_alpha).squeeze(-1)

    def pred_cov(self, x_star: torch.Tensor) -> torch.Tensor:

        device = self.device

        with torch.no_grad():
            w_star_z = self._interp(x_star).to(device)
            q_star_x = w_star_z @ self.hat_k_xz.T.to(device)

            q_star_x /= torch.sqrt(self.deriv_noise.to(device))
            batches = int(np.floor(self.n / self.fit_chunk_size)) + int(
                self.n % self.fit_chunk_size > 0
            )
            for i in range(batches):
                start = i * self.fit_chunk_size * self.d_p
                if i == batches - 1:
                    end = self.n - i * self.fit_chunk_size
                else:
                    end = i * self.fit_chunk_size * self.d_p + self.fit_chunk_size
                q_star_x[start:end, :] *= torch.sqrt(self.deriv_noise.to(device)) / torch.sqrt(
                    self.noise.to(device)
                )
            fit_b = q_star_x.T

            beta = torch.linalg.solve_triangular(
                self.r_matrix.to(device),
                (self.q_matrix.T[:, 0 : self.n * self.d_p].to(device) @ fit_b),
                upper=True,
            )
            q_star_star = w_star_z @ self.k_zz.to(device) @ w_star_z.T

            res = (
                q_star_star
                - q_star_x @ q_star_x.T
                + (q_star_x @ self.fit_buffer[: self.n * self.d_p, :].to(device)) @ beta
            )

            del w_star_z
            del q_star_x
            del fit_b
            del beta
            del q_star_star

            return torch.clamp(res, min=self.noise.to(device))

    def pred_cov_val(self, x_star: torch.Tensor) -> torch.Tensor:

        device = self.device
        with torch.no_grad():
            if self.recompute_fit_artifact:
                dataloader = data_loader_type(
                    tensor_dataset_type(self.hat_k_xz), batch_size=self.fit_chunk_size * self.d_p
                )
                self._tmp_hat_k_xz = torch.cat(
                    [data[0][0 : len(data[0]) // self.d_p, :] for data in dataloader]
                )
                dataloader = data_loader_type(
                    tensor_dataset_type(self.q_matrix), batch_size=self.fit_chunk_size * self.d_p
                )
                self._tmp_q = torch.cat(
                    [data[0][0 : len(data[0]) // self.d_p, :] for data in dataloader]
                )
                self.recompute_fit_artifact = False
                dataloader = data_loader_type(
                    tensor_dataset_type(self.fit_buffer[: self.n * self.d_p, :]),
                    batch_size=self.fit_chunk_size * self.d_p,
                )
                self._tmp_fit_buffer = torch.cat(
                    [data[0][0 : len(data[0]) // self.d_p, :] for data in dataloader]
                )
                self.recompute_fit_artifact = False

            w_star_z = self._interp(x_star, val=True, grad=False).to(device)

            q_star_x = w_star_z @ self._tmp_hat_k_xz.T.to(device)

            q_star_x /= torch.sqrt(self.noise.to(device))

            beta = torch.linalg.solve_triangular(
                self.r_matrix.to(device),
                (self._tmp_q.T[:, 0 : self.n].to(device) @ q_star_x.T),
                upper=True,
            )
            q_star_star = w_star_z @ self.k_zz.to(device) @ w_star_z.T

            res = (
                q_star_star
                - q_star_x @ q_star_x.T
                + (q_star_x @ self._tmp_fit_buffer.to(device)) @ beta
            )

            del w_star_z
            del q_star_x
            del beta
            del q_star_star

            return torch.clamp(res, min=self.noise.to(device))

    def __del__(self):
        if hasattr(self, "fit_buffer"):
            del self.fit_buffer
        if hasattr(self, "fit_b"):
            del self.fit_b
        if hasattr(self, "fit_y"):
            del self.fit_y
        if hasattr(self, 'hat_k_xz'):
            del self.hat_k_xz
        if hasattr(self, 'hat_k_xz_val'):
            del self.hat_k_xz_val
        if hasattr(self, '_tmp_hat_k_xz'):
            del self._tmp_hat_k_xz
        if hasattr(self, '_tmp_q'):
            del self._tmp_q
        if hasattr(self, "_tmp_fit_buffer"):
            del self._tmp_fit_buffer
        if hasattr(self, 'raw_t'):
            del self.raw_t
        if hasattr(self, "raw_noise"):
            del self.raw_noise
        if hasattr(self, "raw_deriv_noise"):
            del self.raw_deriv_noise
        if hasattr(self, "interp_points"):
            del self.interp_points
        if hasattr(self, "kernel"):
            del self.kernel
