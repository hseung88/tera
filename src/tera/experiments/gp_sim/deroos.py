from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict as dict_type, Tuple as tuple_type

import torch

from tera.experiments.gp_sim.utils import cholesky_with_jitter

_l_dense_cache: dict_type[tuple_type[int, str, torch.dtype], torch.Tensor] = {}
_c_colidx_cache: dict_type[tuple_type[int, str], torch.Tensor] = {}


def _cache_key_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{device.index if device.index is not None else 0}"
    return "cpu"


def inv_lengthscale_sq(
    lengthscale: torch.Tensor, d: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if lengthscale.numel() == 1:
        return (
            (1.0 / (lengthscale * lengthscale)).reshape(1).repeat(d).to(device=device, dtype=dtype)
        )
    return (1.0 / (lengthscale * lengthscale)).to(device=device, dtype=dtype)


def k_kp_kpp_from_r(
    r: torch.Tensor, kernel: str, outputscale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    os = torch.as_tensor(outputscale, device=r.device, dtype=r.dtype)
    if kernel == "rbf":
        k = os * torch.exp(-0.5 * r)
        kp = -0.5 * k
        kpp = 0.25 * k
        return k, kp, kpp
    if kernel == "matern52":
        a = torch.sqrt(torch.clamp(r, min=1e-12)) * math.sqrt(5.0)
        ea = torch.exp(-a)
        k = os * (1.0 + a + (a * a) / 3.0) * ea
        kp = -(5.0 / 6.0) * os * (1.0 + a) * ea
        kpp = (25.0 / 12.0) * os * ea
        return k, kp, kpp
    raise ValueError(f"Unknown kernel: {kernel}")


def function_covariance(inputs1, inputs2, lengthscale, outputscale, kernel):
    center = 0.5 * (inputs1.mean(dim=0, keepdim=True) + inputs2.mean(dim=0, keepdim=True))

    if lengthscale.numel() == 1:
        x1s = (inputs1 - center) / lengthscale.reshape(1)
        x2s = (inputs2 - center) / lengthscale.reshape(1)
    else:
        x1s = (inputs1 - center) / lengthscale.view(1, -1)
        x2s = (inputs2 - center) / lengthscale.view(1, -1)

    x1_norm = (x1s * x1s).sum(dim=-1, keepdim=True)
    x2_norm = (x2s * x2s).sum(dim=-1).unsqueeze(0)
    r = x1_norm + x2_norm - 2.0 * (x1s @ x2s.T)
    r = torch.clamp(r, min=0.0)

    k, _, _ = k_kp_kpp_from_r(r, kernel, outputscale)
    return k


def value_grad_cross_blocks(
    xc: torch.Tensor,
    x: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return K(f_C,f_*), K(g_C,f_*), and K(f_*,f_*)."""
    if x.ndim == 1:
        x = x.view(1, -1)
    d = xc.shape[1]
    w = inv_lengthscale_sq(lengthscale, d, xc.device, xc.dtype)
    delta = xc - x
    v = delta * w.view(1, d)
    r = (delta * v).sum(dim=-1)
    k, kp, _ = k_kp_kpp_from_r(r, kernel, outputscale)
    k0 = (-2.0) * kp
    k_f = k.contiguous()
    k_g = (-(k0[:, None] * v)).reshape(xc.shape[0] * d).contiguous()
    k_xx = torch.as_tensor(outputscale, device=xc.device, dtype=xc.dtype)
    return k_f, k_g, k_xx


def value_grad_cross_blocks_many(
    xc: torch.Tensor,
    xq: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return K(f_C,f_Q), K(g_C,f_Q), and K(f_Q,f_Q) diagonal.

    The gradient rows are point-major, matching gc.reshape(M * d).
    """
    if xq.ndim == 1:
        xq = xq.view(1, -1)
    m, d = xc.shape
    q_matrix = xq.shape[0]
    w = inv_lengthscale_sq(lengthscale, d, xc.device, xc.dtype)
    delta = xc[:, None, :] - xq[None, :, :]
    v = delta * w.view(1, 1, d)
    r = (delta * v).sum(dim=-1)
    k, kp, _ = k_kp_kpp_from_r(r, kernel, outputscale)
    k0 = (-2.0) * kp
    k_f = k.contiguous()
    k_g = (-(k0[:, :, None] * v)).permute(0, 2, 1).reshape(m * d, q_matrix).contiguous()
    k_xx = torch.full((q_matrix,), float(outputscale), device=xc.device, dtype=xc.dtype)
    return k_f, k_g, k_xx


def full_value_grad_covariance(
    inputs: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
    *,
    sigma_f: float = 0.0,
    sigma_g: float = 0.0,
) -> torch.Tensor:
    n, d = inputs.shape
    w = inv_lengthscale_sq(lengthscale, d, inputs.device, inputs.dtype)
    delta = inputs[:, None, :] - inputs[None, :, :]
    v = delta * w.view(1, 1, d)
    r = (delta * v).sum(dim=-1)
    k, kp, kpp = k_kp_kpp_from_r(r, kernel, outputscale)
    k0 = (-2.0) * kp
    c2 = (-4.0) * kpp
    diagw = torch.diag(w)
    outer = v[..., :, None] * v[..., None, :]
    kgg = k0[..., None, None] * diagw.view(1, 1, d, d) + c2[..., None, None] * outer
    kfg = k0[..., None] * v
    kgf = -kfg
    blk = torch.zeros((n, n, d + 1, d + 1), device=inputs.device, dtype=inputs.dtype)
    blk[:, :, 0, 0] = k
    blk[:, :, 0, 1:] = kfg
    blk[:, :, 1:, 0] = kgf
    blk[:, :, 1:, 1:] = kgg
    covariance = blk.permute(0, 2, 1, 3).reshape(n * (d + 1), n * (d + 1)).contiguous()
    covariance = 0.5 * (covariance + covariance.T)
    if sigma_f > 0.0:
        idx_f = torch.arange(0, n * (d + 1), step=d + 1, device=inputs.device, dtype=torch.long)
        covariance[idx_f, idx_f] += sigma_f * sigma_f
    if sigma_g > 0.0:
        idx_f = torch.arange(0, n * (d + 1), step=d + 1, device=inputs.device, dtype=torch.long)
        ar = torch.arange(1, d + 1, device=inputs.device, dtype=torch.long)
        idx_g = (idx_f[:, None] + ar[None, :]).reshape(-1)
        covariance[idx_g, idx_g] += sigma_g * sigma_g
    return covariance


def get_l_dense(m: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (m, _cache_key_device(device), dtype)
    cached = _l_dense_cache.get(key)
    if cached is not None:
        return cached
    dim = m * m
    l = -torch.eye(dim, device=device, dtype=dtype)
    a = torch.arange(m, device=device, dtype=torch.long)
    row = (a * m + a).repeat_interleave(m)
    col = a.repeat_interleave(m) * m + torch.arange(m, device=device).repeat(m)
    l[row, col] += 1.0
    _l_dense_cache[key] = l
    return l


def get_c_colidx(m: int, device: torch.device) -> torch.Tensor:
    key = (m, _cache_key_device(device))
    cached = _c_colidx_cache.get(key)
    if cached is not None:
        return cached
    dim = m * m
    col = torch.arange(dim, device=device, dtype=torch.long).reshape(m, m).t().reshape(-1)
    _c_colidx_cache[key] = col
    return col


def cholesky_ex_stacked_with_jitter(
    covariance: torch.Tensor, jitter0: float = 1e-8, jitter_max: float = 1e-3
):
    b, m, _ = covariance.shape
    i = torch.eye(m, device=covariance.device, dtype=covariance.dtype).expand(b, m, m)
    jitter = 0.0
    last_info = None
    while True:
        kj = covariance if jitter == 0.0 else (covariance + jitter * i)
        l, info = torch.linalg.cholesky_ex(kj)
        if bool((info == 0).all().item()):
            return l, float(jitter)
        last_info = info
        jitter = jitter0 if jitter == 0.0 else jitter * 10.0
        if jitter > jitter_max:
            raise RuntimeError(
                f"stacked cholesky_ex failed up to jitter={jitter_max}. last info={last_info}"
            )


def ldl_factor_ex_stacked_with_jitter(
    t: torch.Tensor, jitter0: float = 1e-8, jitter_max: float = 1e-3
):
    b, dim, _ = t.shape
    i = torch.eye(dim, device=t.device, dtype=t.dtype).expand(b, dim, dim)
    jitter = 0.0
    last_info = None
    while True:
        tj = t if jitter == 0.0 else (t + jitter * i)
        ld, piv, info = torch.linalg.ldl_factor_ex(tj, hermitian=True, check_errors=False)
        if bool((info == 0).all().item()):
            return ld, piv, float(jitter)
        last_info = info
        jitter = jitter0 if jitter == 0.0 else jitter * 10.0
        if jitter > jitter_max:
            raise RuntimeError(
                f"stacked ldl_factor_ex failed up to jitter={jitter_max}. last info={last_info}"
            )


def apply_pair_laplacian_stacked(v: torch.Tensor, m: int) -> torch.Tensor:
    b, dim, r_matrix = v.shape
    vmat = v.reshape(b, m, m, r_matrix)
    row_sums = vmat.sum(dim=2)
    av = torch.zeros_like(vmat)
    diag = torch.arange(m, device=v.device)
    av[:, diag, diag, :] = row_sums
    return (av - vmat).reshape(b, dim, r_matrix)


def apply_pair_laplacian_transpose_stacked(v: torch.Tensor, m: int) -> torch.Tensor:
    b, dim, r_matrix = v.shape
    vmat = v.reshape(b, m, m, r_matrix)
    diag = torch.arange(m, device=v.device)
    diag_vals = vmat[:, diag, diag, :]
    atv = diag_vals.unsqueeze(2).expand(b, m, m, r_matrix)
    return (atv - vmat).reshape(b, dim, r_matrix)


@dataclass(slots=True)
class deroos_gradient_state:
    xc: torch.Tensor
    w: torch.Tensor
    m: int
    d: int
    lk: torch.Tensor
    kinv: torch.Tensor
    ld: torch.Tensor
    piv: torch.Tensor


def prepare_deroos_gradient_inverse(
    xc: torch.Tensor,
    w: torch.Tensor,
    k0_local: torch.Tensor,
    kpp4_local: torch.Tensor,
    g_local: torch.Tensor,
) -> deroos_gradient_state:
    device, dtype = xc.device, xc.dtype
    b, m, d = xc.shape
    dim = m * m

    k0 = 0.5 * (k0_local + k0_local.transpose(-1, -2))
    lk, _ = cholesky_ex_stacked_with_jitter(k0)
    i = torch.eye(m, device=device, dtype=dtype).expand(b, m, m)
    kinv = torch.cholesky_solve(i, lk)
    kinv = 0.5 * (kinv + kinv.transpose(-1, -2))

    g = 0.5 * (g_local + g_local.transpose(-1, -2))
    kpp4 = 0.5 * (kpp4_local + kpp4_local.transpose(-1, -2))
    inv = 1.0 / torch.clamp(kpp4, min=kpp4.new_tensor(1e-14))
    vals = inv.reshape(b, dim)

    row = torch.arange(dim, device=device)
    col = get_c_colidx(m, device)
    cinv = torch.zeros((b, dim, dim), device=device, dtype=dtype)
    cinv[:, row, col] = vals
    cinv = 0.5 * (cinv + cinv.transpose(-1, -2))

    l = get_l_dense(m, device, dtype).contiguous()
    a = torch.einsum("bij,bkl->bikjl", kinv, g).reshape(b, dim, dim).contiguous()
    al = torch.matmul(a, l)
    ut = torch.matmul(l.t().unsqueeze(0), al)
    t = cinv + ut
    t = 0.5 * (t + t.transpose(-1, -2))
    ld, piv, _ = ldl_factor_ex_stacked_with_jitter(t)
    return deroos_gradient_state(xc=xc, w=w, m=m, d=d, lk=lk, kinv=kinv, ld=ld, piv=piv)


def apply_deroos_gradient_inverse(state: deroos_gradient_state, v: torch.Tensor) -> torch.Tensor:
    xc = state.xc
    w = state.w
    m = state.m
    d = state.d
    bsz, md, r_matrix = v.shape
    if md != m * d:
        raise ValueError(f"Expected V second dimension {m * d}, got {md}")

    dy = v.reshape(bsz, m, d, r_matrix).permute(0, 2, 1, 3).contiguous()
    dy = dy / w.view(1, d, 1, 1)
    dy2 = dy.permute(0, 2, 1, 3).reshape(bsz, m, d * r_matrix).contiguous()
    sol = torch.cholesky_solve(dy2, state.lk)
    ymat = sol.reshape(bsz, m, d, r_matrix).permute(0, 2, 1, 3).contiguous()

    wy = ymat * w.view(1, d, 1, 1)
    p = torch.einsum("bid,bdjr->bijr", xc, wy).contiguous()
    pvec = p.permute(0, 2, 1, 3).reshape(bsz, m * m, r_matrix).contiguous()
    rhs = apply_pair_laplacian_transpose_stacked(pvec, m)
    s = torch.linalg.ldl_solve(state.ld, state.piv, rhs, hermitian=True)
    tvec = apply_pair_laplacian_stacked(s, m)
    tmat = tvec.reshape(bsz, m, m, r_matrix).permute(0, 2, 1, 3).contiguous()
    a = torch.einsum("bijr,bjk->bikr", tmat, state.kinv).contiguous()
    corr = torch.einsum("bdm,bmnr->bdnr", xc.permute(0, 2, 1), a).contiguous()
    z = ymat - corr
    return z.permute(0, 2, 1, 3).reshape(bsz, m * d, r_matrix).contiguous()


@dataclass(slots=True)
class value_grad_deroos_conditioner:
    xc: torch.Tensor
    yc: torch.Tensor
    gc_flat: torch.Tensor
    lengthscale: torch.Tensor
    outputscale: float
    kernel: str
    sigma_f: float
    w: torch.Tensor
    kff: torch.Tensor
    kfg: torch.Tensor
    kgf: torch.Tensor
    gradient_inverse: deroos_gradient_state
    dinv_kgf: torch.Tensor
    schur_chol: torch.Tensor


def prepare_value_grad_deroos_conditioner(
    xc: torch.Tensor,
    yc: torch.Tensor,
    gc: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
    sigma_f: float = 0.0,
    sigma_g: float = 0.0,
) -> value_grad_deroos_conditioner:
    if sigma_g != 0.0:
        raise ValueError("deRoos value-gradient conditioner currently assumes sigma_g=0.")
    m, d = xc.shape
    device, dtype = xc.device, xc.dtype
    w = inv_lengthscale_sq(lengthscale, d, device, dtype)

    kff = function_covariance(xc, xc, lengthscale, outputscale, kernel)
    kff = 0.5 * (kff + kff.T)
    if sigma_f > 0.0:
        kff = kff + (sigma_f * sigma_f) * torch.eye(m, device=device, dtype=dtype)

    delta_cc = xc[:, None, :] - xc[None, :, :]
    v_cc = delta_cc * w.view(1, 1, d)
    r_cc = (delta_cc * v_cc).sum(dim=-1)
    _, kp_cc, kpp_cc = k_kp_kpp_from_r(r_cc, kernel, outputscale)
    k0_cc = (-2.0) * kp_cc
    kpp4_cc = 4.0 * kpp_cc

    kfg4 = k0_cc[:, :, None] * v_cc
    kfg = kfg4.reshape(m, m * d).contiguous()
    kgf = (-kfg4).permute(0, 2, 1).reshape(m * d, m).contiguous()

    g_local = (xc * w.view(1, d)) @ xc.T
    g_local = 0.5 * (g_local + g_local.T)
    state = prepare_deroos_gradient_inverse(
        xc.unsqueeze(0).contiguous(),
        w,
        k0_cc.unsqueeze(0).contiguous(),
        kpp4_cc.unsqueeze(0).contiguous(),
        g_local.unsqueeze(0).contiguous(),
    )
    dinv_kgf = apply_deroos_gradient_inverse(state, kgf.unsqueeze(0)).squeeze(0)
    s = kff - kfg @ dinv_kgf
    s = 0.5 * (s + s.T)
    schur_chol = cholesky_with_jitter(s)
    return value_grad_deroos_conditioner(
        xc=xc,
        yc=yc.reshape(m).contiguous(),
        gc_flat=gc.reshape(m * d).contiguous(),
        lengthscale=lengthscale,
        outputscale=outputscale,
        kernel=kernel,
        sigma_f=sigma_f,
        w=w,
        kff=kff,
        kfg=kfg,
        kgf=kgf,
        gradient_inverse=state,
        dinv_kgf=dinv_kgf,
        schur_chol=schur_chol,
    )


def predict_one_value_grad_deroos(
    cond: value_grad_deroos_conditioner, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 1:
        x = x.view(1, -1)
    m, d = cond.xc.shape
    k_f, k_g, k_xx = value_grad_cross_blocks(
        cond.xc, x, cond.lengthscale, cond.outputscale, cond.kernel
    )
    kfi = k_f.view(m, 1)
    kgi0 = k_g.view(m * d, 1)
    dinv_kgi = apply_deroos_gradient_inverse(cond.gradient_inverse, kgi0.unsqueeze(0)).squeeze(0)
    rhs_f = kfi - cond.kfg @ dinv_kgi
    x_f = torch.cholesky_solve(rhs_f, cond.schur_chol).squeeze(-1)
    x_g = dinv_kgi.squeeze(-1) - cond.dinv_kgf @ x_f
    mean = torch.dot(x_f, cond.yc) + torch.dot(x_g, cond.gc_flat)
    var = k_xx - torch.dot(k_f, x_f) - torch.dot(k_g, x_g)
    var = torch.clamp(var, min=torch.finfo(x.dtype).eps)
    return mean, var


def predict_all_value_grad_deroos(
    cond: value_grad_deroos_conditioner,
    x_eval: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict scalar function marginals for all query points at once.

    The matrix K(g_C, f_Q) is sent through the deRoos representation of
    K(g_C, g_C)^{-1} as one multi-right-hand-side solve. There is no query
    chunking and no query loop. The remaining bottom block of the full
    inverse solve reuses the precomputed matrix K(g_C,g_C)^{-1}K(g_C,f_C).
    """
    if x_eval.ndim == 1:
        x_eval = x_eval.view(1, -1)
    kfi, kgi0, k_xx = value_grad_cross_blocks_many(
        cond.xc, x_eval, cond.lengthscale, cond.outputscale, cond.kernel
    )
    dinv_kgi = apply_deroos_gradient_inverse(cond.gradient_inverse, kgi0.unsqueeze(0)).squeeze(0)
    rhs_f = kfi - cond.kfg @ dinv_kgi
    x_f = torch.cholesky_solve(rhs_f, cond.schur_chol)
    x_g = dinv_kgi - cond.dinv_kgf @ x_f
    mean = (x_f * cond.yc[:, None]).sum(dim=0) + (x_g * cond.gc_flat[:, None]).sum(dim=0)
    var = k_xx - (kfi * x_f).sum(dim=0) - (kgi0 * x_g).sum(dim=0)
    var = torch.clamp(var, min=torch.finfo(x_eval.dtype).eps)
    return mean.contiguous(), var.contiguous()


def deroos_apply_u(pair_weights: torch.Tensor, inputs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Apply the de Roos low-rank map U to a vector indexed by ordered point pairs.

    If A is the n by n matrix formed from ``pair_weights``, this returns
    ``(L(A) @ X) diag(w)`` where L(A) has rows
    ``L(A)[i, j] = 1{i=j} sum_b A[i, b] - A[i, j]``.

    The identity used by the sampler is

        K_gg = K0 \\otimes diag(w) + U C U^T,

    with C pairing ordered pairs (i,j) and (j,i).  This function never
    materializes U.
    """
    n, d = inputs.shape
    if pair_weights.numel() != n * n:
        raise ValueError(f"Expected {n * n} pair weights, got {pair_weights.numel()}")
    pair_mat = pair_weights.reshape(1, n * n, 1)
    lap_pair = apply_pair_laplacian_stacked(pair_mat, n).reshape(n, n)
    return (lap_pair @ inputs) * w.view(1, d)


def deroos_apply_ut_ainv(
    g: torch.Tensor,
    inputs: torch.Tensor,
    w: torch.Tensor,
    k0_inv: torch.Tensor,
) -> torch.Tensor:
    """Apply U^T A^{-1} to a gradient vector without materializing U.

    Here A = K0 \\otimes diag(w).  ``G`` has shape n by d.
    """
    n, d = inputs.shape
    if g.shape != (n, d):
        raise ValueError(f"Expected G shape {(n, d)}, got {tuple(g.shape)}")

    y = k0_inv @ (g / w.view(1, d))

    p = y @ (inputs * w.view(1, d)).T
    return apply_pair_laplacian_transpose_stacked(p.reshape(1, n * n, 1), n).reshape(n * n)


@dataclass(slots=True)
class deroos_gradient_sampler_state:
    inputs: torch.Tensor
    w: torch.Tensor
    k0: torch.Tensor
    k0_chol: torch.Tensor
    k0_inv: torch.Tensor
    r_eigvecs: torch.Tensor
    r_eigvals: torch.Tensor
    small_sqrt_minus_i: torch.Tensor


def _deroos_pair_c(kpp4: torch.Tensor) -> torch.Tensor:
    n = kpp4.shape[0]
    device, dtype = kpp4.device, kpp4.dtype
    dim = n * n
    row = torch.arange(dim, device=device, dtype=torch.long)
    col = get_c_colidx(n, device)
    c = torch.zeros((dim, dim), device=device, dtype=dtype)
    c[row, col] = kpp4.reshape(-1)
    return 0.5 * (c + c.T)


def prepare_deroos_gradient_sampler(
    inputs: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
    *,
    rank_tol: float | None = None,
) -> deroos_gradient_sampler_state:
    """Prepare an exact sampler for g = grad f(X) using deRoos algebra.

    The expensive preprocessing is independent of the input dimension except
    through n by n geometry matrices, but it contains eigendecompositions of
    n^2 by n^2 matrices.  This is intended for moderate n and large d.
    """
    n, d = inputs.shape
    device, dtype = inputs.device, inputs.dtype
    w = inv_lengthscale_sq(lengthscale, d, device, dtype)

    delta = inputs[:, None, :] - inputs[None, :, :]
    v = delta * w.view(1, 1, d)
    r = (delta * v).sum(dim=-1)
    _, kp, kpp = k_kp_kpp_from_r(r, kernel, outputscale)
    k0 = (-2.0) * kp
    k0 = 0.5 * (k0 + k0.T)
    try:
        k0_chol = torch.linalg.cholesky(k0)
    except RuntimeError:
        k0_chol = cholesky_with_jitter(k0)
    eye_n = torch.eye(n, device=device, dtype=dtype)
    k0_inv = torch.cholesky_solve(eye_n, k0_chol)
    k0_inv = 0.5 * (k0_inv + k0_inv.T)

    kpp4 = 4.0 * kpp
    c = _deroos_pair_c(0.5 * (kpp4 + kpp4.T))

    g = (inputs * w.view(1, d)) @ inputs.T
    g = 0.5 * (g + g.T)
    l = get_l_dense(n, device, dtype).contiguous()
    pair_gram = torch.einsum("ij,kl->ikjl", k0_inv, g).reshape(n * n, n * n).contiguous()
    r_matrix = l.T @ pair_gram @ l
    r_matrix = 0.5 * (r_matrix + r_matrix.T)

    evals, evecs = torch.linalg.eigh(r_matrix)
    max_eval = torch.clamp(evals.max(), min=evals.new_tensor(1.0))
    if rank_tol is None:
        rank_tol = 100.0 * torch.finfo(dtype).eps * max(n * n, 1)
    keep = evals > (rank_tol * max_eval)
    if not bool(keep.any().item()):
        return deroos_gradient_sampler_state(
            inputs=inputs,
            w=w,
            k0=k0,
            k0_chol=k0_chol,
            k0_inv=k0_inv,
            r_eigvecs=evecs[:, :0].contiguous(),
            r_eigvals=evals[:0].contiguous(),
            small_sqrt_minus_i=torch.zeros((0, 0), device=device, dtype=dtype),
        )

    q_matrix = evecs[:, keep].contiguous()
    lam = evals[keep].contiguous()
    sqrt_lam = torch.sqrt(lam)
    c_range = q_matrix.T @ c @ q_matrix
    s = (
        torch.eye(lam.numel(), device=device, dtype=dtype)
        + (sqrt_lam[:, None] * c_range) * sqrt_lam[None, :]
    )
    s = 0.5 * (s + s.T)
    s_eval, s_evec = torch.linalg.eigh(s)

    min_allowed = (
        -1000.0
        * torch.finfo(dtype).eps
        * torch.clamp(s_eval.abs().max(), min=s_eval.new_tensor(1.0))
    )
    if bool((s_eval < min_allowed).any().item()):
        raise RuntimeError(
            f"deRoos sampler square-root matrix is not PSD. min eigenvalue={float(s_eval.min().item()):.3e}"
        )
    s_sqrt = torch.sqrt(torch.clamp(s_eval, min=1e-12))
    s_matrix_sqrt = (s_evec * s_sqrt.view(1, -1)) @ s_evec.T
    small_sqrt_minus_i = s_matrix_sqrt - torch.eye(lam.numel(), device=device, dtype=dtype)
    small_sqrt_minus_i = 0.5 * (small_sqrt_minus_i + small_sqrt_minus_i.T)

    return deroos_gradient_sampler_state(
        inputs=inputs,
        w=w,
        k0=k0,
        k0_chol=k0_chol,
        k0_inv=k0_inv,
        r_eigvecs=q_matrix,
        r_eigvals=lam,
        small_sqrt_minus_i=small_sqrt_minus_i.contiguous(),
    )


def _deroos_lowrank_sqrt_update_action(
    beta: torch.Tensor, state: deroos_gradient_sampler_state
) -> torch.Tensor:
    """Return M beta where I + B C B^T = (I + B M B^T)^2 on range(B)."""
    q_matrix = state.r_eigvecs
    lam = state.r_eigvals
    if lam.numel() == 0:
        return torch.zeros_like(beta)
    sqrt_lam = torch.sqrt(lam)
    coeff = (q_matrix.T @ beta) / sqrt_lam
    coeff = state.small_sqrt_minus_i @ coeff
    coeff = coeff / sqrt_lam
    return q_matrix @ coeff


def sample_gradients_deroos(
    inputs: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
    *,
    generator: torch.Generator | None = None,
    state: deroos_gradient_sampler_state | None = None,
) -> torch.Tensor:
    """Draw an exact finite-dimensional sample of grad f(X).

    The sampler uses the gradient covariance decomposition of the stationary derivative
    covariance. It avoids the n d by n d covariance matrix and draws from the
    exact covariance through a Kronecker base sample plus an exact low-rank
    square-root update over ordered point pairs.
    """
    if state is None:
        state = prepare_deroos_gradient_sampler(inputs, lengthscale, outputscale, kernel)
    n, d = inputs.shape
    eps = torch.randn((n, d), generator=generator, device=inputs.device, dtype=inputs.dtype)

    base = (state.k0_chol @ eps) * torch.sqrt(state.w).view(1, d)
    beta = deroos_apply_ut_ainv(base, state.inputs, state.w, state.k0_inv)
    pair_update = _deroos_lowrank_sqrt_update_action(beta, state)
    return base + deroos_apply_u(pair_update, state.inputs, state.w)


def sample_value_grad_observations_deroos(
    inputs: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
    *,
    sigma_f: float = 0.0,
    sigma_g: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw exact noiseless-gradient derivative-GP observations at X.

    This returns observed function values and observed gradients from the joint
    Gaussian model. Function noise ``sigma_f`` is supported through the
    conditional f | g covariance. Gradient noise is not supported in this
    sampler because the current deRoos gradient algebra assumes sigma_g = 0.
    """
    if sigma_g != 0.0:
        raise ValueError("de Roos exact sampler currently requires sigma_g=0.")
    n, d = inputs.shape
    state = prepare_deroos_gradient_sampler(inputs, lengthscale, outputscale, kernel)
    g = sample_gradients_deroos(
        inputs, lengthscale, outputscale, kernel, generator=generator, state=state
    )

    zeros = torch.zeros(n, device=inputs.device, dtype=inputs.dtype)
    cond = prepare_value_grad_deroos_conditioner(
        inputs,
        zeros,
        g,
        lengthscale=lengthscale,
        outputscale=outputscale,
        kernel=kernel,
        sigma_f=sigma_f,
        sigma_g=0.0,
    )
    x_g = (
        apply_deroos_gradient_inverse(cond.gradient_inverse, g.reshape(n * d, 1).unsqueeze(0))
        .squeeze(0)
        .squeeze(-1)
    )
    mean_f = cond.kfg @ x_g
    eps_f = torch.randn((n,), generator=generator, device=inputs.device, dtype=inputs.dtype)
    f = mean_f + cond.schur_chol @ eps_f
    z = torch.cat([f.unsqueeze(-1), g], dim=-1).reshape(-1).contiguous()
    return f.contiguous(), g.contiguous(), z
