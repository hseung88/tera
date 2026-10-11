from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class synthetic_problem:
    name: str
    dim: int
    active_dim: int
    lower: torch.Tensor
    upper: torch.Tensor
    optimum_value: float
    kind: str

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "synthetic_problem":
        return synthetic_problem(
            name=self.name,
            dim=self.dim,
            active_dim=self.active_dim,
            lower=self.lower.to(device=device, dtype=dtype),
            upper=self.upper.to(device=device, dtype=dtype),
            optimum_value=self.optimum_value,
            kind=self.kind,
        )

    def objective(self, x_unit: torch.Tensor) -> torch.Tensor:
        inputs = self._to_native(x_unit)
        if self.kind == "ackley":
            value = _ackley(inputs)
            return -value
        if self.kind == "levy":
            value = _levy(inputs)
            return -value
        if self.kind == "rastrigin":
            value = _rastrigin(inputs)
            return -value
        if self.kind == "rosenbrock":
            value = _rosenbrock(inputs)
            return -value
        if self.kind == "griewank":
            value = _griewank(inputs)
            return -value
        if self.kind == "hartmann6":
            return _hartmann6_reward(inputs)
        raise ValueError(f"Unknown synthetic problem kind: {self.kind}")

    def observe(
        self,
        x_unit: torch.Tensor,
        *,
        noise_std: float = 0.0,
        with_grad: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_req = x_unit.detach().clone().requires_grad_(with_grad)
        y = self.objective(x_req)
        grad = (
            torch.autograd.grad(y.sum(), x_req, create_graph=False)[0]
            if with_grad
            else torch.zeros_like(x_req)
        )
        if noise_std > 0.0:
            y = y + noise_std * torch.randn(
                y.shape, generator=generator, device=y.device, dtype=y.dtype
            )
        return y.detach(), grad.detach()

    def _to_native(self, x_unit: torch.Tensor) -> torch.Tensor:
        x_active = x_unit[..., : self.active_dim]
        lower = self.lower[: self.active_dim]
        upper = self.upper[: self.active_dim]
        return lower + (upper - lower) * x_active


def get_problem(
    name: str, *, device: torch.device, dtype: torch.dtype, config=None
) -> synthetic_problem:
    key = name.lower()
    parts = key.split("_")
    if len(parts) < 2:
        raise ValueError(f"Benchmark name must include dimension, e.g. ackley_1000. Got {name!r}.")
    dim = int(parts[-1])
    kind = "_".join(parts[:-1])
    if kind == "hartmann6":
        problem = synthetic_problem(
            key, dim, 6, torch.zeros(dim), torch.ones(dim), 3.32237, "hartmann6"
        )
    elif kind == "ackley":
        problem = synthetic_problem(
            key, dim, dim, torch.full((dim,), -5.0), torch.full((dim,), 5.0), 0.0, "ackley"
        )
    elif kind == "levy":
        problem = synthetic_problem(
            key, dim, dim, torch.full((dim,), -10.0), torch.full((dim,), 10.0), 0.0, "levy"
        )
    elif kind == "rastrigin":
        problem = synthetic_problem(
            key, dim, dim, torch.full((dim,), -5.12), torch.full((dim,), 5.12), 0.0, "rastrigin"
        )
    elif kind == "rosenbrock":
        problem = synthetic_problem(
            key, dim, dim, torch.full((dim,), -2.05), torch.full((dim,), 2.05), 0.0, "rosenbrock"
        )
    elif kind == "griewank":
        problem = synthetic_problem(
            key, dim, dim, torch.full((dim,), -10.0), torch.full((dim,), 10.0), 0.0, "griewank"
        )
    else:
        raise ValueError(f"Unknown benchmark {name!r}.")
    return problem.to(device=device, dtype=dtype)


def initial_design(
    n: int, dim: int, *, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=int(seed))
    return engine.draw(int(n)).to(device=device, dtype=dtype).clamp(0.0, 1.0)


def _ackley(inputs: torch.Tensor) -> torch.Tensor:
    sq = (inputs * inputs).mean(dim=-1)
    cos = torch.cos(2.0 * math.pi * inputs).mean(dim=-1)
    return -20.0 * torch.exp(-0.2 * torch.sqrt(sq)) - torch.exp(cos) + 20.0 + math.e


def _levy(inputs: torch.Tensor) -> torch.Tensor:
    w = 1.0 + (inputs - 1.0) / 4.0
    term1 = torch.sin(math.pi * w[..., 0]).pow(2)
    term3 = (w[..., -1] - 1.0).pow(2) * (1.0 + torch.sin(2.0 * math.pi * w[..., -1]).pow(2))
    mid = (w[..., :-1] - 1.0).pow(2) * (1.0 + 10.0 * torch.sin(math.pi * w[..., :-1] + 1.0).pow(2))
    return term1 + mid.sum(dim=-1) + term3


def _rastrigin(inputs: torch.Tensor) -> torch.Tensor:
    d = inputs.shape[-1]
    return 10.0 * d + (inputs.pow(2) - 10.0 * torch.cos(2.0 * math.pi * inputs)).sum(dim=-1)


def _rosenbrock(inputs: torch.Tensor) -> torch.Tensor:
    return (100.0 * (inputs[..., 1:] - inputs[..., :-1].pow(2)).pow(2) + (1.0 - inputs[..., :-1]).pow(2)).sum(
        dim=-1
    )


def _griewank(inputs: torch.Tensor) -> torch.Tensor:
    d = inputs.shape[-1]
    idx = torch.arange(1, d + 1, device=inputs.device, dtype=inputs.dtype)
    sum_term = inputs.pow(2).sum(dim=-1) / 4000.0
    prod_term = torch.cos(inputs / torch.sqrt(idx)).prod(dim=-1)
    return 1.0 + sum_term - prod_term


def _hartmann6_reward(inputs: torch.Tensor) -> torch.Tensor:
    alpha = inputs.new_tensor([1.0, 1.2, 3.0, 3.2])
    a = inputs.new_tensor(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ]
    )
    p = 1.0e-4 * inputs.new_tensor(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    diff2 = (inputs[..., None, :] - p).pow(2)
    inner = (a * diff2).sum(dim=-1)
    return (alpha * torch.exp(-inner)).sum(dim=-1)
