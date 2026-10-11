from __future__ import annotations

from tera.experiments.gp_sim.config import method_exact_dgp, method_exact_dgp_deroos, method_exact_gp, method_tera, method_vecchia_dgp, method_vecchia_dgp_deroos, method_vecchia_gp
from tera.experiments.gp_sim.models.base import marginal_predictor
from tera.experiments.gp_sim.models.exact_dgp import exact_derivative_gp_predictor
from tera.experiments.gp_sim.models.exact_dgp_deroos import exact_derivative_gp_deroos_predictor
from tera.experiments.gp_sim.models.exact_gp import exact_gp_predictor
from tera.experiments.gp_sim.models.tera import tera_predictor
from tera.experiments.gp_sim.models.vecchia_dgp import vecchia_dgp_predictor
from tera.experiments.gp_sim.models.vecchia_dgp_deroos import vecchia_dgp_deroos_predictor
from tera.experiments.gp_sim.models.vecchia_gp import vecchia_gp_predictor


def create_predictor(
    method_name: str, *, m: int, prediction_batch_size: int = 32
) -> marginal_predictor:
    if method_name == method_exact_gp:
        return exact_gp_predictor()
    if method_name == method_exact_dgp:
        return exact_derivative_gp_predictor()
    if method_name == method_exact_dgp_deroos:
        return exact_derivative_gp_deroos_predictor()
    if method_name == method_vecchia_gp:
        return vecchia_gp_predictor(m=m)
    if method_name == method_vecchia_dgp:
        return vecchia_dgp_predictor(m=m)
    if method_name == method_vecchia_dgp_deroos:
        return vecchia_dgp_deroos_predictor(m=m)
    if method_name == method_tera:
        return tera_predictor(m=m, prediction_batch_size=prediction_batch_size)
    raise ValueError(f"Unknown method: {method_name}")
