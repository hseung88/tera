from __future__ import annotations

from gp_sim_kl.config import (
    METHOD_EXACT_DGP,
    METHOD_EXACT_DGP_DEROOS,
    METHOD_EXACT_GP,
    METHOD_TERA,
    METHOD_VECCHIA_DGP,
    METHOD_VECCHIA_DGP_DEROOS,
    METHOD_VECCHIA_GP,
)
from gp_sim_kl.models.base import MarginalPredictor
from gp_sim_kl.models.exact_dgp import ExactDerivativeGPPredictor
from gp_sim_kl.models.exact_dgp_deroos import ExactDerivativeGPDeroosPredictor
from gp_sim_kl.models.exact_gp import ExactGPPredictor
from gp_sim_kl.models.tera import TERAPredictor
from gp_sim_kl.models.vecchia_dgp import VecchiaDGPPredictor
from gp_sim_kl.models.vecchia_dgp_deroos import VecchiaDGPDeroosPredictor
from gp_sim_kl.models.vecchia_gp import VecchiaGPPredictor

def create_predictor(method_name: str, *, m: int) -> MarginalPredictor:
    """Construct the predictor associated with a validated method name."""
    if method_name == METHOD_EXACT_GP:
        return ExactGPPredictor()
    if method_name == METHOD_EXACT_DGP:
        return ExactDerivativeGPPredictor()
    if method_name == METHOD_EXACT_DGP_DEROOS:
        return ExactDerivativeGPDeroosPredictor()
    if method_name == METHOD_VECCHIA_GP:
        return VecchiaGPPredictor(m=m)
    if method_name == METHOD_VECCHIA_DGP:
        return VecchiaDGPPredictor(m=m)
    if method_name == METHOD_VECCHIA_DGP_DEROOS:
        return VecchiaDGPDeroosPredictor(m=m)
    if method_name == METHOD_TERA:
        return TERAPredictor(m=m)
    raise ValueError(f"Unknown method: {method_name}")
