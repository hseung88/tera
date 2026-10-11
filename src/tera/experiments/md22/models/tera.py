from tera.methods.tera.model import tera_model as __model

from .validation import md22_validation


class tera_model(md22_validation, __model):
    """Batched tera with MD22 validation metrics."""

    pass
