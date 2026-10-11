from .model import tera_model as _tera_model


class tera_model(_tera_model):
    def _batch_nll(self, **kwargs):
        from .batched_training import batch_nll

        return batch_nll(gram_distances=True, **kwargs)
