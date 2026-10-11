import random
from contextlib import contextmanager
from enum import IntEnum as int_enum_type

import numpy as np
import torch


class random_stream(int_enum_type):
    """Fixed stream IDs; changing them changes the generated seeds."""

    gp_train_inputs = 0
    gp_eval_inputs = 1
    gp_observations = 2
    bo_step = 4
    training = 5
    acquisition = 6
    objective_noise = 7
    turbo_candidates = 8
    turbo_mask = 9
    turbo_posterior = 10
    turbo_restart = 11
    training_batch = 12
    derivative_subset = 13
    initialization = 14
    acquisition_mc = 15


def seed_for(seed: int, stream: random_stream, *indices: int) -> int:
    sequence = np.random.SeedSequence(
        int(seed), spawn_key=(int(stream), *(int(i) for i in indices))
    )
    return int(sequence.generate_state(1)[0])


@contextmanager
def rng_scope(seed: int, *, device: torch.device | str = "cpu"):
    device = torch.device(device)
    devices = []
    if device.type == "cuda":
        devices = [torch.cuda.current_device() if device.index is None else device.index]
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    with torch.random.fork_rng(devices=devices):
        try:
            torch.default_generator.manual_seed(int(seed))
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(int(seed))
            np.random.seed(np.random.SeedSequence(int(seed)).generate_state(1))
            random.seed(int(seed))
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
