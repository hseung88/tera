from __future__ import annotations

import gc
import time
from collections.abc import Callable

import torch

def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        _sync_if_cuda(device)
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        _sync_if_cuda(device)

def bytes_to_gb(x: int) -> float:
    return float(x) / (1024.0 ** 3)

def time_build_predict_eval_peak_alloc(
    device: torch.device,
    build_fn: Callable[[], object],
    predict_fn: Callable[[], object],
    eval_fn: Callable[[object], object],
) -> tuple[object, float, float, float]:
    """Time model setup, prediction, metric evaluation, and CUDA peak memory."""
    if device.type == "cuda":
        _clear_cuda_cache(device)
        torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            build_fn()
        _sync_if_cuda(device)
        build_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        with torch.no_grad():
            pred = predict_fn()
            out = eval_fn(pred)
        _sync_if_cuda(device)
        predict_time = time.perf_counter() - t1

        peak_alloc = torch.cuda.max_memory_allocated(device)
        return out, build_time, predict_time, bytes_to_gb(int(peak_alloc))

    t0 = time.perf_counter()
    with torch.no_grad():
        build_fn()
    build_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    with torch.no_grad():
        pred = predict_fn()
        out = eval_fn(pred)
    predict_time = time.perf_counter() - t1

    return out, build_time, predict_time, 0.0
