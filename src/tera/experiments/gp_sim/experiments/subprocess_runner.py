from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path as path_type
from typing import Any as any_type

from tera.experiments.gp_sim.config import method_vecchia_dgp_deroos, experiment_config, validate_methods
from tera.experiments.gp_sim.experiments.gp_sim_d_sweep import (
    choose_exact_reference_method,
    is_method_feasible,
    log_setting_summary,
    make_skipped_row,
    resolve_conditioning_size,
)
from tera.experiments.gp_sim.experiments.subprocess_worker import config_to_json
from tera.experiments.gp_sim.results import experiment_row, save_rows


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    src_dir = str(path_type(__file__).resolve().parents[4])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if existing == "" else src_dir + os.pathsep + existing
    return env


def _run_worker(args: list[str]) -> dict[str, any_type]:
    proc = subprocess.run(
        [sys.executable, "-m", "tera.experiments.gp_sim.cli.subprocess_worker", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_worker_env(),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.strip()
            or proc.stdout.strip()
            or f"worker exited with code {proc.returncode}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("worker produced no JSON output")
    return json.loads(lines[-1])


def _row_from_json(raw: dict[str, any_type]) -> experiment_row:
    return experiment_row(**raw)


def _prepare_cache_subprocess(
    *,
    cfg_json: str,
    cache_path: path_type,
    d: int,
    n_train: int,
    m_run: int,
    repeat: int,
    ref_method: str | None,
    vec_ref_method: str | None,
) -> None:
    _run_worker(
        [
            "prepare",
            "--config-json",
            cfg_json,
            "--cache-path",
            str(cache_path),
            "--d",
            str(d),
            "--n-train",
            str(n_train),
            "--m",
            str(m_run),
            "--repeat",
            str(repeat),
            "--ref-method",
            ref_method or "",
            "--vec-ref-method",
            vec_ref_method or "",
        ]
    )


def _run_method_subprocess(
    *,
    cfg_json: str,
    cache_path: path_type,
    method_name: str,
    d: int,
    n_train: int,
    n_eval: int,
    m_run: int,
    repeat: int,
) -> experiment_row:
    raw = _run_worker(
        [
            "run-method",
            "--config-json",
            cfg_json,
            "--cache-path",
            str(cache_path),
            "--method",
            method_name,
            "--d",
            str(d),
            "--n-train",
            str(n_train),
            "--n-eval",
            str(n_eval),
            "--m",
            str(m_run),
            "--repeat",
            str(repeat),
        ]
    )
    return _row_from_json(raw)


def run_gp_sim_d_sweep_subprocess(
    cfg: experiment_config,
    *,
    verbose: bool = False,
    cache_dir: str | path_type | None = None,
    keep_cache: bool = False,
) -> list[experiment_row]:
    """Run each method in an isolated process.

    Data and reference marginals are shared across methods and loaded before timing.
    """
    rows: list[experiment_row] = []
    n_train_values = cfg.n_train_values if cfg.n_train_values is not None else [cfg.n_train]
    m_values_raw = cfg.m_values if cfg.m_values is not None else [cfg.m]
    methods = validate_methods(list(cfg.methods))

    if cache_dir is None:
        tmp_obj = tempfile.TemporaryDirectory(prefix="gp_sim_kl_subproc_")
        cache_root = path_type(tmp_obj.name)
    else:
        tmp_obj = None
        cache_root = path_type(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)

    total_settings = len(cfg.d_values) * len(n_train_values) * len(m_values_raw)
    setting_ctr = 0

    try:
        for n_train in n_train_values:
            cfg_n = cfg if n_train == cfg.n_train else replace(cfg, n_train=n_train)
            cfg_json = config_to_json(cfg_n)

            for d in cfg.d_values:
                for m_spec in m_values_raw:
                    m_run = resolve_conditioning_size(m_spec, n_train)
                    setting_ctr += 1
                    setting_rows: list[experiment_row] = []

                    if verbose:
                        print(
                            f"\n[setting {setting_ctr}/{total_settings}] "
                            f"n={n_train} d={d} m={m_run} | "
                            f"running {cfg.repeats} repeat(s) with subprocess isolation",
                            flush=True,
                        )

                    for repeat in range(cfg.repeats):

                        def add_row(row: experiment_row) -> None:
                            rows.append(row)
                            setting_rows.append(row)

                        ref_method = choose_exact_reference_method(
                            methods,
                            cfg_n,
                            n_train=n_train,
                            d=d,
                            m_run=m_run,
                        )
                        vec_ref_method: str | None = None
                        if method_vecchia_dgp_deroos in methods:
                            ok_vec, _ = is_method_feasible(
                                cfg_n,
                                method_vecchia_dgp_deroos,
                                n_train=n_train,
                                d=d,
                                m_run=m_run,
                            )
                            if ok_vec:
                                vec_ref_method = method_vecchia_dgp_deroos

                        cache_path = cache_root / f"cache_n{n_train}_d{d}_m{m_run}_r{repeat}.pt"
                        try:
                            if verbose:
                                refs = (
                                    ", ".join(
                                        x for x in [ref_method, vec_ref_method] if x is not None
                                    )
                                    or "none"
                                )
                                print(
                                    f"  repeat {repeat + 1}/{cfg.repeats} | prepare data/ref ({refs})",
                                    flush=True,
                                )
                            _prepare_cache_subprocess(
                                cfg_json=cfg_json,
                                cache_path=cache_path,
                                d=d,
                                n_train=n_train,
                                m_run=m_run,
                                repeat=repeat,
                                ref_method=ref_method,
                                vec_ref_method=vec_ref_method,
                            )
                        except Exception as exc:
                            status = (
                                f"data/reference preparation failed: {type(exc).__name__}: {exc}"
                            )
                            if verbose:
                                print(f"  repeat {repeat + 1}/{cfg.repeats} | {status}", flush=True)
                            for method_name in methods:
                                add_row(
                                    make_skipped_row(
                                        cfg,
                                        repeat,
                                        d,
                                        n_train,
                                        cfg.n_eval,
                                        m_run,
                                        method_name,
                                        status,
                                    )
                                )
                            continue

                        for method_name in methods:
                            ok, reason = is_method_feasible(
                                cfg_n,
                                method_name,
                                n_train=n_train,
                                d=d,
                                m_run=m_run,
                            )
                            if not ok:
                                add_row(
                                    make_skipped_row(
                                        cfg,
                                        repeat,
                                        d,
                                        n_train,
                                        cfg.n_eval,
                                        m_run,
                                        method_name,
                                        reason,
                                    )
                                )
                                continue

                            try:
                                if verbose:
                                    print(
                                        f"  repeat {repeat + 1}/{cfg.repeats} | {method_name}",
                                        flush=True,
                                    )
                                row = _run_method_subprocess(
                                    cfg_json=cfg_json,
                                    cache_path=cache_path,
                                    method_name=method_name,
                                    d=d,
                                    n_train=n_train,
                                    n_eval=cfg.n_eval,
                                    m_run=m_run,
                                    repeat=repeat,
                                )
                                add_row(row)
                            except Exception as exc:
                                status = f"failed: {type(exc).__name__}: {exc}"
                                if verbose:
                                    print(
                                        f"  repeat {repeat + 1}/{cfg.repeats} | {method_name} | {status}",
                                        flush=True,
                                    )
                                add_row(
                                    make_skipped_row(
                                        cfg,
                                        repeat,
                                        d,
                                        n_train,
                                        cfg.n_eval,
                                        m_run,
                                        method_name,
                                        status,
                                    )
                                )

                        if not keep_cache:
                            try:
                                cache_path.unlink(missing_ok=True)
                            except TypeError:
                                if cache_path.exists():
                                    cache_path.unlink()

                    if verbose:
                        log_setting_summary(
                            setting_rows,
                            n_train=n_train,
                            d=d,
                            m_run=m_run,
                            repeats=cfg.repeats,
                        )

    finally:
        if tmp_obj is not None:
            tmp_obj.cleanup()
        elif (not keep_cache) and cache_dir is not None:
            try:
                if not any(path_type(cache_dir).iterdir()):
                    shutil.rmtree(cache_dir)
            except FileNotFoundError:
                pass

    return rows


def run_and_save_subprocess(
    cfg: experiment_config,
    outdir: str | path_type,
    *,
    verbose: bool = False,
    cache_dir: str | path_type | None = None,
    keep_cache: bool = False,
) -> list[experiment_row]:
    outdir = path_type(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = run_gp_sim_d_sweep_subprocess(
        cfg,
        verbose=verbose,
        cache_dir=cache_dir,
        keep_cache=keep_cache,
    )
    save_rows(rows, outdir / "results.csv")
    return rows
