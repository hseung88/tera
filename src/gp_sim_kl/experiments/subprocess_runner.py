from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from gp_sim_kl.config import (
    METHOD_VECCHIA_DGP_DEROOS,
    ExperimentConfig,
    validate_methods,
)
from gp_sim_kl.experiments.gp_sim_d_sweep import (
    choose_exact_reference_method,
    log_setting_summary,
    is_method_feasible,
    make_skipped_row,
    resolve_conditioning_size,
)
from gp_sim_kl.experiments.subprocess_worker import config_to_json
from gp_sim_kl.results import ExperimentRow, save_rows

def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parents[2])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if existing == "" else src_dir + os.pathsep + existing
    return env

def _run_worker(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "gp_sim_kl.cli.subprocess_worker", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_worker_env(),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"worker exited with code {proc.returncode}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("worker produced no JSON output")
    return json.loads(lines[-1])

def _row_from_json(raw: dict[str, Any]) -> ExperimentRow:
    return ExperimentRow(**raw)

def _prepare_cache_subprocess(
    *,
    cfg_json: str,
    cache_path: Path,
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
    cache_path: Path,
    method_name: str,
    d: int,
    n_train: int,
    n_eval: int,
    m_run: int,
    repeat: int,
) -> ExperimentRow:
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
    cfg: ExperimentConfig,
    *,
    verbose: bool = False,
    cache_dir: str | Path | None = None,
    keep_cache: bool = False,
) -> list[ExperimentRow]:
    """Run the GP-simulated experiment with subprocess-isolated methods.

    Run each method in an isolated process.

    Data and reference marginals are prepared once per setting and loaded by each
    method worker before timing starts.
    """
    rows: list[ExperimentRow] = []
    n_train_values = cfg.n_train_values if cfg.n_train_values is not None else [cfg.n_train]
    m_values_raw = cfg.m_values if cfg.m_values is not None else [cfg.m]
    methods = validate_methods(list(cfg.methods))

    if cache_dir is None:
        tmp_obj = tempfile.TemporaryDirectory(prefix="gp_sim_kl_subproc_")
        cache_root = Path(tmp_obj.name)
    else:
        tmp_obj = None
        cache_root = Path(cache_dir)
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
                    setting_rows: list[ExperimentRow] = []

                    if verbose:
                        print(
                            f"\n[setting {setting_ctr}/{total_settings}] "
                            f"n={n_train} d={d} m={m_run} | "
                            f"running {cfg.repeats} repeat(s) with subprocess isolation",
                            flush=True,
                        )

                    for repeat in range(cfg.repeats):
                        def add_row(row: ExperimentRow) -> None:
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
                        if METHOD_VECCHIA_DGP_DEROOS in methods:
                            ok_vec, _ = is_method_feasible(
                                cfg_n,
                                METHOD_VECCHIA_DGP_DEROOS,
                                n_train=n_train,
                                d=d,
                                m_run=m_run,
                            )
                            if ok_vec:
                                vec_ref_method = METHOD_VECCHIA_DGP_DEROOS

                        cache_path = cache_root / f"cache_n{n_train}_d{d}_m{m_run}_r{repeat}.pt"
                        try:
                            if verbose:
                                refs = ", ".join(x for x in [ref_method, vec_ref_method] if x is not None) or "none"
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
                            status = f"data/reference preparation failed: {type(exc).__name__}: {exc}"
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
                if not any(Path(cache_dir).iterdir()):
                    shutil.rmtree(cache_dir)
            except FileNotFoundError:
                pass

    return rows

def run_and_save_subprocess(
    cfg: ExperimentConfig,
    outdir: str | Path,
    *,
    verbose: bool = False,
    cache_dir: str | Path | None = None,
    keep_cache: bool = False,
) -> list[ExperimentRow]:
    rows = run_gp_sim_d_sweep_subprocess(
        cfg,
        verbose=verbose,
        cache_dir=cache_dir,
        keep_cache=keep_cache,
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, outdir / "results.csv")
    return rows
