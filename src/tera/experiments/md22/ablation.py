from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path as path_type

import numpy as np
import pandas as pd
import yaml

from tera.methods.names import md22_method_label
from tera.experiments.md22.runner import result_columns

from tera.experiments.md22.runner import (
    build_parser as main_parser,
)
from tera.experiments.md22.runner import (
    config_from_args as main_config,
)


def _positive_csv(value):
    try:
        values = list(dict.fromkeys(int(x.strip()) for x in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated positive integers.") from exc
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("Values must be positive.")
    return values


def _noise_levels(value):
    levels = []
    for item in value.split(","):
        token = item.strip().lower()
        try:
            fraction = (
                0.0
                if token == "clean"
                else (float(token[:-1]) / 100 if token.endswith("%") else float(token))
            )
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Use clean,10%,30% or 0,0.1,0.3.") from exc
        if not np.isfinite(fraction) or fraction < 0:
            raise argparse.ArgumentTypeError("Noise fractions must be finite and nonnegative.")
        if fraction not in levels:
            levels.append(fraction)
    return levels


def build_parser():
    parser = main_parser()
    parser.description = (
        "Run MD22 m or training/prediction batch sweeps with the main experiment options."
    )
    parser.add_argument("--sweep", choices=["m", "batch", "noise"], required=True)
    parser.add_argument("--m-values", type=_positive_csv, default="5,10,20,30,50")
    parser.add_argument("--batch-sizes", type=_positive_csv, default="16,32,64,128,256,512")
    parser.add_argument(
        "--noise-values",
        default="0.0001,0.001,0.01,0.1",
        help="Legacy noise sweep: comma-separated noise settings.",
    )
    parser.add_argument(
        "--noise-levels",
        type=_noise_levels,
        default=None,
        help="Training observation noise: clean,10%%,30%% or 0,0.1,0.3. Crossed with the sweep.",
    )
    parser.add_argument(
        "--energy-noise-levels",
        type=_noise_levels,
        default=None,
        help="Energy noise SD / clean training energy SD. E.g. clean,10%%,30%%.",
    )
    parser.add_argument(
        "--force-noise-levels",
        type=_noise_levels,
        default=None,
        help="Force noise SD / pooled clean training force SD. Crossed with energy levels.",
    )
    return parser


def config_from_args(args):
    cfg = main_config(args)

    if args.methods is None:
        cfg.methods = [m for m in cfg.methods if m in {"tera", "tera_batched", "vecchia"}]
    cfg.methods = list(dict.fromkeys(cfg.methods))
    if not cfg.methods or any(m not in {"tera", "tera_batched", "vecchia"} for m in cfg.methods):
        raise ValueError("MD22 ablations support --methods tera,vecchia only.")
    if not cfg.datasets or not cfg.seeds:
        raise ValueError("At least one dataset and seed are required.")
    if cfg.m < 1:
        raise ValueError("m must be positive.")
    return cfg


def _base_sweep_settings(cfg, args):
    if args.sweep == "m":
        return [
            (f"m_sweep_m{m}", replace(cfg, m=m, tera_m=None, vecchia_m=None), {"sweep": "m"})
            for m in args.m_values
        ]
    if args.sweep == "batch":
        return [
            (
                f"batch{b}",
                replace(
                    cfg,
                    tera_train_batch_size=b,
                    tera_prediction_batch_size=b,
                    vecchia_train_batch_size=b,
                    vecchia_prediction_batch_size=b,
                ),
                {"sweep": "batch", "batch_size": b},
            )
            for b in args.batch_sizes
        ]
    values = [float(x) for x in args.noise_values.split(",")]
    if not values or any(not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("Noise values must be finite and nonnegative.")
    return [
        (
            f"noise_sweep_sf{sf:g}_sg{sg:g}".replace(".", "p").replace("-", "m"),
            replace(
                cfg,
                sigma_f=sf,
                sigma_g=sg,
                tera_learn_sigma_f=False,
                tera_learn_sigma_g=False,
            ),
            {"sweep": "noise", "sigma_f_sweep": sf, "sigma_g_sweep": sg},
        )
        for sf in values
        for sg in values
    ]


def sweep_settings(cfg, args):
    from tera.experiments.md22.observation_noise import config_noise_fractions, noise_tag

    base = _base_sweep_settings(cfg, args)
    levels = getattr(args, "noise_levels", None)
    energy_levels = getattr(args, "energy_noise_levels", None)
    force_levels = getattr(args, "force_noise_levels", None)
    if levels is not None and (
        energy_levels is not None
        or force_levels is not None
        or getattr(args, "energy_noise_fraction", None) is not None
        or getattr(args, "force_noise_fraction", None) is not None
    ):
        raise ValueError("Use --noise-levels OR independent energy/force noise options, not both.")
    for kind, values in (("energy", energy_levels), ("force", force_levels)):
        if values is not None and getattr(args, f"{kind}_noise_fraction", None) is not None:
            raise ValueError(f"Use --{kind}-noise-levels OR --{kind}-noise-fraction, not both.")
    energy, force = config_noise_fractions(cfg)
    pairs = (
        [(v, v) for v in levels]
        if levels is not None
        else [
            (e, f)
            for e in (energy_levels if energy_levels is not None else [energy])
            for f in (force_levels if force_levels is not None else [force])
        ]
    )
    return [
        (
            tag if e == f == 0 else f"{tag}_{noise_tag(e, f)}",
            replace(
                setting,
                observation_noise_fraction=e if e == f else 0.0,
                energy_noise_fraction=e,
                force_noise_fraction=f,
            ),
            dict(
                metadata,
                observation_noise_fraction=e if e == f else np.nan,
                energy_noise_fraction=e,
                force_noise_fraction=f,
            ),
        )
        for e, f in pairs
        for tag, setting, metadata in base
    ]


def _classify_status(status):
    value = str(status)
    if value == "ok":
        return "ok"
    if "out of memory" in value.lower() or "outofmemoryerror" in value.lower():
        return "oom"
    return "error"


def _run_job(cfg, job_dir):
    """Fit and predict from scratch in a fresh process, without a time limit."""
    from tera.experiments.md22.config import resolve_method_config

    cfg = resolve_method_config(cfg, cfg.methods[0])
    job_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "input_config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    result_path = job_dir / cfg.out_csv_name
    result_path.unlink(missing_ok=True)
    env = os.environ.copy()
    src = str(path_type(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH", "")]))
    command = [
        sys.executable,
        "-u",
        "-m",
        "tera.experiments.md22.runner",
        "--config",
        str(config_path.resolve()),
        "--outdir",
        str(job_dir.resolve()),
        "--no-export-fair-data",
    ]
    start = time.perf_counter()
    with (job_dir / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    elapsed = time.perf_counter() - start
    row = {
        key: np.nan
        for key in (
            "n_train",
            "n_test",
            "d",
            "raw_energy_rmse_per_atom",
            "normalized_energy_rmse_per_atom",
            "fit_time_sec",
            "predict_time_sec",
            "wall_time_sec",
            "fit_peak_mem_gb",
            "predict_peak_mem_gb",
            "peak_mem_gb",
        )
    }
    status = "error"
    error = f"Worker exited with code {proc.returncode} without a valid result. See run.log."
    if proc.returncode == 0 and result_path.exists():
        try:
            frame = pd.read_csv(result_path)
            if len(frame) == 1:
                row.update(frame.iloc[0].to_dict())
                status = _classify_status(row["status"])
                error = "" if status == "ok" else str(row["status"])
        except (ValueError, KeyError, pd.errors.ParserError) as exc:
            error = f"Invalid worker result: {exc}"
    if status == "error" and not result_path.exists():
        lines = (job_dir / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
            error += " " + lines[-1][:500]
    if status == "ok" and cfg.m > row["n_train"]:
        status, error = "error", "Requested m exceeds n_train; not a valid m-sweep measurement."
    method = cfg.methods[0]
    prefix = "tera" if method == "tera_batched" else method
    from tera.experiments.md22.observation_noise import config_noise_fractions

    energy, force = config_noise_fractions(cfg)
    row.update(
        dataset=cfg.datasets[0],
        method=md22_method_label(method),
        seed=cfg.seeds[0],
        m=cfg.m,
        train_steps=getattr(cfg, f"{prefix}_train_steps"),
        train_epochs=getattr(cfg, f"{prefix}_train_epochs"),
        train_batch_size=getattr(cfg, f"{prefix}_train_batch_size"),
        prediction_batch_size=getattr(cfg, f"{prefix}_prediction_batch_size"),
        learning_rate=getattr(cfg, f"{prefix}_lr"),
        dtype=cfg.dtype,
        kernel=cfg.kernel,
        use_ard=cfg.use_ard,
        device=cfg.device,
        status=status,
        error=error,
        process_elapsed_sec=elapsed,
        worker_exit_code=proc.returncode,
        run_dir=str(job_dir.resolve()),
        observation_noise_fraction=energy if energy == force else np.nan,
        energy_noise_fraction=energy,
        force_noise_fraction=force,
    )
    if not cfg.device.startswith("cuda"):
        for key in ("peak_mem_gb", "fit_peak_mem_gb", "predict_peak_mem_gb"):
            row[key] = np.nan
    return row


def run_sweep(cfg, args):
    settings = sweep_settings(cfg, args)
    out = path_type(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for dataset in cfg.datasets:
        if path_type(dataset).name != dataset or dataset in {".", ".."}:
            raise ValueError("Dataset names must not contain path components.")
    rows = []
    count = len(settings) * len(cfg.datasets) * len(cfg.seeds) * len(cfg.methods)
    for tag, setting, metadata in settings:
        setting_dir = out / "runs" / tag
        setting_dir.mkdir(parents=True, exist_ok=True)
        setting_rows = []
        (setting_dir / "run.log").write_text("")
        for dataset in cfg.datasets:
            for seed in cfg.seeds:
                for method in cfg.methods:
                    job_cfg = replace(setting, datasets=[dataset], methods=[method], seeds=[seed])
                    print(
                        f"[{len(rows) + 1}/{count}] {dataset} {md22_method_label(method)} seed={seed} {tag}",
                        flush=True,
                    )
                    with tempfile.TemporaryDirectory(prefix="tera-md22-") as temp:
                        job_dir = path_type(temp)
                        row = _run_job(job_cfg, job_dir)
                        with (setting_dir / "run.log").open("a") as log:
                            log.write((job_dir / "run.log").read_text())
                    setting_rows.append({key: row.get(key, np.nan) for key in result_columns})
                    pd.DataFrame(setting_rows).to_csv(setting_dir / "results.csv", index=False)
                    row.update(metadata)
                    row.update(
                        figure={"m": "fig1", "noise": "fig2_panel1", "batch": "batch"}[args.sweep],
                        ablation=f"{args.sweep}_sweep",
                        sigma_f_sweep=metadata.get("sigma_f_sweep", np.nan),
                        sigma_g_sweep=metadata.get("sigma_g_sweep", np.nan),
                        n_train_requested=cfg.n_train, n_test_requested=cfg.n_test,
                    )
                    rows.append(row)
                    tmp = out / "tera_ablation_results.csv.tmp"
                    columns = result_columns + ["figure", "ablation", "sigma_f_sweep", "sigma_g_sweep", "n_train_requested", "n_test_requested"]
                    if args.sweep == "batch" or args.noise_levels or args.energy_noise_levels or args.force_noise_levels:
                        columns += ["sweep", "train_batch_size", "prediction_batch_size", "energy_noise_fraction", "force_noise_fraction"]
                    pd.DataFrame(rows).reindex(columns=columns).to_csv(tmp, index=False)
                    tmp.replace(out / "tera_ablation_results.csv")
                    print(
                        f"  {row['status']}: RMSE={row['raw_energy_rmse_per_atom']} "
                        f"time={row['wall_time_sec']} s memory={row['peak_mem_gb']} GiB",
                        flush=True,
                    )
    return rows


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
        run_sweep(cfg, args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
