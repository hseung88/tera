from __future__ import annotations

import argparse
from pathlib import Path as path_type

from tera.experiments.gp_sim.cli.run_gp_sim_kl import _apply_overrides
from tera.experiments.gp_sim.cli.run_gp_sim_kl import build_parser as build_standard_parser
from tera.experiments.gp_sim.config import load_config
from tera.experiments.gp_sim.experiments.subprocess_runner import run_and_save_subprocess


def build_parser() -> argparse.ArgumentParser:
    parser = build_standard_parser()
    parser.description = (
        "Run GP-simulated fidelity experiments with subprocess-isolated "
        "method timing and CUDA memory measurement."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for per-setting data/reference caches. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep per-setting data/reference cache files after the run.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args)
    outdir = path_type(args.outdir) if args.outdir is not None else path_type("outputs") / cfg.experiment_name
    run_and_save_subprocess(
        cfg,
        outdir,
        verbose=args.verbose,
        cache_dir=args.cache_dir,
        keep_cache=args.keep_cache,
    )
    print(f"Saved subprocess-isolated results to {outdir}")


if __name__ == "__main__":
    main()
