from __future__ import annotations

from pathlib import Path as path_type

from tera.experiments.gp_sim.config import experiment_config
from tera.experiments.gp_sim.experiments.gp_sim_d_sweep import run_gp_sim_d_sweep
from tera.experiments.gp_sim.results import save_rows


def run_and_save(cfg: experiment_config, outdir: str | path_type, verbose: bool = False):
    outdir = path_type(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = run_gp_sim_d_sweep(cfg, verbose=verbose)
    save_rows(rows, outdir / "results.csv")
    return rows
