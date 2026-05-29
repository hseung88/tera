from __future__ import annotations

from pathlib import Path

from gp_sim_kl.config import ExperimentConfig
from gp_sim_kl.experiments.gp_sim_d_sweep import run_gp_sim_d_sweep
from gp_sim_kl.results import save_rows

def run_and_save(cfg: ExperimentConfig, outdir: str | Path, verbose: bool = False):
    rows = run_gp_sim_d_sweep(cfg, verbose=verbose)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_rows(rows, outdir / 'results.csv')
    return rows
