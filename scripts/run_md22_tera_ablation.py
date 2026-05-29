
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from md22_regression.config import MD22Config, load_config
from md22_regression.main import run as run_md22

DEFAULT_M_VALUES = [5, 10, 20, 30]
DEFAULT_NOISE_VALUES = [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4]

def _parse_csv_ints(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]

def _parse_csv_floats(value: str | None, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    return [float(x.strip()) for x in value.split(",") if x.strip()]

def _parse_seed_csv(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]

def _base_cfg(args: argparse.Namespace) -> MD22Config:
    cfg = load_config(args.config)
    cfg.datasets = [args.dataset]
    cfg.methods = ["TERA"]
    cfg.seeds = _parse_seed_csv(args.seeds, cfg.seeds)
    cfg.device = args.device
    cfg.kernel = args.kernel
    cfg.use_ard = bool(args.use_ard)
    cfg.lengthscale = args.lengthscale
    cfg.exact_max_train = 0
    cfg.export_fair_data = False
    cfg.out_csv_name = "results.csv"
    cfg.verbose = bool(args.verbose)
    cfg.tera_train_epochs = int(args.train_epochs)
    cfg.tera_train_steps = int(args.train_steps)
    cfg.tera_batch_size = int(args.batch_size)
    cfg.tera_lr = float(args.lr)
    cfg.tera_learn_lengthscale = bool(args.learn_lengthscale)
    cfg.tera_learn_sigma_g = bool(args.learn_sigma_g)
    cfg.tera_gradient_noise_model = args.gradient_noise_model
    if args.n_train is not None:
        cfg.n_train = int(args.n_train)
    if args.n_test is not None:
        cfg.n_test = int(args.n_test)
    return cfg

def _run_setting(cfg: MD22Config, outdir: Path, *, tag: str, meta: dict[str, object]) -> list[dict[str, object]]:
    setting_dir = outdir / "runs" / tag
    rows = run_md22(cfg, setting_dir)
    out: list[dict[str, object]] = []
    for row in rows:
        record = asdict(row)
        record.update(meta)
        record["n_train_requested"] = cfg.n_train
        record["n_test_requested"] = cfg.n_test
        out.append(record)
    return out

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the MD22 TERA ablations used for the paper figures.")
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--dataset", default="buckyball-catcher")
    p.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to the config seeds.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--kernel", default="rbf", choices=["rbf", "matern52"])
    p.add_argument("--use-ard", dest="use_ard", action="store_true", default=False)
    p.add_argument("--no-use-ard", dest="use_ard", action="store_false")
    p.add_argument("--lengthscale", type=float, default=1.0)
    p.add_argument("--learn-lengthscale", dest="learn_lengthscale", action="store_true", default=True)
    p.add_argument("--no-learn-lengthscale", dest="learn_lengthscale", action="store_false")
    p.add_argument("--sigma-f", type=float, default=1.0e-3)
    p.add_argument("--sigma-g", type=float, default=1.0e-3)
    p.add_argument("--gradient-noise-model", default="iid", choices=["iid", "scaled"])
    p.add_argument("--train-epochs", type=int, default=1)
    p.add_argument("--train-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--learn-sigma-g", dest="learn_sigma_g", action="store_true", default=True)
    p.add_argument("--no-learn-sigma-g", dest="learn_sigma_g", action="store_false")
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--m-values", default=None, help="Comma-separated values. Default: 5,10,20,30.")
    p.add_argument("--noise-values", default=None, help="Comma-separated variance grid. Default: 1e-1,1e-2,1e-3,1e-4.")
    p.add_argument("--skip-m-sweep", action="store_true")
    p.add_argument("--skip-noise-sweep", action="store_true")
    p.add_argument("--append-results", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / "tera_ablation_results.csv"

    all_rows: list[dict[str, object]] = []
    if args.append_results and result_path.exists():
        all_rows.extend(pd.read_csv(result_path).to_dict("records"))

    base = _base_cfg(args)
    base.sigma_f = float(args.sigma_f)
    base.sigma_g = float(args.sigma_g)

    if not args.skip_m_sweep:
        for m in _parse_csv_ints(args.m_values, DEFAULT_M_VALUES):
            cfg = _base_cfg(args)
            cfg.m = int(m)
            cfg.sigma_f = float(args.sigma_f)
            cfg.sigma_g = float(args.sigma_g)
            tag = f"m_sweep_m{m}"
            all_rows.extend(
                _run_setting(
                    cfg,
                    outdir,
                    tag=tag,
                    meta={
                        "figure": "fig1",
                        "ablation": "m_sweep",
                        "sigma_f_sweep": math.nan,
                        "sigma_g_sweep": math.nan,
                    },
                )
            )
            pd.DataFrame(all_rows).to_csv(result_path, index=False)

    if not args.skip_noise_sweep:
        for sigma_f in _parse_csv_floats(args.noise_values, DEFAULT_NOISE_VALUES):
            for sigma_g in _parse_csv_floats(args.noise_values, DEFAULT_NOISE_VALUES):
                cfg = _base_cfg(args)
                cfg.m = 20
                cfg.sigma_f = float(sigma_f)
                cfg.sigma_g = float(sigma_g)
                tag = f"noise_sweep_sf{sigma_f:g}_sg{sigma_g:g}".replace(".", "p").replace("-", "m")
                all_rows.extend(
                    _run_setting(
                        cfg,
                        outdir,
                        tag=tag,
                        meta={
                            "figure": "fig2_panel1",
                            "ablation": "noise_sweep",
                            "sigma_f_sweep": float(sigma_f),
                            "sigma_g_sweep": float(sigma_g),
                        },
                    )
                )
                pd.DataFrame(all_rows).to_csv(result_path, index=False)

    pd.DataFrame(all_rows).to_csv(result_path, index=False)
    print(f"Wrote {len(all_rows)} rows to {result_path}")

if __name__ == "__main__":
    main()
