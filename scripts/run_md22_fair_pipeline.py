
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from gp_sim_kl.utils import dtype_from_name
from md22_regression.config import MD22Config, load_config
from md22_regression.data import load_md22_raw, make_split, save_fair_split
from md22_regression.main import run as run_internal_md22

@dataclass(slots=True)
class PipelineSplitRow:
    dataset: str
    seed: int
    split_id: str
    preprocessing_version: str
    n_train: int
    n_test: int
    d: int
    n_atoms: int
    x_scale: float
    fair_split_npz: str
    metadata_json: str

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _default_dsoftki_repo() -> Path:
    return _repo_root() / "third_party" / "dsoftki_gp"

def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]

def _parse_int_csv(value: str | None) -> list[int]:
    if value is None:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]

def _apply_common_overrides(cfg: MD22Config, args: argparse.Namespace) -> MD22Config:
    if args.datasets:
        cfg.datasets = _split_csv(args.datasets)
    if args.seeds:
        cfg.seeds = _parse_int_csv(args.seeds)
    if args.n_train is not None:
        cfg.n_train = args.n_train
    if args.n_test is not None:
        cfg.n_test = args.n_test
    if args.device is not None:
        cfg.device = args.device
    if args.dtype is not None:
        cfg.dtype = args.dtype
    if getattr(args, "use_ard", None) is not None:
        cfg.use_ard = bool(args.use_ard)
    if args.kernel is not None:
        cfg.kernel = args.kernel
    if args.m is not None:
        cfg.m = args.m
    if args.sigma_f is not None:
        cfg.sigma_f = args.sigma_f
    if args.sigma_g is not None:
        cfg.sigma_g = args.sigma_g
    if args.lengthscale is not None:
        cfg.lengthscale = args.lengthscale
    if args.lengthscale_init is not None:
        cfg.lengthscale_init = args.lengthscale_init
    if args.learn_lengthscale is not None:
        cfg.exact_learn_lengthscale = args.learn_lengthscale
        cfg.tera_learn_lengthscale = args.learn_lengthscale
    if args.exact_max_train is not None:
        cfg.exact_max_train = args.exact_max_train
    if args.exact_train_epochs is not None:
        cfg.exact_train_epochs = args.exact_train_epochs
    if args.exact_train_steps is not None:
        cfg.exact_train_steps = args.exact_train_steps
    if args.exact_lr is not None:
        cfg.exact_lr = args.exact_lr
    if args.tera_train_steps is not None:
        cfg.tera_train_steps = args.tera_train_steps
    if args.tera_train_epochs is not None:
        cfg.tera_train_epochs = args.tera_train_epochs
    if args.tera_graph_refresh_epochs is not None:
        cfg.tera_graph_refresh_epochs = args.tera_graph_refresh_epochs
    if args.tera_batch_size is not None:
        cfg.tera_batch_size = args.tera_batch_size
    if args.tera_lr is not None:
        cfg.tera_lr = args.tera_lr
    if args.tera_learn_sigma_g is not None:
        cfg.tera_learn_sigma_g = args.tera_learn_sigma_g
    if args.tera_gradient_noise_model is not None:
        cfg.tera_gradient_noise_model = args.tera_gradient_noise_model
    cfg.log_training_curves = bool(getattr(args, "log_training_curves", False))
    cfg.curve_log_every = int(getattr(args, "curve_log_every", 5))
    if cfg.log_training_curves:
        cfg.exact_log_every = cfg.curve_log_every
        cfg.tera_log_every = cfg.curve_log_every
    else:
        cfg.exact_log_every = 0
        cfg.tera_log_every = 0
    if args.verbose:
        cfg.verbose = True
    return cfg

def _export_fair_splits(cfg: MD22Config, outdir: Path) -> pd.DataFrame:
    device = torch.device(cfg.device if torch.cuda.is_available() or not cfg.device.startswith("cuda") else "cpu")
    dtype = dtype_from_name(cfg.dtype)
    split_dir = outdir / "fair_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    rows: list[PipelineSplitRow] = []
    for dataset in cfg.datasets:
        raw = load_md22_raw(cfg.data_dir, dataset, device=device, dtype=dtype)
        for seed in cfg.seeds:
            split = make_split(
                raw,
                seed=seed,
                train_frac=cfg.train_frac,
                test_frac=cfg.test_frac,
                n_train=cfg.n_train,
                n_test=cfg.n_test,
                x_scale=cfg.x_scale,
                preprocessing_version=cfg.preprocessing_version,
            )
            npz_path = split_dir / f"{dataset}_seed{seed}_{split.split_id}.npz"
            save_fair_split(
                split,
                npz_path,
                seed=seed,
                train_frac=cfg.train_frac,
                test_frac=cfg.test_frac,
                n_train=cfg.n_train,
                n_test=cfg.n_test,
            )
            rows.append(
                PipelineSplitRow(
                    dataset=dataset,
                    seed=seed,
                    split_id=split.split_id,
                    preprocessing_version=split.preprocessing_version,
                    n_train=int(split.X_train.shape[0]),
                    n_test=int(split.X_test.shape[0]),
                    d=split.d,
                    n_atoms=split.n_atoms,
                    x_scale=float(split.scaler.x_scale),
                    fair_split_npz=str(npz_path),
                    metadata_json=str(npz_path.with_suffix(".json")),
                )
            )
    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(outdir / "fair_splits_manifest.csv", index=False)
    print(f"Exported {len(rows)} fair splits to {split_dir}", flush=True)
    print(f"Wrote manifest to {outdir / 'fair_splits_manifest.csv'}", flush=True)
    return df

def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    rendered = " ".join(shlex.quote(str(c)) for c in cmd)
    prefix = f"(cd {cwd} &&) " if cwd is not None else ""
    print(f"$ {prefix}{rendered}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, env=env, check=True)

def _default_external_batch_size(d: int) -> int:
    if d >= 1000:
        return 128
    if d >= 300:
        return 256
    if d >= 180:
        return 512
    return 1024

def _default_external_lr(d: int, method: str) -> float:
    """Released MD22 learning-rate schedule from DSoftKI's exp/run_md22.sh."""
    if method == "DDSVGP":
        if d >= 1000:
            return 0.0015
        if d >= 300:
            return 0.003
        if d >= 180:
            return 0.006
        return 0.012
    if d >= 1000:
        return 0.001
    if d >= 300:
        return 0.002
    if d >= 180:
        return 0.004
    return 0.008

def _scalar_lengthscale_arg(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("External DSoftKI/DDSVGP adapter only supports scalar lengthscale initialization.")
        return float(value[0])
    return float(value)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the complete fair MD22 pipeline from this repository. The pipeline exports canonical NPZ splits, "
            "optionally launches DSoftKI/DDSVGP from their released repository through a repo-local adapter, "
            "then runs internal methods and merges validated results."
        )
    )
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--datasets", default=None)
    p.add_argument("--seeds", default=None, help="Comma-separated seeds overriding the config, e.g. 6535 or 6535,8830.")
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=["float32", "float64"])
    p.add_argument("--kernel", default=None, choices=["rbf", "matern52"], help="Kernel for internal methods and default DSoftKI external kernel. DDSVGP external remains RBF-only.")
    p.add_argument("--m", type=int, default=None, help="TERA conditioning set size.")
    p.add_argument("--sigma-f", "--sigma_f", dest="sigma_f", type=float, default=None, help="Function/value noise variance for MD22 methods.")
    p.add_argument("--sigma-g", "--sigma_g", dest="sigma_g", type=float, default=None, help="Gradient noise variance for internal MD22 derivative methods. DSoftKI uses sigma_f*d, DDSVGP requires sigma_g=sigma_f.")
    p.add_argument("--lengthscale", type=float, default=None, help="Initial scalar lengthscale override. This is an initialization unless lengthscale learning is disabled.")
    p.add_argument("--lengthscale-init", default=None, choices=["median", "one"])
    p.add_argument("--learn-lengthscale", dest="learn_lengthscale", action="store_true", default=None)
    p.add_argument("--no-learn-lengthscale", dest="learn_lengthscale", action="store_false")
    p.add_argument("--internal-methods", default="Exact GP,TERA")
    p.add_argument("--external-methods", default="DSoftKI,DDSVGP")
    p.add_argument("--skip-internal", action="store_true")
    p.add_argument("--skip-external", action="store_true")
    p.add_argument("--dsoftki-repo", type=Path, default=None, help="Path to the vendored DSoftKI repo. Defaults to third_party/dsoftki_gp.")
    p.add_argument("--dsoftki-python", default=sys.executable, help="Python executable used for DSoftKI/DDSVGP. Defaults to the current Python.")
    p.add_argument("--external-runner", type=Path, default=None)
    p.add_argument("--external-results-csv", type=Path, default=None)
    p.add_argument("--external-epochs", type=int, default=50)
    p.add_argument("--external-batch-size", type=int, default=None)
    p.add_argument("--external-eval-batch-size", type=int, default=512)
    p.add_argument("--external-lr", type=float, default=None)
    p.add_argument("--external-kernel", default=None, choices=["rbf", "matern52"], help="Kernel for external DSoftKI. If omitted, follows --kernel/config. DDSVGP is RBF-only in the released code.")
    p.add_argument("--external-num-inducing", type=int, default=512)
    p.add_argument("--external-num-directions", type=int, default=2)
    p.add_argument("--external-dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--external-extra-args", default="")
    p.add_argument(
        "--external-original-force-scale",
        action="store_true",
        help=(
            "For external DSoftKI/DDSVGP reproduction diagnostics, remove the chain-rule factor from "
            "canonical gradients before fitting external methods. The canonical split remains g=x_scale*F/s, "
            "but the external adapter uses g_external=g/x_scale."
        ),
    )
    p.add_argument(
        "--external-joint-scale",
        dest="external_joint_scale",
        action="store_true",
        default=True,
        help=(
            "For external DSoftKI/DDSVGP, use the released joint value-derivative training scale. "
            "This is the default. Internal methods remain on the canonical energy-only scale."
        ),
    )
    p.add_argument(
        "--external-energy-scale",
        dest="external_joint_scale",
        action="store_false",
        help=(
            "Diagnostic mode: force external DSoftKI/DDSVGP onto the canonical energy-only scale."
        ),
    )
    p.add_argument(
        "--log-training-curves",
        action="store_true",
        help="Evaluate and log raw energy RMSE during training. This is expensive and intended for a separate curve run.",
    )
    p.add_argument("--curve-log-every", type=int, default=5, help="Training-step interval for RMSE curve logging.")
    p.add_argument("--exact-max-train", type=int, default=None, help="Maximum n_train allowed for dense Exact GP before skipping.")
    p.add_argument("--exact-train-epochs", type=int, default=None)
    p.add_argument("--exact-train-steps", type=int, default=None)
    p.add_argument("--exact-lr", "--exact_lr", dest="exact_lr", type=float, default=None)
    p.add_argument("--tera-train-steps", type=int, default=None)
    p.add_argument("--tera-train-epochs", type=int, default=None)
    p.add_argument("--tera-graph-refresh-epochs", type=int, default=None)
    p.add_argument("--tera-batch-size", type=int, default=None)
    p.add_argument("--tera-lr", "--tera_lr", dest="tera_lr", type=float, default=None)
    p.add_argument("--tera-gradient-noise-model", choices=["iid", "scaled"], default=None)
    p.add_argument("--tera-learn-sigma-g", "--tera_learn_sigma_g", dest="tera_learn_sigma_g", action="store_true",
                   default=None)
    p.add_argument("--no-tera-learn-sigma-g", dest="tera_learn_sigma_g", action="store_false")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--use-ard", dest="use_ard", action="store_true", default=None, help="Enable ARD lengthscales.")
    p.add_argument("--no-use-ard", dest="use_ard", action="store_false", help="Disable ARD lengthscales.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    root = _repo_root()
    outdir = Path(args.outdir).resolve()
    fair_outdir = outdir / "canonical_fair_data"
    external_csv = (args.external_results_csv or (outdir / "external_results.csv")).resolve()
    external_history_csv = (outdir / "external_training_curves.csv").resolve()
    final_outdir = outdir / "internal_and_merged"
    external_runner = (args.external_runner or (root / "external_runners" / "dsoftki" / "run_fair_md22_from_npz.py")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = _apply_common_overrides(load_config(args.config), args)
    manifest = pd.DataFrame() if args.dry_run else _export_fair_splits(cfg, fair_outdir)
    if args.dry_run:
        print(f"[dry-run] would export fair splits to {fair_outdir}", flush=True)

    external_methods = _split_csv(args.external_methods)
    if external_methods and not args.skip_external:
        if not external_runner.exists():
            raise FileNotFoundError(f"External runner not found: {external_runner}")
        dsoftki_repo = (args.dsoftki_repo or _default_dsoftki_repo()).resolve()
        if not dsoftki_repo.exists():
            raise FileNotFoundError(
                f"DSoftKI repository not found: {dsoftki_repo}. "
                "Place the released implementation under third_party/dsoftki_gp or pass --dsoftki-repo."
            )
        env = os.environ.copy()
        python_paths = [str(dsoftki_repo), str(root / "src")]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env.setdefault("WANDB_MODE", "disabled")
        extra = shlex.split(args.external_extra_args)
        external_lengthscale = _scalar_lengthscale_arg(cfg.lengthscale)
        external_kernel = args.external_kernel or cfg.kernel
        for _, row in manifest.iterrows():
            d = int(row["d"])
            batch_size = args.external_batch_size or _default_external_batch_size(d)
            for method in external_methods:
                lr = args.external_lr if args.external_lr is not None else _default_external_lr(d, method)
                cmd = [
                    args.dsoftki_python,
                    str(external_runner),
                    "--npz",
                    str(row["fair_split_npz"]),
                    "--method",
                    method,
                    "--out-csv",
                    str(external_csv),
                    "--history-out-csv",
                    str(external_history_csv),
                    "--dataset",
                    str(row["dataset"]),
                    "--seed",
                    str(int(row["seed"])),
                    "--device",
                    args.device or cfg.device,
                    "--dtype",
                    args.external_dtype,
                    "--epochs",
                    str(args.external_epochs),
                    "--batch-size",
                    str(batch_size),
                    "--eval-batch-size",
                    str(args.external_eval_batch_size),
                    "--lr",
                    str(lr),
                    "--num-inducing",
                    str(args.external_num_inducing),
                    "--kernel",
                    external_kernel,
                    "--sigma-f",
                    str(cfg.sigma_f),
                ]
                if external_lengthscale is not None:
                    cmd += ["--lengthscale", str(external_lengthscale)]
                if method == "DSoftKI":
                    cmd += ["--use-ard" if cfg.use_ard else "--no-use-ard"]
                if method == "DDSVGP":
                    cmd += ["--num-directions", str(args.external_num_directions)]
                if args.external_joint_scale:
                    cmd.append("--external-joint-scale")
                if getattr(args, "external_original_force_scale", False):
                    cmd.append("--external-original-force-scale")
                if args.log_training_curves:
                    cmd += ["--log-training-curves", "--curve-log-every", str(args.curve_log_every)]
                cmd += extra
                _run(cmd, cwd=dsoftki_repo, env=env, dry_run=args.dry_run)

    internal_methods = _split_csv(args.internal_methods)
    merged_methods = internal_methods + ([] if args.skip_external else external_methods)
    if merged_methods and not args.skip_internal:
        run_cfg = _apply_common_overrides(load_config(args.config), args)
        run_cfg.methods = merged_methods
        run_cfg.external_results_csv = str(external_csv)
        run_cfg.strict_external_fairness = True
        run_cfg.export_fair_data = True
        run_cfg.fair_data_dir = str(final_outdir / "fair_splits")
        if args.verbose:
            run_cfg.verbose = True
        if not args.dry_run:
            rows = run_internal_md22(run_cfg, final_outdir)
            print(f"Wrote {len(rows)} merged rows to {final_outdir / run_cfg.out_csv_name}", flush=True)
            internal_curve_path = final_outdir / "training_curves.csv"
            curve_frames: list[pd.DataFrame] = []
            if internal_curve_path.exists():
                curve_frames.append(pd.read_csv(internal_curve_path))
            if external_history_csv.exists():
                curve_frames.append(pd.read_csv(external_history_csv))
            if curve_frames:
                pd.concat(curve_frames, ignore_index=True).to_csv(final_outdir / "training_curves.csv", index=False)
                print(f"Wrote merged training curves to {final_outdir / 'training_curves.csv'}", flush=True)
        else:
            print(f"[dry-run] would run merged methods {merged_methods} into {final_outdir}", flush=True)

    print("\nPipeline outputs:")
    print(f"  fair splits:     {fair_outdir / 'fair_splits'}")
    print(f"  manifest:        {fair_outdir / 'fair_splits_manifest.csv'}")
    print(f"  external CSV:    {external_csv}")
    print(f"  merged results:  {final_outdir / 'results.csv'}")

if __name__ == "__main__":
    main()
