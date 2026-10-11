"""BO parameter sweeps."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path as path_type

import pandas as pd

from tera.experiments.bo.cli import build_parser as build_run_parser
from tera.experiments.bo.cli import config_from_args as resolve_run_config


def positive_csv(value):
    try:
        values = list(dict.fromkeys(int(x.strip()) for x in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated positive integers.") from exc
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("Sweep values must be positive.")
    return values


def build_parser():
    parser = build_run_parser()
    parser.description = "Run BO parameter sweeps."
    parser.add_argument("--sweep", choices=["m", "prediction-batch", "budget"], required=True)
    parser.add_argument(
        "--values",
        type=positive_csv,
        required=True,
        help="Comma-separated positive integers; overrides the swept setting.",
    )
    return parser


def config_from_args(args):
    cfg = resolve_run_config(args)
    if args.sweep == "m" and any(m not in {"tera"} for m in cfg.methods):
        raise ValueError("m sweep supports tera")
    if args.sweep == "prediction-batch" and any(m not in {"tera"} for m in cfg.methods):
        raise ValueError("Prediction batch sweep supports tera")
    if cfg.n_init < 1 or cfg.batch_size < 1:
        raise ValueError("n_init and batch_size must be positive.")
    budgets = args.values if args.sweep == "budget" else [cfg.budget]
    if min(budgets) < cfg.n_init:
        raise ValueError("Every evaluation budget must be at least n_init.")
    return cfg


def sweep_settings(cfg, sweep, values):
    for value in values:
        if sweep == "m":
            setting = replace(cfg, tera_m=value)
        elif sweep == "prediction-batch":
            setting = replace(cfg, tera_prediction_batch_size=value)
        else:
            setting = replace(cfg, budget=value)
        yield f"{sweep}_{value}", value, setting


def run_sweep(cfg, args):

    from tera.experiments.bo.runner import run

    out = path_type(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    from tera.experiments.bo.runner import result_columns
    result_record_fields = result_columns
    rows = []
    for tag, value, setting in sweep_settings(cfg, args.sweep, args.values):
        print(f"[sweep] {args.sweep}={value}, methods={setting.methods}", flush=True)
        for row in run(setting, out / "runs" / tag, verbose=args.verbose):
            rows.append(
                dict(
                    asdict(row),
                    run_id=tag,
                    sweep=args.sweep,
                    sweep_value=value,
                    budget=setting.budget,
                    n_init=setting.n_init,
                )
            )
        pd.DataFrame([{key: row[key] for key in result_record_fields} for row in rows]).to_csv(out / "results.csv", index=False)
    print(f"Wrote {len(rows)} rows to {out / 'results.csv'}")
    return rows


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_sweep(config_from_args(args), args)


if __name__ == "__main__":
    main()
