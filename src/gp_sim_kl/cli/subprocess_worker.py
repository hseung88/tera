from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from gp_sim_kl.experiments.subprocess_worker import (
    config_from_json,
    prepare_cache,
    run_method_from_cache,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Worker for subprocess-isolated GP simulation benchmarks.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--config-json", required=True)
    p_prepare.add_argument("--cache-path", required=True)
    p_prepare.add_argument("--d", type=int, required=True)
    p_prepare.add_argument("--n-train", type=int, required=True)
    p_prepare.add_argument("--m", type=int, required=True)
    p_prepare.add_argument("--repeat", type=int, required=True)
    p_prepare.add_argument("--ref-method", default="")
    p_prepare.add_argument("--vec-ref-method", default="")

    p_run = sub.add_parser("run-method")
    p_run.add_argument("--config-json", required=True)
    p_run.add_argument("--cache-path", required=True)
    p_run.add_argument("--method", required=True)
    p_run.add_argument("--d", type=int, required=True)
    p_run.add_argument("--n-train", type=int, required=True)
    p_run.add_argument("--n-eval", type=int, required=True)
    p_run.add_argument("--m", type=int, required=True)
    p_run.add_argument("--repeat", type=int, required=True)
    return parser

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config_from_json(args.config_json)

    if args.cmd == "prepare":
        prepare_cache(
            cfg=cfg,
            cache_path=args.cache_path,
            d=args.d,
            n_train=args.n_train,
            m_run=args.m,
            repeat=args.repeat,
            ref_method=args.ref_method or None,
            vec_ref_method=args.vec_ref_method or None,
        )
        print(json.dumps({"status": "ok"}))
        return

    if args.cmd == "run-method":
        row = run_method_from_cache(
            cfg=cfg,
            cache_path=args.cache_path,
            method_name=args.method,
            d=args.d,
            n_train=args.n_train,
            n_eval=args.n_eval,
            m_run=args.m,
            repeat=args.repeat,
        )
        print(json.dumps(asdict(row), allow_nan=True))
        return

    raise RuntimeError(f"Unknown worker command: {args.cmd}")

if __name__ == "__main__":
    main(sys.argv[1:])
