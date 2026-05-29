from __future__ import annotations

import argparse

from md22_regression.plotting import plot_md22_results

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MD22 regression CSV results.")
    parser.add_argument("--csv", required=True, help="results.csv from run_md22_regression or run_md22_fair_pipeline")
    parser.add_argument("--out", required=True, help="Output figure path, e.g. figures/md22_regression.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    plot_md22_results(args.csv, args.out, title=args.title)
