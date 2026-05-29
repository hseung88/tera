
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import argparse

from md22_regression.plotting import plot_md22_results

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MD22 regression CSV results.")
    parser.add_argument("--csv", required=True, help="results.csv from run_md22_regression or run_md22_fair_pipeline")
    parser.add_argument("--out", required=True, help="Output figure path, e.g. figures/md22_regression.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    plot_md22_results(args.csv, args.out, title=args.title)

if __name__ == "__main__":
    main()
