"""Matched-design cost comparison with explicit per-method location budgets."""

from dataclasses import asdict, replace
from pathlib import Path as path_type

import pandas as pd

from tera.experiments.bo.cli import build_parser, config_from_args
from tera.experiments.bo.runner import result_columns, run


def summarize(frame):
    records = []
    for (benchmark, method, seed), group in frame.groupby(["benchmark", "method", "seed"]):
        group = group.sort_values("eval_index")
        valid = group[group.status.isin(["init", "ok"])]
        if len(valid) != len(group) or valid.empty:
            records.append(dict(benchmark=benchmark, method=method, seed=seed, status="error"))
            continue
        last = valid.iloc[-1]
        records.append(
            dict(
                benchmark=benchmark,
                method=method,
                seed=seed,
                budget=int(last.get("budget", len(valid))),
                actual_locations=int(last.eval_index),
                n_init=int(last.get("n_init", valid.status.eq("init").sum())),
                regret=last.regret,
                wall_time_sec=float(valid[["fit_time_sec", "acq_time_sec", "eval_time_sec"]].sum().sum()),
                fit_time_sec=valid.fit_time_sec.sum(),
                acq_time_sec=valid.acq_time_sec.sum(),
                eval_time_sec=valid.eval_time_sec.sum(),
                peak_memory_gib=valid.iter_peak_memory_gb.max(),
                gradients_observed=bool(last.get("gradients_observed", method in {"tera", "lbfgsb"})),
                optimizer_message=last.get("optimizer_message", ""),
                optimizer_converged=last.get("optimizer_converged", None),
                optimizer_stopped_by_budget=last.get("optimizer_stopped_by_budget", None),
                status="ok"
                if method == "lbfgsb" or int(last.eval_index) == int(last.get("budget", len(valid)))
                else "incomplete",
            )
        )
    return pd.DataFrame(records)


def plot_cost(frame, outdir):
    """Plot each seed separately so early-stopping and unequal budgets remain visible."""
    import matplotlib.pyplot as plt

    from tera.experiments.bo.plotting import method_colors, method_labels, _setup_style

    _setup_style()

    for benchmark, sub in frame.groupby("benchmark"):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
        for method, runs in sub.groupby("method"):
            for j, (_, run_frame) in enumerate(runs.groupby("seed")):
                run_frame = run_frame[run_frame.status.isin(["init", "ok"])].sort_values(
                    "eval_index"
                )
                run_frame = run_frame.copy()
                run_frame["wall_time_sec"] = run_frame[["fit_time_sec", "acq_time_sec", "eval_time_sec"]].sum(axis=1).cumsum()
                for ax, x in zip(axes, ["eval_index", "wall_time_sec"]):
                    ax.step(
                        run_frame[x],
                        run_frame.regret.clip(lower=1e-14),
                        where="post",
                        label=method_labels.get(method, method) if j == 0 else None,
                        color=method_colors.get(method),
                        alpha=0.65,
                    )
        for ax, label in zip(
            axes, ["Evaluated locations (including initial design)", "Wall-clock time (s)"]
        ):
            ax.set(xlabel=label, ylabel="Regret", yscale="log", title=benchmark)
            ax.grid(alpha=0.2)
        axes[0].legend(frameon=False)
        fig.savefig(path_type(outdir) / f"{benchmark}_cost.pdf")
        plt.close(fig)


def main(argv=None):
    parser = build_parser()
    parser.description = "tera/vbo/l-bfgs-b cost comparison using shared initial designs."
    parser.add_argument("--tera-budget", type=int, default=150)
    parser.add_argument("--vbo-budget", type=int, default=300)
    parser.add_argument("--lbfgsb-budget", type=int, default=300)
    args = parser.parse_args(argv)
    if args.append_results:
        parser.error(
            "Cost suites use independent output directories; --append-results is unsupported."
        )
    cfg = config_from_args(args)
    methods = ["tera", "vbo", "lbfgsb"] if args.methods is None else cfg.methods
    if any(m not in {"tera", "vbo", "lbfgsb"} for m in methods):
        parser.error("Cost suite supports tera,vbo,lbfgsb.")
    budgets = {m: getattr(args, f"{m}_budget") for m in methods}
    if min(budgets.values()) < cfg.n_init or cfg.batch_size != 1:
        parser.error("Require all budgets >= n_init and query batch size 1.")
    outdir = path_type(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in methods:
        setting = replace(cfg, methods=[method], budget=budgets[method])
        rows.extend(
            dict(asdict(row), budget=setting.budget, n_init=setting.n_init)
            for row in run(setting, outdir / "runs" / method, verbose=args.verbose)
        )
        frame = pd.DataFrame(rows)
        frame.reindex(columns=result_columns).to_csv(outdir / "results.csv", index=False)
        summarize(frame).to_csv(outdir / "summary.csv", index=False)
    plot_cost(pd.DataFrame(rows), outdir)


if __name__ == "__main__":
    main()
