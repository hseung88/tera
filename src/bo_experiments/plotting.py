from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

METHOD_ORDER = ["sobol", "vbo", "turbo-logei", "turbo", "tera"]
METHOD_LABELS = {
    "sobol": "Sobol",
    "vbo": "VBO",
    "turbo": "TuRBO-1 TS",
    "turbo-logei": "TuRBO-1",
    "tera": "TERA",
}
METHOD_COLORS = {
    "sobol": "#000000",
    "vbo": "#1F77B4",
    "turbo": "#FF7F0E",
    "turbo-logei": "#FF7F0E",
    "tera": "#D62728",
}
MARKERS = {
    "sobol": "X",
    "vbo": "o",
    "turbo": "^",
    "turbo-logei": "D",
    "tera": "*",
}
LINESTYLES = {
    "sobol": "--",
    "vbo": "-",
    "turbo": "-",
    "turbo-logei": "-",
    "tera": "-",
}

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method"] = df["method"].replace(
        {"turbo_logei": "turbo-logei", "random": "sobol"}
    )
    if "status" in df.columns:
        df = df[df["status"].astype(str).isin(["init", "ok"])]

    numeric_cols = [
        "eval_index",
        "regret",
        "best_y",
        "fit_time_sec",
        "acq_time_sec",
        "eval_time_sec",
        "fit_peak_memory_gb",
        "acq_peak_memory_gb",
        "iter_peak_memory_gb",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    time_cols = ["fit_time_sec", "acq_time_sec", "eval_time_sec"]
    if all(col in df.columns for col in time_cols):
        df["bo_wall_time_sec"] = (
            df["fit_time_sec"].fillna(0.0)
            + df["acq_time_sec"].fillna(0.0)
            + df["eval_time_sec"].fillna(0.0)
        )
        df = df.sort_values(["benchmark", "method", "seed", "eval_index"])
        df["cum_wall_time_sec"] = df.groupby(["benchmark", "method", "seed"])[
            "bo_wall_time_sec"
        ].cumsum()

    return df

def _ordered(values: Iterable[str]) -> list[str]:
    existing = list(dict.fromkeys(str(v) for v in values))
    return [m for m in METHOD_ORDER if m in existing] + [m for m in existing if m not in METHOD_ORDER]

def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
        }
    )

def _mean_sem_by_x(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    grouped = (
        df.dropna(subset=[x_col, y_col])
        .groupby(x_col, as_index=False)[y_col]
        .agg(mean="mean", sem="sem")
        .sort_values(x_col)
    )
    grouped["sem"] = grouped["sem"].fillna(0.0)
    return grouped

def _run_level_peak_memory(df: pd.DataFrame) -> pd.DataFrame:
    if "iter_peak_memory_gb" not in df.columns:
        raise ValueError("CSV is missing iter_peak_memory_gb, which is required for the memory panel.")
    return (
        df.dropna(subset=["iter_peak_memory_gb"])
        .groupby(["benchmark", "method", "seed"], as_index=False)["iter_peak_memory_gb"]
        .max()
        .rename(columns={"iter_peak_memory_gb": "peak_mem_gb"})
    )

def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)

def _benchmark_label(benchmark: str) -> str:
    return str(benchmark).replace("_", " ").title()


def _run_level_total_time(df: pd.DataFrame) -> pd.DataFrame:
    """Return run-level total BO wall-clock time.

    The canonical output column is wall_time_sec.  The total_time_sec
    alias is kept for compatibility with newer plotting code.
    """
    if "cum_wall_time_sec" not in df.columns:
        raise ValueError(
            "CSV is missing cumulative wall time. Required columns are "
            "fit_time_sec, acq_time_sec, and eval_time_sec."
        )
    out = (
        df.dropna(subset=["cum_wall_time_sec"])
        .groupby(["benchmark", "method", "seed"], as_index=False)["cum_wall_time_sec"]
        .max()
        .rename(columns={"cum_wall_time_sec": "wall_time_sec"})
    )
    out["total_time_sec"] = out["wall_time_sec"]
    return out

def _plot_resource_summary(
    df: pd.DataFrame,
    benchmark: str,
    out_path: Path,
    *,
    title: str | None = None,
) -> Path:
    sub = df[df["benchmark"].astype(str) == str(benchmark)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for benchmark={benchmark!r}")
    _setup_style()
    methods = _ordered(sub["method"].astype(str))
    times = _run_level_total_time(sub)
    mem = _run_level_peak_memory(sub)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3), constrained_layout=True)

    x = 0.8 * np.arange(len(methods), dtype=float)
    bar_width = 0.42

    def _bar_panel(ax, data: pd.DataFrame, value_col: str, ylabel: str) -> None:
        for i, method in enumerate(methods):
            vals = data[data["method"].astype(str) == method][value_col].dropna()
            if vals.empty:
                continue
            mean = float(vals.mean())
            err = float(vals.sem()) if len(vals) > 1 else 0.0
            ax.bar(
                x[i],
                mean,
                width=bar_width,
                color=METHOD_COLORS.get(method, None),
                edgecolor="black",
                linewidth=0.6,
                alpha=0.88,
                zorder=2,
            )
            if err > 0.0:
                ax.errorbar(x[i], mean, yerr=err, fmt="none", color="black", capsize=3, linewidth=1.0, zorder=3)
        ax.set_xlabel("Method")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([_method_label(m) for m in methods], rotation=0, ha="center")
        ax.set_xlim(x[0] - 0.35, x[-1] + 0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _bar_panel(axes[0], times, "wall_time_sec", "Wall-clock time (s)")
    _bar_panel(axes[1], mem, "peak_mem_gb", "Peak GPU memory (GB)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_bo_main_regret(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    benchmarks: list[str] | tuple[str, ...] = ("levy_200", "ackley_200"),
    title: str | None = None,
) -> Path:
    df = _clean(pd.read_csv(csv_path))
    _setup_style()

    benchs = [str(b) for b in benchmarks]
    if len(benchs) != 2:
        raise ValueError("plot_bo_main_regret expects exactly two benchmarks.")

    sub = df[df["benchmark"].astype(str).isin(benchs)].copy()
    if sub.empty:
        raise ValueError("No rows found for requested benchmarks.")

    methods = _ordered(sub["method"].astype(str))

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.6), constrained_layout=True)

    legend_handles: dict[str, object] = {}
    legend_labels: dict[str, str] = {}

    for panel_idx, benchmark in enumerate(benchs):
        ax = axes[panel_idx]
        bdf = sub[sub["benchmark"].astype(str) == benchmark].copy()

        for method in methods:
            mdf = bdf[bdf["method"].astype(str) == method]
            if mdf.empty:
                continue

            agg = _mean_sem_by_x(mdf, "eval_index", "regret")
            if agg.empty:
                continue

            color = METHOD_COLORS.get(method, None)
            is_tera = method in {"tera"}
            label = _method_label(method)

            (line,) = ax.plot(
                agg["eval_index"],
                agg["mean"],
                color=color,
                linestyle=LINESTYLES.get(method, "-"),
                linewidth=2.8 if is_tera else 2.2,
                alpha=1.0 if is_tera else 0.9,
                zorder=4 if is_tera else 3,
                label=label,
            )
            ax.fill_between(
                agg["eval_index"].to_numpy(dtype=float),
                (agg["mean"] - agg["sem"]).to_numpy(dtype=float),
                (agg["mean"] + agg["sem"]).to_numpy(dtype=float),
                color=color,
                alpha=0.18 if is_tera else 0.12,
                linewidth=0,
                zorder=1,
            )

            legend_handles.setdefault(method, line)
            legend_labels.setdefault(method, label)

        ax.set_xlabel("Query budget")
        if panel_idx == 0:
            ax.set_ylabel("Regret")
        else:
            ax.set_ylabel("")
        ax.set_yscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.03)

    ax_time = axes[2]
    ax_mem = ax_time.twinx()

    resource_benchmark = benchs[-1]
    rdf = sub[sub["benchmark"].astype(str) == resource_benchmark].copy()
    if rdf.empty:
        raise ValueError(f"No rows found for resource benchmark {resource_benchmark!r}")

    total_time = _run_level_total_time(rdf)
    peak_mem = _run_level_peak_memory(rdf)

    x = np.arange(len(methods), dtype=float)
    width = 0.34

    for i, method in enumerate(methods):
        color = METHOD_COLORS.get(method, None)

        tvals = total_time[total_time["method"].astype(str) == method]["wall_time_sec"].dropna()
        if not tvals.empty:
            tmean = float(tvals.mean())
            terr = float(tvals.sem()) if len(tvals) > 1 else 0.0
            ax_time.bar(
                x[i] - width / 2,
                tmean,
                width=width,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.90,
                zorder=2,
            )
            if terr > 0.0:
                ax_time.errorbar(
                    x[i] - width / 2,
                    tmean,
                    yerr=terr,
                    fmt="none",
                    color="black",
                    capsize=3,
                    linewidth=1.0,
                    zorder=3,
                )

        mvals = peak_mem[peak_mem["method"].astype(str) == method]["peak_mem_gb"].dropna()
        if not mvals.empty:
            mmean = float(mvals.mean())
            merr = float(mvals.sem()) if len(mvals) > 1 else 0.0
            ax_mem.bar(
                x[i] + width / 2,
                mmean,
                width=width,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.55,
                hatch="//",
                zorder=2,
            )
            if merr > 0.0:
                ax_mem.errorbar(
                    x[i] + width / 2,
                    mmean,
                    yerr=merr,
                    fmt="none",
                    color="black",
                    capsize=3,
                    linewidth=1.0,
                    zorder=3,
                )

    ax_time.set_xlabel("Method")
    ax_time.set_ylabel("Wall-clock time (s)")
    ax_mem.set_ylabel("Peak GPU memory (GB)")

    ax_time.set_xticks(x)
    ax_time.set_xticklabels([_method_label(m) for m in methods], rotation=0, ha="center")

    ax_time.spines["top"].set_visible(False)
    ax_mem.spines["top"].set_visible(False)
    ax_time.margins(x=0.05)

    resource_handles = [
        Patch(facecolor="gray", edgecolor="black", label="Time"),
        Patch(facecolor="gray", edgecolor="black", hatch="//", alpha=0.55, label="Memory"),
    ]
    ax_time.legend(
        handles=resource_handles,
        loc="upper left",
        frameon=False,
        fontsize=10,
    )

    handles = [legend_handles[m] for m in methods if m in legend_handles]
    labels = [legend_labels[m] for m in methods if m in legend_labels]
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 6),
            frameon=False,
            bbox_to_anchor=(0.5, 1.14),
            handlelength=2.4,
            columnspacing=1.1,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_bo_resource_summary(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    benchmark: str = "ackley_200",
    title: str | None = None,
) -> Path:
    """Create a 1x2 wall-clock and memory summary figure for one benchmark."""
    df = _clean(pd.read_csv(csv_path))
    return _plot_resource_summary(df, str(benchmark), Path(out_path), title=title)

def _plot_time_regret_and_memory(
    df: pd.DataFrame,
    benchmark: str,
    out_path: Path,
    *,
    title: str | None = None,
) -> Path:
    sub = df[df["benchmark"].astype(str) == str(benchmark)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for benchmark={benchmark!r}")
    if "regret" not in sub.columns:
        raise ValueError("CSV is missing regret.")
    if "eval_index" not in sub.columns:
        raise ValueError("CSV is missing eval_index.")
    if "cum_wall_time_sec" not in sub.columns:
        raise ValueError(
            "CSV is missing cumulative wall time. Required columns are "
            "fit_time_sec, acq_time_sec, and eval_time_sec."
        )

    _setup_style()
    methods = _ordered(sub["method"].astype(str))

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.6), constrained_layout=True)

    legend_handles: dict[str, object] = {}
    legend_labels: dict[str, str] = {}

    for method in methods:
        mdf = sub[sub["method"].astype(str) == method]
        if mdf.empty:
            continue

        color = METHOD_COLORS.get(method, None)
        is_tera = method in {"tera"}
        label = _method_label(method)

        agg_q = _mean_sem_by_x(mdf, "eval_index", "regret")
        if not agg_q.empty:
            (line,) = axes[0].plot(
                agg_q["eval_index"],
                agg_q["mean"],
                color=color,
                linestyle=LINESTYLES.get(method, "-"),
                linewidth=2.6,
                alpha=1.0 if is_tera else 0.88,
                label=label,
                zorder=4 if is_tera else 3,
            )
            axes[0].fill_between(
                agg_q["eval_index"].to_numpy(dtype=float),
                (agg_q["mean"] - agg_q["sem"]).to_numpy(dtype=float),
                (agg_q["mean"] + agg_q["sem"]).to_numpy(dtype=float),
                color=color,
                alpha=0.18 if is_tera else 0.12,
                linewidth=0,
                zorder=1,
            )
            legend_handles.setdefault(method, line)
            legend_labels.setdefault(method, label)

        agg_regret = _mean_sem_by_x(mdf, "eval_index", "regret")
        time_by_eval = (
            mdf.dropna(subset=["eval_index", "cum_wall_time_sec"])
            .groupby("eval_index", as_index=False)["cum_wall_time_sec"]
            .mean()
            .sort_values("eval_index")
        )

        if not agg_regret.empty and not time_by_eval.empty:
            merged = agg_regret.merge(time_by_eval, on="eval_index", how="inner")
            if not merged.empty:
                axes[1].plot(
                    merged["cum_wall_time_sec"],
                    merged["mean"],
                    color=color,
                    linestyle=LINESTYLES.get(method, "-"),
                    linewidth=2.6,
                    alpha=1.0 if is_tera else 0.88,
                    label=label,
                    zorder=4 if is_tera else 3,
                )
                axes[1].fill_between(
                    merged["cum_wall_time_sec"].to_numpy(dtype=float),
                    (merged["mean"] - merged["sem"]).to_numpy(dtype=float),
                    (merged["mean"] + merged["sem"]).to_numpy(dtype=float),
                    color=color,
                    alpha=0.18 if is_tera else 0.12,
                    linewidth=0,
                    zorder=1,
                )

    axes[0].set_xlabel("Query budget")
    axes[0].set_ylabel("Regret")

    def _k_formatter(x, pos):
        if abs(x) >= 1000:
            return f"{x / 1000:g}k"
        return f"{x:g}"

    axes[0].xaxis.set_major_formatter(FuncFormatter(_k_formatter))

    axes[1].set_xlabel("Wall-clock time (s)")
    axes[1].set_ylabel("Regret")

    peak = _run_level_peak_memory(sub)
    x = 0.8 * np.arange(len(methods), dtype=float)
    bar_width = 0.42

    for i, method in enumerate(methods):
        vals = peak[peak["method"].astype(str) == method]["peak_mem_gb"].dropna()
        if vals.empty:
            continue

        mean = float(vals.mean())
        err = float(vals.sem()) if len(vals) > 1 else 0.0

        axes[2].bar(
            x[i],
            mean,
            width=bar_width,
            color=METHOD_COLORS.get(method, None),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.88,
            zorder=2,
        )
        if err > 0.0:
            axes[2].errorbar(
                x[i],
                mean,
                yerr=err,
                fmt="none",
                color="black",
                capsize=3,
                linewidth=1.0,
                zorder=3,
            )

    axes[2].set_xlabel("Method")
    axes[2].set_ylabel("Peak GPU memory (GB)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([_method_label(m) for m in methods], rotation=0, ha="center")
    if len(x) > 0:
        axes[2].set_xlim(x[0] - 0.35, x[-1] + 0.35)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.04)

    handles = [legend_handles[m] for m in methods if m in legend_handles]
    labels = [legend_labels[m] for m in methods if m in legend_labels]
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 6),
            frameon=False,
            bbox_to_anchor=(0.5, 1.16),
            handlelength=2.3,
            columnspacing=1.1,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_bo_appendix_time_regret(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    benchmark: str = "ackley_200",
    title: str | None = None,
) -> Path:
    """Create the appendix 1x2 BO figure.

    Panels are simple regret versus cumulative wall-clock time and
    run-level peak GPU memory versus method.
    """
    df = _clean(pd.read_csv(csv_path))
    return _plot_time_regret_and_memory(df, str(benchmark), Path(out_path), title=title)

def _plot_tera_m_sweep(
    df: pd.DataFrame,
    out_path: Path,
    *,
    benchmark: str | None = None,
    method: str = "tera",
    title: str | None = None,
) -> Path:
    """Create a 1x3 TERA m-sweep summary figure."""
    if "tera_m" not in df.columns:
        raise ValueError(
            "CSV is missing tera_m. Re-run the BO sweep with the patched "
            "run_bo_synthetic.py so results.csv records the --tera-m value."
        )
    required = {
        "benchmark",
        "method",
        "seed",
        "eval_index",
        "regret",
        "iter_peak_memory_gb",
    }
    if "iter_wall_time_sec" not in df.columns and "cum_wall_time_sec" not in df.columns:
        raise ValueError(
            "CSV must contain either iter_wall_time_sec or cum_wall_time_sec "
            "for m-sweep wall-clock plotting."
        )
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns for m-sweep plotting: {missing}")

    sub = df[df["method"].astype(str) == str(method)].copy()
    if benchmark is not None:
        sub = sub[sub["benchmark"].astype(str) == str(benchmark)].copy()
    else:
        benchmarks = list(dict.fromkeys(sub["benchmark"].astype(str)))
        if len(benchmarks) != 1:
            raise ValueError("CSV contains multiple benchmarks. Pass --benchmark for the m-sweep plot.")
        benchmark = benchmarks[0]

    sub = sub.dropna(subset=["tera_m", "eval_index"])
    if sub.empty:
        raise ValueError("No rows remain after filtering by benchmark and method for the m-sweep plot.")
    sub["tera_m"] = pd.to_numeric(sub["tera_m"], errors="coerce")
    sub = sub.dropna(subset=["tera_m"])
    sub["tera_m"] = sub["tera_m"].astype(int)

    run_keys = ["benchmark", "method", "seed", "tera_m"]
    final_regret = (
        sub.dropna(subset=["regret"])
        .sort_values(run_keys + ["eval_index"])
        .groupby(run_keys, as_index=False)
        .tail(1)
        .loc[:, run_keys + ["regret"]]
    )

    if "iter_wall_time_sec" in sub.columns:
        run_time = (
            sub.dropna(subset=["iter_wall_time_sec"])
            .groupby(run_keys, as_index=False)["iter_wall_time_sec"]
            .sum()
            .rename(columns={"iter_wall_time_sec": "wall_time_sec"})
        )
    else:
        run_time = (
            sub.dropna(subset=["cum_wall_time_sec"])
            .sort_values(run_keys + ["eval_index"])
            .groupby(run_keys, as_index=False)
            .tail(1)
            .loc[:, run_keys + ["cum_wall_time_sec"]]
            .rename(columns={"cum_wall_time_sec": "wall_time_sec"})
        )

        run_time["wall_time_sec"] = run_time["wall_time_sec"] / 5.0

    final = final_regret.merge(run_time, on=run_keys, how="inner")
    peak = (
        sub.dropna(subset=["iter_peak_memory_gb"])
        .groupby(run_keys, as_index=False)["iter_peak_memory_gb"]
        .max()
        .rename(columns={"iter_peak_memory_gb": "peak_mem_gb"})
    )
    summary = final.merge(peak, on=run_keys, how="inner")
    if summary.empty:
        raise ValueError("Could not build run-level m-sweep summary from the CSV.")

    def _agg(value_col: str) -> pd.DataFrame:
        out = (
            summary.groupby("tera_m", as_index=False)[value_col]
            .agg(mean="mean", sem="sem")
            .sort_values("tera_m")
        )
        out["sem"] = out["sem"].fillna(0.0)
        return out

    panels = [
        (_agg("regret"), "Regret"),
        (_agg("wall_time_sec"), "Wall-clock time (s)"),
        (_agg("peak_mem_gb"), "Peak GPU memory (GB)"),
    ]

    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 2.9), constrained_layout=True)

    color = METHOD_COLORS.get(method, "#D55E00")
    xticks = sorted(summary["tera_m"].unique())
    for ax, (data, ylabel) in zip(axes, panels):
        ax.errorbar(
            data["tera_m"].to_numpy(dtype=float),
            data["mean"].to_numpy(dtype=float),
            yerr=data["sem"].to_numpy(dtype=float),
            color=color,
            marker="o",
            linestyle="-",
            linewidth=3.0,
            markersize=8.0,
            markeredgewidth=0.8,
            capsize=3,
            elinewidth=1.0,
            zorder=3,
        )
        ax.set_xlabel("m")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xticks)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.05)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_tera_m_sweep(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    benchmark: str | None = None,
    method: str = "tera",
    title: str | None = None,
) -> Path:
    """Create the BO TERA conditioning-size m-sweep figure."""
    df = _clean(pd.read_csv(csv_path))
    return _plot_tera_m_sweep(df, Path(out_path), benchmark=benchmark, method=method, title=title)
