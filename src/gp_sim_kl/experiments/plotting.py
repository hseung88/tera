from __future__ import annotations
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

DRAW_ORDER = [
    "Exact GP",
    "Exact dGP-deRoos",
    "Exact dGP",
    "Vecchia GP",
    "Vecchia dGP",
    "Vecchia dGP-deRoos",
    "TERA",
]

LEGEND_METHOD_ORDER = [
    "Exact GP",
    "Exact dGP",
    "Exact dGP-deRoos",
    "Vecchia GP",
    "Vecchia dGP",
    "Vecchia dGP-deRoos",
    "TERA",
]

METHOD_COLORS = {
    "Exact GP": "#1F77B4",
    "Exact dGP": "#FF7F0E",
    "Exact dGP-deRoos": "#8C564B",
    "Vecchia dGP": "#9467BD",
    "Vecchia dGP-deRoos": "#2CA02C",
    "TERA": "#D62728",
}

METHOD_MARKERS = {
    "Exact GP": "s",
    "Exact dGP": "o",
    "Exact dGP-deRoos": "D",
    "Vecchia GP": "P",
    "Vecchia dGP": "^",
    "Vecchia dGP-deRoos": "v",
    "TERA": "*",
}

METHOD_LINESTYLES = {
    "Exact GP": "-",
    "Exact dGP": "-",
    "Exact dGP-deRoos": "-",
    "Vecchia GP": "-",
    "Vecchia dGP": "-",
    "Vecchia dGP-deRoos": "-",
    "TERA": "-",
}

METHOD_ZORDER = {
    "Exact GP": 2,
    "Exact dGP-deRoos": 3,
    "Exact dGP": 4,
    "Vecchia GP": 4,
    "Vecchia dGP": 5,
    "Vecchia dGP-deRoos": 6,
    "TERA": 7,
}

METHOD_LINEWIDTH = {
    "Exact GP": 4.0,
    "Exact dGP-deRoos": 4.0,
    "Exact dGP": 4.0,
    "Vecchia dGP": 4.0,
    "Vecchia dGP-deRoos": 4.0,
    "TERA": 4.0,
}

METHOD_ALPHA = {
    "Exact GP": 1.0,
    "Exact dGP-deRoos": 1.0,
    "Exact dGP": 1.0,
    "Vecchia GP": 1.0,
    "Vecchia dGP": 1.0,
    "Vecchia dGP-deRoos": 1.0,
    "TERA": 1.0,
}

METHOD_MARKERSIZE = {
    "Exact GP": 13.0,
    "Exact dGP-deRoos": 18.0,
    "Exact dGP": 15.0,
    "Vecchia dGP": 19.0,
    "Vecchia dGP-deRoos": 19.0,
    "TERA": 15.0,
}

HOLLOW_MARKERS = {
    "Exact dGP-deRoos",
    "Vecchia dGP",
    "Vecchia dGP-deRoos",
}

FIDELITY_METHODS = [
    "Exact GP",
    "Exact dGP-deRoos",
    "Exact dGP",
    "Vecchia GP",
    "Vecchia dGP",
    "Vecchia dGP-deRoos",
    "TERA",
]

SCALING_METHODS = [
    "Exact GP",
    "Exact dGP-deRoos",
    "Exact dGP",
    "Vecchia GP",
    "Vecchia dGP",
    "Vecchia dGP-deRoos",
    "TERA",
]

def _method_style(method: str) -> dict:
    color = METHOD_COLORS.get(method, "black")
    return {
        "color": color,
        "marker": METHOD_MARKERS.get(method, "o"),
        "linestyle": METHOD_LINESTYLES.get(method, "-"),
        "linewidth": METHOD_LINEWIDTH.get(method, 2.0),
        "markersize": METHOD_MARKERSIZE.get(method, 6.0),
        "markeredgewidth": 1.45,
        "markeredgecolor": color,
        "markerfacecolor": "white" if method in HOLLOW_MARKERS else color,
        "alpha": METHOD_ALPHA.get(method, 1.0),
        "zorder": METHOD_ZORDER.get(method, 3),
    }

def _errorbar_style(method: str) -> dict:
    if method in {"TERA", "Exact dGP", "Exact dGP-deRoos", "Vecchia dGP", "Vecchia dGP-deRoos"}:
        return {"elinewidth": 3.6, "capsize": 8.4, "capthick": 3.4}
    return {"elinewidth": 2.9, "capsize": 7.0, "capthick": 2.9}

def _legend_handle(method: str) -> Line2D:
    color = METHOD_COLORS.get(method, "black")
    return Line2D(
        [0],
        [0],
        color=color,
        marker=METHOD_MARKERS.get(method, "o"),
        linestyle="-",
        linewidth=METHOD_LINEWIDTH.get(method, 2.0),
        markersize=METHOD_MARKERSIZE.get(method, 6.0),
        markeredgewidth=1.8,
        markeredgecolor=color,
        markerfacecolor="white" if method in HOLLOW_MARKERS else color,
        alpha=1.0,
        label=method,
    )

def _apply_mpl_style(plt) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["legend.fontsize"] = 22
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18

def _normalize_method_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()

def _infer_sweep_column(df: pd.DataFrame) -> str:
    if "m" in df.columns and df["m"].nunique() > 1 and df["d"].nunique() <= 1:
        return "m"
    if "n_train" in df.columns and df["n_train"].nunique() > 1 and df["d"].nunique() <= 1:
        return "n_train"
    return "d"

def _sem_or_zero(series: pd.Series) -> float:
    val = series.sem()
    return 0.0 if pd.isna(val) else float(val)

def _x_label(sweep_col: str) -> str:
    if sweep_col == "n_train":
        return "n"
    return sweep_col

def _format_values_for_name(values, prefix: str, max_list: int = 8) -> str:
    vals = pd.Series(values).dropna().unique().tolist()
    vals = sorted(vals)

    def fmt(v) -> str:
        try:
            fv = float(v)
            if fv.is_integer():
                return str(int(fv))
            return f"{fv:g}"
        except Exception:
            return str(v)

    if len(vals) == 0:
        return f"{prefix}NA"
    if len(vals) == 1:
        return f"{prefix}{fmt(vals[0])}"
    if len(vals) <= max_list:
        return f"{prefix}{'-'.join(fmt(v) for v in vals)}"
    return f"{prefix}{fmt(vals[0])}-{fmt(vals[-1])}_k{len(vals)}"

def _plot_filename(df: pd.DataFrame, sweep_col: str) -> str:
    if "kernel" in df.columns and df["kernel"].nunique() == 1:
        kernel_tag = str(df["kernel"].iloc[0])
    else:
        kernel_tag = "mixedkernel"

    n_tag = _format_values_for_name(df["n_train"], "n")
    d_tag = _format_values_for_name(df["d"], "d")
    m_tag = _format_values_for_name(df["m"], "m")
    return f"{sweep_col}_sweep_{kernel_tag}_{n_tag}_{d_tag}_{m_tag}.png"

def _read_ok_rows(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    return _normalize_method_names(df)

def _aggregate(df: pd.DataFrame, sweep_col: str) -> pd.DataFrame:
    df = df.copy()
    df["wall_clock_time_sec"] = df["build_time_sec"] + df["predict_time_sec"]
    return df.groupby([sweep_col, "method"], as_index=False).agg(
        avg_marginal_kl_mean=("avg_marginal_kl", "mean"),
        avg_marginal_kl_sem=("avg_marginal_kl", _sem_or_zero),
        wall_clock_time_sec_mean=("wall_clock_time_sec", "mean"),
        wall_clock_time_sec_sem=("wall_clock_time_sec", _sem_or_zero),
        peak_mem_gb_mean=("peak_mem_gb", "mean"),
        peak_mem_gb_sem=("peak_mem_gb", _sem_or_zero),
    )

def _ordered_present_methods(df: pd.DataFrame, preferred: list[str] | None = None) -> list[str]:
    present_methods = set(df["method"].dropna().unique())
    if preferred is None:
        preferred = list(DRAW_ORDER)
    ordered = [m for m in preferred if m in present_methods]
    ordered.extend([m for m in sorted(present_methods) if m not in ordered])
    return ordered

def _plot_series(
    ax,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    yerr_col: str | None,
    method: str,
    ylabel: str,
) -> None:
    if sub.empty:
        return
    yerr = sub[yerr_col] if yerr_col is not None and yerr_col in sub.columns else None
    ax.errorbar(
        sub[x_col],
        sub[y_col],
        yerr=yerr,
        label=method,
        **_method_style(method),
        **_errorbar_style(method),
    )
    ax.set_xlabel(_x_label(x_col))
    ax.set_ylabel(ylabel)

def _collect_legend(fig, axes, preferred=LEGEND_METHOD_ORDER, ncol: int = 6, y: float = 0.995) -> None:
    present = set()
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        present.update(labels)

    ordered = [m for m in preferred if m in present]
    handles = [_legend_handle(m) for m in ordered]

    fig.legend(
        handles=handles,
        labels=ordered,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        columnspacing=1.2,
        handlelength=3.0,
        handleheight=1.4,
        handletextpad=0.7,
        borderpad=0.3,
        labelspacing=0.8,
    )

def plot_sweep(csv_path: str | Path, outdir: str | Path, sweep_col: str | None = None) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _read_ok_rows(csv_path)
    if sweep_col is None:
        sweep_col = _infer_sweep_column(df)
    grp = _aggregate(df, sweep_col)

    _apply_mpl_style(plt)
    panel_width = 8.0
    panel_height = panel_width * 0.6789
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.0 * panel_width, panel_height))

    specs = [
        (ax1, "avg_marginal_kl_mean", "avg_marginal_kl_sem", "Marginal KL"),
        (ax2, "wall_clock_time_sec_mean", "wall_clock_time_sec_sem", "Wall-clock time (s)"),
        (ax3, "peak_mem_gb_mean", "peak_mem_gb_sem", "Peak GPU memory (GB)"),
    ]

    ordered_methods = _ordered_present_methods(grp)
    for method in ordered_methods:
        sub = grp[grp["method"] == method].sort_values(sweep_col)
        for ax, col, ecol, ylabel in specs:
            _plot_series(ax, sub, sweep_col, col, ecol, method, ylabel)

    _collect_legend(fig, (ax1, ax2, ax3), preferred=ordered_methods)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(outdir / _plot_filename(df, sweep_col), dpi=300, bbox_inches="tight")
    plt.close(fig)

def _main_filename(df_m: pd.DataFrame, df_d: pd.DataFrame, df_scaling: pd.DataFrame) -> str:
    kernel_tag = "mixedkernel"
    kernels = []
    for df in [df_m, df_d, df_scaling]:
        if "kernel" in df.columns and df["kernel"].nunique() == 1:
            kernels.append(str(df["kernel"].iloc[0]))
    kernels = sorted(set(kernels))
    if len(kernels) == 1:
        kernel_tag = kernels[0]
    elif len(kernels) > 1:
        kernel_tag = "mixedkernel"

    n_tag = _format_values_for_name(pd.concat([df_m["n_train"], df_d["n_train"], df_scaling["n_train"]]), "n")
    d_tag = _format_values_for_name(pd.concat([df_m["d"], df_d["d"], df_scaling["d"]]), "d")
    m_tag = _format_values_for_name(pd.concat([df_m["m"], df_d["m"], df_scaling["m"]]), "m")
    return f"main_panel_{kernel_tag}_{n_tag}_{d_tag}_{m_tag}.png"

def plot_main_four_panel(
    csv_m_sweep: str | Path,
    csv_d_fidelity: str | Path,
    csv_d_scaling: str | Path,
    outdir: str | Path,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_m = _read_ok_rows(csv_m_sweep)
    df_d = _read_ok_rows(csv_d_fidelity)
    df_s = _read_ok_rows(csv_d_scaling)

    grp_m = _aggregate(df_m, "m")
    grp_d = _aggregate(df_d, "d")
    grp_s = _aggregate(df_s, "d")

    _apply_mpl_style(plt)
    panel_width = 6.4
    panel_height = panel_width * 0.72

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(2.0 * panel_width, 2.0 * panel_height),
        constrained_layout=False,
    )

    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    fidelity_methods = _ordered_present_methods(grp_m, preferred=FIDELITY_METHODS)
    scaling_methods = _ordered_present_methods(grp_s, preferred=SCALING_METHODS)

    for method in fidelity_methods:
        sub_m = grp_m[grp_m["method"] == method].sort_values("m")
        if not sub_m.empty:
            _plot_series(ax1, sub_m, "m", "avg_marginal_kl_mean", "avg_marginal_kl_sem", method, "Marginal KL")
        sub_d = grp_d[grp_d["method"] == method].sort_values("d")
        if not sub_d.empty:
            _plot_series(ax2, sub_d, "d", "avg_marginal_kl_mean", "avg_marginal_kl_sem", method, "Marginal KL")

    for method in scaling_methods:
        sub_s = grp_s[grp_s["method"] == method].sort_values("d")
        if sub_s.empty:
            continue
        _plot_series(ax3, sub_s, "d", "wall_clock_time_sec_mean", "wall_clock_time_sec_sem", method, "Wall-clock time (s)")
        _plot_series(ax4, sub_s, "d", "peak_mem_gb_mean", "peak_mem_gb_sem", method, "Peak GPU memory (GB)")

    _collect_legend(fig, [ax1, ax2, ax3, ax4], preferred=LEGEND_METHOD_ORDER)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(outdir / _main_filename(df_m, df_d, df_s), dpi=300, bbox_inches="tight")
    plt.close(fig)

def _select_and_aggregate(
    csv_path: str | Path,
    sweep_col: str,
    methods: list[str],
    fixed_filters: dict[str, int | float] | None = None,
) -> pd.DataFrame:
    df = _read_ok_rows(csv_path)
    if fixed_filters:
        for col, val in fixed_filters.items():
            if col in df.columns:
                df = df[df[col] == val].copy()
    grp = _aggregate(df, sweep_col)
    grp = grp[grp["method"].isin(methods)].copy()
    return grp.sort_values(["method", sweep_col])

def _plot_panel(ax, grp: pd.DataFrame, sweep_col: str, y_col: str, yerr_col: str, ylabel: str, methods: list[str]) -> None:
    for method in methods:
        sub = grp[grp["method"] == method].sort_values(sweep_col)
        if sub.empty:
            continue
        _plot_series(ax, sub, sweep_col, y_col, yerr_col, method, ylabel)

def _plot_xy_series(
    ax,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    method: str,
    xlabel: str,
    ylabel: str,
    xerr_col: str | None = None,
    yerr_col: str | None = None,
) -> None:
    if sub.empty:
        return
    xerr = sub[xerr_col] if xerr_col is not None and xerr_col in sub.columns else None
    yerr = sub[yerr_col] if yerr_col is not None and yerr_col in sub.columns else None
    ax.errorbar(
        sub[x_col],
        sub[y_col],
        xerr=xerr,
        yerr=yerr,
        label=method,
        **_method_style(method),
        **_errorbar_style(method),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def _plot_metric_panel(
    ax,
    grp: pd.DataFrame,
    methods: list[str],
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    xerr_col: str | None = None,
    yerr_col: str | None = None,
) -> None:
    for method in methods:
        sub = grp[grp["method"] == method].sort_values(x_col)
        if sub.empty:
            continue
        _plot_xy_series(
            ax,
            sub,
            x_col=x_col,
            y_col=y_col,
            method=method,
            xlabel=xlabel,
            ylabel=ylabel,
            xerr_col=xerr_col,
            yerr_col=yerr_col,
        )

def _infer_target_correlation_from_path(path: str | Path) -> float | None:
    text = str(path)
    match = re.search(r"C([0-9]+p[0-9]+|[0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))

def plot_target_correlation_d_sweep(
    csv_paths: list[str | Path],
    outdir: str | Path,
    correlations: list[float] | None = None,
    methods: list[str] | None = None,
) -> None:
    """Plot Marginal KL vs d for multiple target median correlation settings.

    Each CSV should correspond to one d-sweep run produced with a different
    --target-median-correlation value. If correlations are not provided, the
    function infers them from path fragments such as C0p1, C0p3, C0p5, C0p7.
    """
    if not csv_paths:
        raise ValueError("At least one CSV path is required.")
    if correlations is not None and len(correlations) != len(csv_paths):
        raise ValueError("The number of correlations must match the number of CSV paths.")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = ["Exact GP", "Exact dGP-deRoos", "TERA"]

    panels: list[tuple[float, pd.DataFrame]] = []
    for idx, path in enumerate(csv_paths):
        corr = correlations[idx] if correlations is not None else _infer_target_correlation_from_path(path)
        if corr is None:
            raise ValueError(
                f"Could not infer target correlation from {path}. "
                "Pass --target-correlations explicitly."
            )
        df = _read_ok_rows(path)
        grp = _aggregate(df, "d")
        grp = grp[grp["method"].isin(methods)].copy()
        panels.append((float(corr), grp))

    panels.sort(key=lambda item: item[0])

    _apply_mpl_style(plt)
    panel_width = 5.2
    panel_height = 5.2
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(panel_width * len(panels), panel_height),
        sharey=True,
        constrained_layout=False,
    )
    if len(panels) == 1:
        axes = [axes]

    ordered_methods = [m for m in LEGEND_METHOD_ORDER if m in methods]
    ordered_methods.extend([m for m in methods if m not in ordered_methods])

    for ax, (corr, grp) in zip(axes, panels):
        _plot_metric_panel(
            ax,
            grp,
            ordered_methods,
            x_col="d",
            y_col="avg_marginal_kl_mean",
            xlabel="d",
            ylabel="Marginal KL",
            yerr_col="avg_marginal_kl_sem",
        )
        ax.set_title(rf"$\rho={corr:g}$")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[1:]:
        ax.set_ylabel("")

    _collect_legend(fig, axes, preferred=ordered_methods, ncol=len(ordered_methods), y=1.02)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(outdir / "target_correlation_d_sweep.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "target_correlation_d_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_combined_summary_figure(
    csv_m_sweep: str | Path,
    csv_d_scaling: str | Path,
    csv_n_sweep: str | Path,
    outdir: str | Path,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _apply_mpl_style(plt)

    grp_m = _aggregate(_read_ok_rows(csv_m_sweep), "m")
    grp_d = _aggregate(_read_ok_rows(csv_d_scaling), "d")
    grp_n = _aggregate(_read_ok_rows(csv_n_sweep), "n_train")

    methods_m = _ordered_present_methods(grp_m)
    methods_d = _ordered_present_methods(grp_d)
    methods_n = _ordered_present_methods(grp_n)

    fig, axes = plt.subplots(3, 3, figsize=(19.5, 14.0), constrained_layout=False)

    _plot_metric_panel(
        axes[0, 0], grp_m, methods_m,
        x_col="m", y_col="avg_marginal_kl_mean",
        xlabel="m", ylabel="Marginal KL",
        yerr_col="avg_marginal_kl_sem",
    )
    _plot_metric_panel(
        axes[0, 1], grp_m, methods_m,
        x_col="wall_clock_time_sec_mean", y_col="avg_marginal_kl_mean",
        xlabel="Wall-clock time (s)", ylabel="Marginal KL",
        xerr_col="wall_clock_time_sec_sem", yerr_col="avg_marginal_kl_sem",
    )
    _plot_metric_panel(
        axes[0, 2], grp_m, methods_m,
        x_col="peak_mem_gb_mean", y_col="avg_marginal_kl_mean",
        xlabel="Peak GPU memory (GB)", ylabel="Marginal KL",
        xerr_col="peak_mem_gb_sem", yerr_col="avg_marginal_kl_sem",
    )

    _plot_metric_panel(
        axes[1, 0], grp_d, methods_d,
        x_col="d", y_col="avg_marginal_kl_mean",
        xlabel="d", ylabel="Marginal KL",
        yerr_col="avg_marginal_kl_sem",
    )
    _plot_metric_panel(
        axes[1, 1], grp_d, methods_d,
        x_col="d", y_col="wall_clock_time_sec_mean",
        xlabel="d", ylabel="Wall-clock time (s)",
        yerr_col="wall_clock_time_sec_sem",
    )
    _plot_metric_panel(
        axes[1, 2], grp_d, methods_d,
        x_col="d", y_col="peak_mem_gb_mean",
        xlabel="d", ylabel="Peak GPU memory (GB)",
        yerr_col="peak_mem_gb_sem",
    )

    _plot_metric_panel(
        axes[2, 0], grp_n, methods_n,
        x_col="n_train", y_col="avg_marginal_kl_mean",
        xlabel="n", ylabel="Marginal KL",
        yerr_col="avg_marginal_kl_sem",
    )
    _plot_metric_panel(
        axes[2, 1], grp_n, methods_n,
        x_col="n_train", y_col="wall_clock_time_sec_mean",
        xlabel="n", ylabel="Wall-clock time (s)",
        yerr_col="wall_clock_time_sec_sem",
    )
    _plot_metric_panel(
        axes[2, 2], grp_n, methods_n,
        x_col="n_train", y_col="peak_mem_gb_mean",
        xlabel="n", ylabel="Peak GPU memory (GB)",
        yerr_col="peak_mem_gb_sem",
    )

    axes[0, 1].set_xscale("log")
    axes[0, 2].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 2].set_yscale("log")
    axes[2, 1].set_yscale("log")
    axes[2, 2].set_yscale("log")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _collect_legend(fig, axes.flatten(), preferred=LEGEND_METHOD_ORDER, ncol=3, y=1.03)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(outdir / "fidelity_scaling_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_fidelity_and_scaling_figures(
    csv_m_fidelity: str | Path,
    csv_d_fidelity: str | Path,
    csv_n_fidelity: str | Path,
    csv_m_scaling: str | Path,
    csv_d_scaling: str | Path,
    csv_n_scaling: str | Path,
    outdir: str | Path,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _apply_mpl_style(plt)

    fidelity_specs = [
        {
            "csv": csv_m_fidelity,
            "sweep_col": "m",
            "methods": ["Exact GP", "Exact dGP", "Exact dGP-deRoos", "Vecchia dGP", "Vecchia dGP-deRoos", "TERA"],
            "xlabel": "m",
        },
        {
            "csv": csv_d_fidelity,
            "sweep_col": "d",
            "methods": ["Exact GP", "Exact dGP", "Exact dGP-deRoos", "Vecchia dGP", "Vecchia dGP-deRoos", "TERA"],
            "xlabel": "d",
        },
        {
            "csv": csv_n_fidelity,
            "sweep_col": "n_train",
            "methods": ["Exact GP", "Exact dGP", "Exact dGP-deRoos", "Vecchia dGP", "Vecchia dGP-deRoos", "TERA"],
            "xlabel": "n",
        },
    ]

    scaling_specs = [
        {
            "csv": csv_m_scaling,
            "sweep_col": "m",
            "methods": ["Exact GP", "Vecchia dGP", "Vecchia dGP-deRoos", "TERA"],
            "y_col": "wall_clock_time_sec_mean",
            "yerr_col": "wall_clock_time_sec_sem",
            "ylabel": "Wall-clock time (s)",
        },
        {
            "csv": csv_d_scaling,
            "sweep_col": "d",
            "methods": ["Exact GP", "Exact dGP", "Vecchia dGP", "Vecchia dGP-deRoos", "TERA"],
            "y_col": "wall_clock_time_sec_mean",
            "yerr_col": "wall_clock_time_sec_sem",
            "ylabel": "Wall-clock time (s)",
        },
        {
            "csv": csv_n_scaling,
            "sweep_col": "n_train",
            "methods": ["Exact GP", "Exact dGP-deRoos", "Vecchia dGP", "Vecchia dGP-deRoos", "TERA"],
            "y_col": "wall_clock_time_sec_mean",
            "yerr_col": "wall_clock_time_sec_sem",
            "ylabel": "Wall-clock time (s)",
        },
        {
            "csv": csv_m_scaling,
            "sweep_col": "m",
            "methods": ["Exact GP", "Vecchia dGP", "Vecchia dGP-deRoos", "TERA"],
            "y_col": "peak_mem_gb_mean",
            "yerr_col": "peak_mem_gb_sem",
            "ylabel": "Peak GPU memory (GB)",
        },
        {
            "csv": csv_d_scaling,
            "sweep_col": "d",
            "methods": ["Exact GP", "Exact dGP", "Vecchia dGP", "Vecchia dGP-deRoos", "TERA"],
            "y_col": "peak_mem_gb_mean",
            "yerr_col": "peak_mem_gb_sem",
            "ylabel": "Peak GPU memory (GB)",
        },
        {
            "csv": csv_n_scaling,
            "sweep_col": "n_train",
            "methods": ["Exact GP", "Exact dGP-deRoos", "Vecchia dGP", "Vecchia dGP-deRoos", "TERA"],
            "y_col": "peak_mem_gb_mean",
            "yerr_col": "peak_mem_gb_sem",
            "ylabel": "Peak GPU memory (GB)",
        },
    ]

    fidelity_fig, fidelity_axes = plt.subplots(1, 3, figsize=(19.5, 5.6), constrained_layout=False)
    for ax, spec in zip(fidelity_axes, fidelity_specs):
        grp = _select_and_aggregate(spec["csv"], spec["sweep_col"], spec["methods"])
        _plot_panel(ax, grp, spec["sweep_col"], "avg_marginal_kl_mean", "avg_marginal_kl_sem", "Marginal KL", spec["methods"])
    _collect_legend(fidelity_fig, fidelity_axes, preferred=LEGEND_METHOD_ORDER, ncol=5, y=0.995)
    fidelity_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fidelity_fig.savefig(outdir / "fidelity_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fidelity_fig)

    scaling_fig, scaling_axes = plt.subplots(2, 3, figsize=(19.5, 10.2), constrained_layout=False)
    for ax, spec in zip(scaling_axes.flatten(), scaling_specs):
        grp = _select_and_aggregate(spec["csv"], spec["sweep_col"], spec["methods"])
        _plot_panel(ax, grp, spec["sweep_col"], spec["y_col"], spec["yerr_col"], spec["ylabel"], spec["methods"])
    _collect_legend(scaling_fig, scaling_axes.flatten(), preferred=LEGEND_METHOD_ORDER, ncol=5, y=0.995)
    scaling_fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    scaling_fig.savefig(outdir / "scaling_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(scaling_fig)
