from __future__ import annotations

from pathlib import Path as path_type
from typing import Iterable as iterable_type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tera.experiments.md22.io import read_results

method_order = [
    'exact gp',
    'ddsvgp',
    'dsoftki',
    'tera',
]

method_labels = {
    'exact gp': 'Exact GP',
    'ddsvgp': 'DDSVGP',
    'dsoftki': 'DSoftKI',
    'tera': 'TERA',
}

markers = {
    'exact gp': "o",
    'ddsvgp': "^",
    'dsoftki': "v",
    'tera': "*",
}

linestyles = {
    'exact gp': "-",
    'ddsvgp': ":",
    'dsoftki': "-.",
    'tera': "-",
}

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "status" in out.columns:
        out = out[out["status"].astype(str).isin(["ok", "external_csv"])]
    for col in ["normalized_energy_rmse_per_atom", "raw_energy_rmse_per_atom", "wall_time_sec", "peak_mem_gb"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["raw_energy_rmse_per_atom"])

def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "method"]
    vals = ["normalized_energy_rmse_per_atom", "raw_energy_rmse_per_atom", "wall_time_sec", "peak_mem_gb"]
    agg = df.groupby(keys, as_index=False)[vals].agg(["mean", "std", "count"])
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.to_flat_index()]
    agg = agg.rename(columns={"dataset_": "dataset", "method_": "method"})
    return agg

def _ordered(values: iterable_type[str]) -> list[str]:
    existing = list(dict.fromkeys(values))
    return [m for m in method_order if m in existing] + [m for m in existing if m not in method_order]

method_colors = {
    "exact gp": "#1F77B4",           # blue
    "ddsvgp": "#2CA02C", # green
    "dsoftki": "#FF7F0E",        # orange
    "tera": "#D62728",               # red
}

dataset_order = [
    "DHA",
    "AT-AT",
    "stachyose",
    "AT-AT-CG-CG",
    "buckyball-catcher",
    "double-walled-nanotube",
]

dataset_labels = {
    "DHA": "DHA", # \n$d=168$",
    "AT-AT": "AT-AT", # \n$d=180$",
    "stachyose": "Stach", # \n$d=261$",
    "AT-AT-CG-CG": "AT-CG", # \n$d=354$",
    "buckyball-catcher": "Bucky", # \n$d=444$",
    "double-walled-nanotube": "DWNT", # \n$d=1110$",
}

def _ordered_datasets(values: iterable_type[str]) -> list[str]:
    existing = [str(v) for v in dict.fromkeys(values)]
    return [d for d in dataset_order if d in existing] + [d for d in existing if d not in dataset_order]

def _dataset_label(dataset: str) -> str:
    return dataset_labels.get(str(dataset), str(dataset))


def plot_md22_results(csv_path: str | path_type, out_path: str | path_type, *, title: str | None = None) -> None:
    csv_path = path_type(csv_path)
    out_path = path_type(out_path)
    df = _clean(read_results(csv_path))
    if df.empty:
        raise ValueError(f"No successful rows found in {csv_path}")

    agg = _aggregate(df)
    datasets = _ordered_datasets(df["dataset"].astype(str))
    methods = _ordered(df["method"].astype(str))
    x = np.arange(len(datasets), dtype=float)

    panels = [
        ("raw_energy_rmse_per_atom", "Test RMSE / atom"),
        ("wall_time_sec", "Wall-clock time (s)"),
        ("peak_mem_gb", "Peak GPU memory (GB)"),
    ]

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

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 2.4), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=10)

    for ax, (metric, ylabel) in zip(axes, panels):
        for method in methods:
            sub = agg[agg["method"].astype(str) == method]

            means: list[float] = []
            errs: list[float] = []

            for dataset in datasets:
                row = sub[sub["dataset"].astype(str) == dataset]
                if row.empty:
                    means.append(np.nan)
                    errs.append(np.nan)
                    continue

                means.append(float(row[f"{metric}_mean"].iloc[0]))
                std_col = f"{metric}_std"
                errs.append(float(row[std_col].iloc[0]) if std_col in row else np.nan)

            y = np.asarray(means, dtype=float)
            yerr = np.asarray(errs, dtype=float)
            yerr[~np.isfinite(yerr)] = 0.0

            is_tera = method == "tera"
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=markers.get(method, "o"),
                linestyle="none",
                color=method_colors.get(method, None),
                capsize=5.0,
                elinewidth=1.0 if is_tera else 0.8,
                linewidth=0.0,
                markersize=14 if is_tera else 10,
                markeredgewidth=1.0,
                # markerfacecolor=method_colors.get(method) if is_tera else "none",
                # markeredgecolor=method_colors.get(method),
                alpha=1.0,
                label=method_labels.get(method, method),
                zorder=4 if is_tera else 3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([_dataset_label(d) for d in datasets], rotation=0, ha="center")
        ax.set_ylabel(ylabel)
        fig.supxlabel("Benchmark", y=-0.1)
        # ax.grid(True, axis="y", linewidth=0.45, alpha=0.28)
        # ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.04)

        if metric in {"wall_time_sec", "peak_mem_gb"}:
            positive = df[metric][df[metric] > 0]
            if len(positive) > 0:
                ax.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 1.15),
        handlelength=2.3,
        columnspacing=1.1,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_md22_training_curves(csv_path: str | path_type, out_path: str | path_type, *, title: str | None = None) -> None:
    csv_path = path_type(csv_path)
    out_path = path_type(out_path)
    df = read_results(csv_path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).isin(["ok", "external_csv"])]
    for col in ["step", "raw_energy_rmse_per_atom"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["dataset", "method", "step", "raw_energy_rmse_per_atom"])
    if df.empty:
        raise ValueError(f"No training-curve rows found in {csv_path}")

    datasets = list(dict.fromkeys(df["dataset"].astype(str)))
    methods = _ordered(df["method"].astype(str))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.1 * len(datasets), 3.2), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    if title:
        fig.suptitle(title, fontsize=11)

    for ax, dataset in zip(axes, datasets):
        sub_ds = df[df["dataset"].astype(str) == dataset].copy()
        for method in methods:
            sub = sub_ds[sub_ds["method"].astype(str) == method].copy()
            if sub.empty:
                continue
            grouped = (
                sub.groupby("step")["raw_energy_rmse_per_atom"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            grouped.columns = ["step", "mean", "std", "count"]
            x = grouped["step"].to_numpy(dtype=float)
            y = grouped["mean"].to_numpy(dtype=float)
            ystd = grouped["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker=markers.get(method, "o"),
                linestyle=linestyles.get(method, "-"),
                linewidth=1.4,
                markersize=4.0 if method != 'tera' else 6.0,
                label=method_labels.get(method, method),
            )
            if np.any(ystd > 0):
                ax.fill_between(x, y - ystd, y + ystd, alpha=0.15)
        ax.set_title(dataset)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Raw energy RMSE per atom")
        ax.grid(True, linewidth=0.4, alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False, bbox_to_anchor=(0.5, 1.08))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
