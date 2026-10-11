
from __future__ import annotations

import argparse
from pathlib import Path as path_type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tera.experiments.md22.io import read_results

tera_red = "#D62728"
tera_red_light = "#D62728"

m_keep = [5, 10, 20, 30, 50]
noise_grid = [1e-1, 1e-2, 1e-3, 1e-4]
ntrain_grid = [500, 1000, 2000, 4000]

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "status" in out.columns:
        status = out["status"].astype("string")
        keep = status.str.lower().eq("ok")
        out = out[keep].copy()

    numeric_cols = [
        "m",
        "sigma_f_sweep",
        "sigma_g_sweep",
        "n_train_requested",
        "n_test_requested",
        "raw_energy_rmse_per_atom",
        "wall_time_sec",
        "peak_mem_gb",
        "seed",
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "m" in out.columns:
        out = out[out["m"].isin(m_keep)].copy()

    return out

def _fmt_cell(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    if x >= 1:
        return f"{x:.2f}"
    if x >= 0.1:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    if x >= 0.01:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.5f}".rstrip("0").rstrip(".")

def _style():
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

def plot_fig1(df: pd.DataFrame, outdir: path_type, errorbar="std") -> None:
    sub = df[(df["figure"] == "fig1") & (df["ablation"] == "m_sweep")].copy()
    sub = sub[sub["m"].isin(m_keep)].copy()
    if sub.empty:
        raise ValueError("No rows for Figure 1.")

    agg = (
        sub.groupby("m", as_index=False)
        .agg(
            rmse_mean=("raw_energy_rmse_per_atom", "mean"),
            rmse_std=("raw_energy_rmse_per_atom", errorbar),
            time_mean=("wall_time_sec", "mean"),
            time_std=("wall_time_sec", errorbar),
            mem_mean=("peak_mem_gb", "mean"),
            mem_std=("peak_mem_gb", errorbar),
        )
        .sort_values("m")
    )

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 2.9), constrained_layout=True)

    panels = [
        ("RMSE", "Test RMSE / atom", "rmse_mean", "rmse_std"),
        ("Wall-clock time", "Wall-clock time (s)", "time_mean", "time_std"),
        ("Peak GPU memory", "Peak GPU memory (GB)", "mem_mean", "mem_std"),
    ]

    x = agg["m"].to_numpy(dtype=float)

    for ax, (title, ylabel, mean_col, std_col) in zip(axes, panels):
        y = agg[mean_col].to_numpy(dtype=float)
        yerr = agg[std_col].fillna(0.0).to_numpy(dtype=float)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=tera_red,
            marker="o",
            linestyle="-",
            linewidth=3.0,
            markersize=8.0,
            markeredgewidth=0.8,
            capsize=3.0,
            elinewidth=1.0,
            zorder=3,
        )

        ax.set_xlabel(r"m")
        ax.set_ylabel(ylabel)
        ax.set_xticks(m_keep)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.06)

    fig.savefig(outdir / "md22_tera_ablation_m_sweep.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "md22_tera_ablation_m_sweep.pdf", bbox_inches="tight")
    plt.close(fig)

def _make_matrix(sub: pd.DataFrame, row_vals, col_vals, row_col: str, col_col: str, value_col: str):
    mat = np.full((len(row_vals), len(col_vals)), np.nan, dtype=float)
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            rows = sub[(sub[row_col] == rv) & (sub[col_col] == cv)]
            if not rows.empty:
                mat[i, j] = rows[value_col].mean()
    return mat

def _draw_heatmap(
    ax,
    mat,
    row_labels,
    col_labels,
    title,
    xlabel,
    ylabel,
    cbar_label,
    show_colorbar=True,
):

    im = ax.imshow(mat, aspect="equal", cmap="Blues")
    ax.set_box_aspect(mat.shape[0] / mat.shape[1])

    if title:
        ax.set_title(title, pad=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)

    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(
                    j,
                    i,
                    _fmt_cell(mat[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="black",
                )

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        if cbar_label:
            cbar.ax.set_ylabel(cbar_label, rotation=90)
        cbar.ax.tick_params(labelsize=10)

def plot_fig2(df: pd.DataFrame, outdir: path_type) -> None:
    _style()
    fig, axes = plt.subplots(1, 1, figsize=(4.4, 3.4), constrained_layout=True)

    sub1 = df[(df["figure"] == "fig2_panel1") & (df["ablation"] == "noise_sweep")].copy()
    sub1 = sub1[sub1["m"] == 20].copy()
    sub1 = sub1[
        sub1["sigma_f_sweep"].isin(noise_grid)
        & sub1["sigma_g_sweep"].isin(noise_grid)
    ].copy()
    if sub1.empty:
        raise ValueError("No rows for Figure 2 panel 1.")

    noise_x_order = [1e-4, 1e-3, 1e-2, 1e-1]

    mat1 = _make_matrix(
        sub1,
        noise_grid,
        noise_x_order,
        "sigma_f_sweep",
        "sigma_g_sweep",
        "raw_energy_rmse_per_atom",
    )

    row_labels1 = [r"$10^{-1}$", r"$10^{-2}$", r"$10^{-3}$", r"$10^{-4}$"]
    col_labels1 = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]

    _draw_heatmap(
        axes,
        mat1,
        row_labels1,
        col_labels1,
        "",
        r"$\sigma_g^2$",
        r"$\sigma_y^2$",
        "",
        show_colorbar=True,
    )

    fig.savefig(outdir / "md22_tera_ablation_heatmaps.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "md22_tera_ablation_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)

def plot(csv_path, outdir, *, phase="total", errorbar="std"):
    df = _clean(read_results(csv_path))
    if "ablation" not in df:
        df["ablation"] = df["sweep"].map({"m": "m_sweep", "noise": "noise_sweep"})
        df["figure"] = df["sweep"].map({"m": "fig1", "noise": "fig2_panel1"})
    if phase != "total":
        df["wall_time_sec"] = df["fit_time_sec" if phase == "train" else "predict_time_sec"]
        df["peak_mem_gb"] = df["fit_peak_mem_gb" if phase == "train" else "predict_peak_mem_gb"]
    for dataset, sub in df.groupby("dataset", sort=False):
        dest = path_type(outdir) / dataset if df.dataset.nunique() > 1 else path_type(outdir)
        dest.mkdir(parents=True, exist_ok=True)
        if sub.ablation.eq("m_sweep").any():
            plot_fig1(sub, dest, errorbar=errorbar)
        if sub.ablation.eq("noise_sweep").any():
            plot_fig2(sub, dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--phase", choices=["total", "train", "predict"], default="total")
    p.add_argument("--errorbar", choices=["std", "sem"], default="std")
    args = p.parse_args()
    plot(args.csv, args.outdir, phase=args.phase, errorbar=args.errorbar)


if __name__ == "__main__":
    main()
