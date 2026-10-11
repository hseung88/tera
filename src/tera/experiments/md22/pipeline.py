"""Run the MD22 comparison with the original tera output layout."""

import json
import shutil
from dataclasses import replace
from pathlib import Path as path_type

import pandas as pd

from .runner import build_parser, config_from_args, result_record, run


def run_pipeline(cfg, outdir):
    outdir = path_type(outdir)
    final = outdir / "internal_and_merged"
    rows = run(replace(cfg, export_fair_data=True, fair_data_dir=str(final / "fair_splits")), final)
    canonical = outdir / "canonical_fair_data"
    splits = canonical / "fair_splits"
    splits.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset, seed, split_id in dict.fromkeys((r.dataset, r.seed, r.split_id) for r in rows):
        name = f"{dataset}_seed{seed}_{split_id}"
        for suffix in (".npz", ".json"):
            shutil.copyfile(final / "fair_splits" / (name + suffix), splits / (name + suffix))
        meta = json.loads((splits / (name + ".json")).read_text())
        manifest.append({
            **{key: meta[key] for key in (
                "dataset", "seed", "split_id", "preprocessing_version", "n_train",
                "n_test", "d", "n_atoms", "x_scale",
            )},
            "fair_split_npz": str(splits / (name + ".npz")),
            "metadata_json": str(splits / (name + ".json")),
        })
    pd.DataFrame(manifest).to_csv(canonical / "fair_splits_manifest.csv", index=False)
    external = []
    for row in rows:
        if row.method not in {'dsoftki', 'ddsvgp'}:
            continue
        record = result_record(row)
        external.append({
            **{key: record[key] for key in (
                "dataset", "seed", "method", "split_id", "preprocessing_version",
                "n_train", "n_test", "d", "n_atoms", "x_scale",
            )},
            "external_original_force_scale": False,
            "external_joint_scale": cfg.baseline_joint_scale,
            "external_scale_name": "joint_value_derivative_train_scale" if cfg.baseline_joint_scale else "train_energy_std",
            "external_scale": row.model_target_scale,
            **{key: record[key] for key in (
                "normalized_energy_rmse_per_atom", "raw_energy_rmse_per_atom", "wall_time_sec",
                "fit_time_sec", "predict_time_sec", "peak_mem_gb",
            )},
            "status": "ok" if row.status == "ok" else "error",
            "error": "" if row.status == "ok" else row.status,
            "source_npz": str(splits / f"{row.dataset}_seed{row.seed}_{row.split_id}.npz"),
        })
    if external:
        pd.DataFrame(external).to_csv(outdir / "external_results.csv", index=False)
    history_path = final / "training_curves.csv"
    if history_path.exists() and cfg.log_training_curves:
        history = pd.read_csv(history_path)
        curves = []
        for record in external:
            selected = history[
                (history.dataset == record["dataset"])
                & (history.seed == record["seed"])
                & (history.method == record["method"])
                & (history.split_id == record["split_id"])
            ]
            for _, point in selected.iterrows():
                curves.append({
                    **{key: record[key] for key in (
                        "dataset", "seed", "method", "split_id", "preprocessing_version",
                        "n_train", "n_test", "d", "n_atoms", "x_scale",
                        "external_joint_scale", "external_scale_name", "external_scale",
                    )},
                    **{key: point[key] for key in (
                        "step", "normalized_energy_rmse_per_atom", "raw_energy_rmse_per_atom",
                    )},
                    "release_normalized_energy_rmse": point["raw_energy_rmse_per_atom"] * record["n_atoms"] / record["external_scale"],
                    "fit_time_sec": point.get("fit_time_sec", float("nan")),
                    "status": point["status"],
                    "source_npz": record["source_npz"],
                })
        if curves:
            pd.DataFrame(curves).to_csv(outdir / "external_training_curves.csv", index=False)
    return rows


def main():
    args = build_parser().parse_args()
    rows = run_pipeline(config_from_args(args), args.outdir)
    print(f"Wrote {len(rows)} rows to {path_type(args.outdir) / 'internal_and_merged' / 'results.csv'}")


if __name__ == "__main__":
    main()
