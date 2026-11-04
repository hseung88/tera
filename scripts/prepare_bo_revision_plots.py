from pathlib import Path

import pandas as pd


keys = ["benchmark", "method", "seed", "eval_index"]


def read_results(filename):
    df = pd.read_csv(filename)
    df["method"] = df["method"].str.strip().str.lower()
    if df.duplicated(keys).any() or not df["status"].isin(["init", "ok"]).all():
        raise ValueError(f"Duplicate or failed entries: {filename}")
    df["source_csv"] = str(filename)
    return df


def save_results(frames, directory):
    df = pd.concat(frames, ignore_index=True).sort_values(keys)
    if df.duplicated(keys).any():
        raise ValueError("Overlapping experiment runs")
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / "results.csv", index=False)
    print(directory / "results.csv")
    return df


def main():
    old = read_results("outputs/bo_main_combined/results.csv")
    tera = read_results("outputs/revision/bo_main/results.csv")
    tera = tera.loc[tera["method"].eq("tera")].copy()
    old_keys = pd.MultiIndex.from_frame(old.loc[old["method"].eq("tera"), keys])
    new_keys = pd.MultiIndex.from_frame(tera[keys])
    if len(old_keys.difference(new_keys)) or len(new_keys.difference(old_keys)):
        raise ValueError("Old and new TERA runs do not match")

    lbfgsb = pd.concat([
        read_results(f"outputs/bo_main_{name}_200_lbfgsb/results.csv")
        for name in ("ackley", "levy")
    ], ignore_index=True)
    vbo = pd.concat([
        read_results(f"outputs/bo_main_{name}200_vbo_300/results.csv")
        for name in ("ackley", "levy")
    ], ignore_index=True)

    save_results(
        [old.loc[~old["method"].isin(["tera", "lbfgsb"])], tera, lbfgsb],
        Path("outputs/0_plots/bo_main_rebuttal"),
    )
    directory = Path("outputs/0_plots/bo_rebuttal")
    df = save_results([tera, vbo, lbfgsb], directory)
    run_keys = ["benchmark", "method", "seed"]
    df["wall_time_sec"] = df[
        ["fit_time_sec", "acq_time_sec", "eval_time_sec"]
    ].sum(axis=1, min_count=3)
    final = df.groupby(run_keys).tail(1).set_index(run_keys)
    runs = final[["eval_index", "regret"]].rename(
        columns={"eval_index": "queried_locations"}
    )
    runs["wall_time_sec"] = df.groupby(run_keys)["wall_time_sec"].sum()
    runs["peak_gpu_memory_gb"] = df.groupby(run_keys)["iter_peak_memory_gb"].max()
    runs.to_csv(directory / "run_summary.csv")
    summary = runs.groupby(level=["benchmark", "method"]).agg(["mean", "sem"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(directory / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
