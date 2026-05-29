from __future__ import annotations

import argparse

from gp_sim_kl.experiments.plotting import (
    plot_combined_summary_figure,
    plot_fidelity_and_scaling_figures,
    plot_main_four_panel,
    plot_sweep,
    plot_target_correlation_d_sweep,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--sweep-col", type=str, default=None)
    parser.add_argument("--csv-m-sweep", type=str, default=None)
    parser.add_argument("--csv-d-fidelity", type=str, default=None)
    parser.add_argument("--csv-d-scaling", type=str, default=None)
    parser.add_argument("--csv-n-fidelity", type=str, default=None)
    parser.add_argument("--csv-m-scaling", type=str, default=None)
    parser.add_argument("--csv-n-scaling", type=str, default=None)
    parser.add_argument("--csv-n-sweep", type=str, default=None)
    parser.add_argument(
        "--csv-target-corr-d-sweeps",
        nargs="+",
        default=None,
        help="Results CSVs for d-sweep target median correlation ablation.",
    )
    parser.add_argument(
        "--target-correlations",
        type=str,
        default=None,
        help="Comma-separated target median correlations matching --csv-target-corr-d-sweeps. "
        "If omitted, values are inferred from path tags such as C0p1.",
    )
    parser.add_argument(
        "--target-corr-methods",
        type=str,
        default="Exact GP,Exact dGP-deRoos,TERA",
        help="Comma-separated methods to show in the target correlation ablation figure.",
    )
    args = parser.parse_args()

    if args.csv_target_corr_d_sweeps:
        correlations = None
        if args.target_correlations:
            correlations = [float(v.strip()) for v in args.target_correlations.split(",") if v.strip()]
        methods = [m.strip() for m in args.target_corr_methods.split(",") if m.strip()]
        plot_target_correlation_d_sweep(
            csv_paths=args.csv_target_corr_d_sweeps,
            outdir=args.out,
            correlations=correlations,
            methods=methods,
        )
        print(f"Saved target-correlation d-sweep ablation figure to {args.out}")
        return

    have_combined_bundle = all([
        args.csv_m_sweep,
        args.csv_d_scaling,
        args.csv_n_sweep,
    ])

    if have_combined_bundle:
        plot_combined_summary_figure(
            csv_m_sweep=args.csv_m_sweep,
            csv_d_scaling=args.csv_d_scaling,
            csv_n_sweep=args.csv_n_sweep,
            outdir=args.out,
        )
        print(f"Saved combined summary figure to {args.out}")
        return

    have_new_bundle = all([
        args.csv_m_sweep,
        args.csv_d_fidelity,
        args.csv_n_fidelity,
        args.csv_m_scaling,
        args.csv_d_scaling,
        args.csv_n_scaling,
    ])
    if have_new_bundle:
        plot_fidelity_and_scaling_figures(
            csv_m_fidelity=args.csv_m_sweep,
            csv_d_fidelity=args.csv_d_fidelity,
            csv_n_fidelity=args.csv_n_fidelity,
            csv_m_scaling=args.csv_m_scaling,
            csv_d_scaling=args.csv_d_scaling,
            csv_n_scaling=args.csv_n_scaling,
            outdir=args.out,
        )
        print(f"Saved fidelity and scaling figures to {args.out}")
        return

    if args.csv_m_sweep or args.csv_d_fidelity or args.csv_d_scaling:
        if not (args.csv_m_sweep and args.csv_d_fidelity and args.csv_d_scaling):
            raise ValueError("For the four-panel figure, provide --csv-m-sweep, --csv-d-fidelity, and --csv-d-scaling.")
        plot_main_four_panel(args.csv_m_sweep, args.csv_d_fidelity, args.csv_d_scaling, args.out)
        print(f"Saved four-panel figure to {args.out}")
        return

    if args.csv is None:
        raise ValueError("Provide either --csv for a single sweep plot, the three CSV arguments for the requested triptych figures, the three CSV arguments for the four-panel figure, or the six CSV arguments for the fidelity/scaling figures.")
    plot_sweep(args.csv, args.out, sweep_col=args.sweep_col)
    print(f"Saved plots to {args.out}")

if __name__ == "__main__":
    main()
