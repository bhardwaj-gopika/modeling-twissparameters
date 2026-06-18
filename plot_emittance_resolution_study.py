"""S-spacing resolution study for emittance/beam-size evolution.

Loads Impact-T extraction output (twiss-impact-output/twiss_vs_position.csv),
plots emittance and beam size vs z for one or more samples, and tests how
coarse an s-grid can be before peak values are missed by more than 5%.

Usage:
    python plot_emittance_resolution_study.py \
        --input twiss-impact-output/twiss_vs_position.csv \
        --output-dir resolution-study \
        --sample-idx 0
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Candidate s-grid spacings to test (meters)
CANDIDATE_DS = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

# Quantities to study
QUANTITIES = ["norm_emit_x", "norm_emit_y", "sigma_x", "sigma_y"]


def downsample_and_reconstruct(z_dense, y_dense, ds):
    """Downsample (z_dense, y_dense) to spacing `ds`, then linearly interpolate
    back onto the dense grid. Returns the reconstructed dense-grid values.
    """
    z_min, z_max = z_dense.min(), z_dense.max()
    z_coarse = np.arange(z_min, z_max + ds, ds)
    # Sample y at each coarse z by interpolation from the dense data
    y_coarse = np.interp(z_coarse, z_dense, y_dense)
    # Reconstruct dense grid from coarse samples
    y_reconstructed = np.interp(z_dense, z_coarse, y_coarse)
    return z_coarse, y_coarse, y_reconstructed


def compute_peak_error(y_true, y_reconstructed):
    """Return relative peak error: |peak_true - peak_reconstructed| / |peak_true|"""
    peak_true = np.max(np.abs(y_true))
    peak_recon = np.max(np.abs(y_reconstructed))
    if peak_true == 0:
        return 0.0
    return abs(peak_true - peak_recon) / abs(peak_true)


def plot_evolution(df_sample, output_path):
    """Plot all four quantities vs z for one sample."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    z = df_sample["mean_z"].values

    for ax, qty in zip(axes.flat, QUANTITIES):
        if qty in df_sample.columns:
            ax.plot(z, df_sample[qty].values, linewidth=0.8)
            ax.set_ylabel(qty)
            ax.grid(True, alpha=0.3)
        ax.set_xlabel("z (m)")

    fig.suptitle(f"Beam parameter evolution along z (sample {df_sample['sample_idx'].iloc[0]})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_resolution_study(df_sample, qty, output_path):
    """For one quantity, overlay reconstructions from several ds values."""
    z = df_sample["mean_z"].values
    y = df_sample[qty].values

    fig, axes = plt.subplots(len(CANDIDATE_DS), 1, figsize=(12, 2.2 * len(CANDIDATE_DS)),
                             sharex=True)
    if len(CANDIDATE_DS) == 1:
        axes = [axes]

    for ax, ds in zip(axes, CANDIDATE_DS):
        z_coarse, y_coarse, y_recon = downsample_and_reconstruct(z, y, ds)
        err = compute_peak_error(y, y_recon)
        n_pts = len(z_coarse)

        ax.plot(z, y, "k-", linewidth=0.8, label="true (dense)", alpha=0.7)
        ax.plot(z_coarse, y_coarse, "ro", markersize=3, label=f"ds={ds} m  ({n_pts} pts)")
        ax.plot(z, y_recon, "r--", linewidth=0.8, alpha=0.6, label="reconstructed")
        ax.set_ylabel(qty)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"ds = {ds} m   peak error = {err * 100:.2f}%",
                     fontsize=10,
                     color=("green" if err < 0.05 else "red"))

    axes[-1].set_xlabel("z (m)")
    fig.suptitle(f"Resolution study: {qty}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def make_summary_table(df_sample, output_path):
    """Compute peak error for every quantity × ds combination, write CSV."""
    z = df_sample["mean_z"].values
    rows = []
    for ds in CANDIDATE_DS:
        row = {"ds_m": ds, "n_points": int(np.ceil((z.max() - z.min()) / ds)) + 1}
        for qty in QUANTITIES:
            if qty not in df_sample.columns:
                continue
            y = df_sample[qty].values
            _, _, y_recon = downsample_and_reconstruct(z, y, ds)
            err = compute_peak_error(y, y_recon)
            row[f"{qty}_peak_err"] = err
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved {output_path}")

    # Print to console
    print("\nResolution study summary (peak error, % relative to max value):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Recommend coarsest ds with all errors < 5%
    err_cols = [c for c in df.columns if c.endswith("_peak_err")]
    df["max_err"] = df[err_cols].max(axis=1)
    ok = df[df["max_err"] < 0.05]
    if len(ok) > 0:
        rec = ok.sort_values("ds_m", ascending=False).iloc[0]
        print(f"\nRecommended coarsest ds with <5% peak error in all quantities: "
              f"ds = {rec['ds_m']} m  ({int(rec['n_points'])} points)")
    else:
        print(f"\nWARNING: no candidate ds keeps all quantities under 5% peak error. "
              f"Try smaller ds values.")


def compute_per_sample_errors(df, sample_ids):
    """Compute peak error per (sample, ds, qty). Returns long-form DataFrame."""
    rows = []
    for sid in sample_ids:
        df_s = df[df["sample_idx"] == sid].sort_values("mean_z").reset_index(drop=True)
        if len(df_s) < 2:
            continue
        z = df_s["mean_z"].values
        for ds in CANDIDATE_DS:
            for qty in QUANTITIES:
                if qty not in df_s.columns:
                    continue
                y = df_s[qty].values
                _, _, y_recon = downsample_and_reconstruct(z, y, ds)
                err = compute_peak_error(y, y_recon)
                rows.append({"sample_idx": sid, "ds_m": ds, "quantity": qty,
                             "peak_err": err})
    return pd.DataFrame(rows)


def make_multi_sample_summary(df, sample_ids, output_dir):
    """Run resolution study across many samples; report worst-case per (ds, qty)."""
    print(f"\nRunning multi-sample study on {len(sample_ids)} samples...")
    per_sample = compute_per_sample_errors(df, sample_ids)
    per_sample.to_csv(os.path.join(output_dir, "peak_errors_per_sample.csv"), index=False)

    # Aggregate: worst case, mean, 95th percentile per (ds, qty)
    agg = per_sample.groupby(["ds_m", "quantity"])["peak_err"].agg(
        ["max", "mean", lambda s: float(np.percentile(s, 95))]
    ).reset_index()
    agg.columns = ["ds_m", "quantity", "worst_case", "mean", "p95"]
    agg.to_csv(os.path.join(output_dir, "peak_errors_aggregated.csv"), index=False)

    # Pivot to wide format for easy reading: worst-case per ds × quantity
    worst_wide = agg.pivot(index="ds_m", columns="quantity", values="worst_case")
    worst_wide["max_over_quantities"] = worst_wide.max(axis=1)
    worst_wide = worst_wide.sort_index()
    worst_wide.to_csv(os.path.join(output_dir, "peak_errors_worst_case.csv"))

    print("\nWorst-case peak error across all samples (fraction of peak value):")
    print(worst_wide.to_string(float_format=lambda x: f"{x:.4f}"))

    # Recommendation
    ok = worst_wide[worst_wide["max_over_quantities"] < 0.05]
    if len(ok) > 0:
        rec_ds = ok.index.max()
        print(f"\nRecommended coarsest ds with worst-case <5% across all "
              f"{len(sample_ids)} samples: ds = {rec_ds} m")
    else:
        print("\nWARNING: no ds keeps worst-case under 5%. Use finer spacing.")

    # Plot: worst-case peak error vs ds, one line per quantity
    fig, ax = plt.subplots(figsize=(8, 5))
    for qty in QUANTITIES:
        sub = agg[agg["quantity"] == qty].sort_values("ds_m")
        ax.plot(sub["ds_m"], sub["worst_case"] * 100, "o-", label=f"{qty} (worst)")
    ax.axhline(5.0, color="red", linestyle="--", alpha=0.6, label="5% threshold")
    ax.set_xscale("log")
    ax.set_xlabel("ds (m)")
    ax.set_ylabel("Worst-case peak error (%)")
    ax.set_title(f"Worst-case peak error vs ds  ({len(sample_ids)} samples)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "worst_case_vs_ds.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(output_dir, 'worst_case_vs_ds.png')}")


def build_parser():
    p = argparse.ArgumentParser(
        description="S-spacing resolution study for emittance/beam-size profiles."
    )
    p.add_argument("--input", default="twiss-impact-output/twiss_vs_position.csv",
                   help="Long-form CSV from extract_twiss_from_impact.py")
    p.add_argument("--output-dir", default="resolution-study",
                   help="Output directory for plots and summary")
    p.add_argument("--sample-idx", type=int, default=0,
                   help="Which sample_idx to use for the single-sample plots (default: 0)")
    p.add_argument("--num-samples", type=int, default=1,
                   help="If >1, run multi-sample worst-case study on this many "
                        "random samples (default: 1, single-sample mode only)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for sample selection (default: 0)")
    p.add_argument("--z-min", type=float, default=0.001,
                   help="Minimum z (m) to include. Skips cathode artifacts "
                        "(single-point spikes near z=0). Default: 0.001 m (1 mm)")
    p.add_argument("--z-max", type=float, default=None,
                   help="Maximum z (m) to include. Default: no upper limit.")
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} rows, {df['sample_idx'].nunique()} samples")

    # Filter to the meaningful beam region (skip cathode artifacts at z ~ 0)
    n_before = len(df)
    df = df[df["mean_z"] >= args.z_min]
    if args.z_max is not None:
        df = df[df["mean_z"] <= args.z_max]
    print(f"  After z filter [{args.z_min}, {args.z_max}] m: {len(df)} rows "
          f"({n_before - len(df)} dropped)")

    df_sample = df[df["sample_idx"] == args.sample_idx].sort_values("mean_z").reset_index(drop=True)
    if len(df_sample) == 0:
        raise SystemExit(f"sample_idx {args.sample_idx} not found")
    print(f"Using sample {args.sample_idx}: {len(df_sample)} points, "
          f"z = {df_sample['mean_z'].min():.4f} to {df_sample['mean_z'].max():.4f} m")

    # 1. Overview plot
    plot_evolution(df_sample, os.path.join(args.output_dir, "evolution_overview.png"))

    # 2. Per-quantity resolution plots
    for qty in QUANTITIES:
        if qty in df_sample.columns:
            plot_resolution_study(
                df_sample, qty,
                os.path.join(args.output_dir, f"resolution_{qty}.png"),
            )

    # 3. Single-sample summary table
    make_summary_table(df_sample, os.path.join(args.output_dir, "peak_errors.csv"))

    # 4. Multi-sample worst-case study
    if args.num_samples > 1:
        all_ids = df["sample_idx"].unique()
        rng = np.random.default_rng(args.seed)
        n = min(args.num_samples, len(all_ids))
        chosen = rng.choice(all_ids, size=n, replace=False)
        make_multi_sample_summary(df, chosen, args.output_dir)


if __name__ == "__main__":
    main()
