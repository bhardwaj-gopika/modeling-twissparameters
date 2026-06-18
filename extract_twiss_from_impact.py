"""Extract Twiss parameters and emittances vs beamline position from Impact-T archives.

Reads particles-571.csv, loads each Impact-T archive, and extracts beam
statistics (sigma, emittance, Twiss parameters) as a function of longitudinal
position (mean_z) from I.output['stats'].

Twiss parameters are computed from second moments:
    eps^2  = <x^2><x'^2> - <xx'>^2
    beta   = <x^2> / eps
    alpha  = -<xx'> / eps
    gamma  = (1 + alpha^2) / beta

Usage (on SDF where archive paths are accessible):
    python extract_twiss_from_impact.py particles-571.csv \
        --output-dir twiss-output \
        --max-rows 100 \
        --progress-every 10
"""

import argparse
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import impact


def extract_stats_from_archive(archive_path):
    """Load an Impact-T archive and extract beam stats vs position.

    Parameters
    ----------
    archive_path : str
        Path to the Impact-T archive file.

    Returns
    -------
    dict or None
        Dictionary with stat arrays, or None if loading fails.
    """
    I = impact.Impact()
    I.load_archive(archive_path)

    if "stats" not in I.output:
        return None

    stats = I.output["stats"]

    result = {"archive_path": archive_path}

    # Core position / time arrays
    for key in ["mean_z", "t"]:
        if key in stats:
            result[key] = stats[key]

    # Beam sizes
    for key in ["sigma_x", "sigma_y", "sigma_z"]:
        if key in stats:
            result[key] = stats[key]

    # Mean positions / momenta
    for key in ["mean_x", "mean_y", "mean_px", "mean_py", "mean_pz",
                "mean_kinetic_energy"]:
        if key in stats:
            result[key] = stats[key]

    # Normalized emittances
    for key in ["norm_emit_x", "norm_emit_y", "norm_emit_z"]:
        if key in stats:
            result[key] = stats[key]

    # Second moments / covariances for Twiss computation
    cov_keys = [
        "cov_x__x", "cov_x__px", "cov_px__px",
        "cov_y__y", "cov_y__py", "cov_py__py",
        "cov_z__z", "cov_z__pz", "cov_pz__pz",
    ]
    for key in cov_keys:
        if key in stats:
            result[key] = stats[key]

    # Compute Twiss parameters from second moments if available
    result = compute_twiss(result)

    return result


def compute_twiss(stats):
    """Compute Twiss parameters (beta, alpha, gamma, geometric emittance)
    from covariance data in stats dict.

    Parameters
    ----------
    stats : dict
        Must contain covariance arrays from Impact stats.

    Returns
    -------
    dict
        Input dict augmented with Twiss parameter arrays.
    """
    for plane, (pos, mom) in {"x": ("x", "px"), "y": ("y", "py")}.items():
        cov_qq = f"cov_{pos}__{pos}"    # <x^2>
        cov_qp = f"cov_{pos}__{mom}"    # <x*x'>
        cov_pp = f"cov_{mom}__{mom}"    # <x'^2>

        if cov_qq in stats and cov_qp in stats and cov_pp in stats:
            xx = np.array(stats[cov_qq])
            xxp = np.array(stats[cov_qp])
            xpxp = np.array(stats[cov_pp])

            # Geometric emittance: eps^2 = <x^2><x'^2> - <xx'>^2
            eps_sq = xx * xpxp - xxp**2
            eps_sq = np.maximum(eps_sq, 0)  # guard against numerical noise
            eps = np.sqrt(eps_sq)

            # Twiss parameters (avoid division by zero)
            with np.errstate(divide="ignore", invalid="ignore"):
                beta = np.where(eps > 0, xx / eps, np.nan)
                alpha = np.where(eps > 0, -xxp / eps, np.nan)
                gamma = np.where(eps > 0, xpxp / eps, np.nan)

            stats[f"emit_geometric_{plane}"] = eps
            stats[f"beta_{plane}"] = beta
            stats[f"alpha_{plane}"] = alpha
            stats[f"gamma_{plane}"] = gamma

    return stats


def save_stats_to_csv(all_stats, output_dir):
    """Save per-sample stats to a single wide CSV and a long-form CSV.

    Wide CSV: one row per (sample, s-position), all stat columns.
    Also saves the available stat keys for reference.

    Parameters
    ----------
    all_stats : list of dict
        Each dict has stat arrays plus metadata.
    output_dir : str
        Directory to write output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build long-form DataFrame: one row per (sample_idx, step)
    rows = []
    for sample_idx, stats in enumerate(all_stats):
        archive_path = stats.get("archive_path", "")
        csv_row_idx = stats.get("csv_row_idx", -1)

        # Use mean_z as the position coordinate
        if "mean_z" not in stats:
            continue

        n_steps = len(stats["mean_z"])
        for step in range(n_steps):
            row = {
                "sample_idx": sample_idx,
                "csv_row_idx": csv_row_idx,
                "step": step,
            }
            # Add all array-valued stats at this step
            for key, val in stats.items():
                if key in ("archive_path", "csv_row_idx"):
                    continue
                if isinstance(val, np.ndarray) and len(val) == n_steps:
                    row[key] = val[step]
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, "twiss_vs_position.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows ({len(all_stats)} samples) to {output_path}")

    # Save summary: available keys and their shapes
    if all_stats:
        sample = all_stats[0]
        keys_info = []
        for key, val in sample.items():
            if isinstance(val, np.ndarray):
                keys_info.append({"key": key, "shape": str(val.shape), "dtype": str(val.dtype)})
        keys_df = pd.DataFrame(keys_info)
        keys_path = os.path.join(output_dir, "available_stat_keys.csv")
        keys_df.to_csv(keys_path, index=False)
        print(f"Saved stat key reference to {keys_path}")


def save_endpoint_summary(all_stats, input_df, output_dir):
    """Save a summary CSV with endpoint (screen) values and input parameters.

    One row per sample with Twiss/emittance at the final s-position,
    plus the emittances from the original CSV (emit_x_241, etc.).

    Parameters
    ----------
    all_stats : list of dict
        Extracted stats per sample.
    input_df : pd.DataFrame
        Original particles-571.csv rows (filtered to those with archives).
    output_dir : str
        Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    for stats in all_stats:
        csv_row_idx = stats.get("csv_row_idx", -1)
        row = {"csv_row_idx": csv_row_idx}

        if "mean_z" not in stats:
            continue

        # Final position values
        for key, val in stats.items():
            if key in ("archive_path", "csv_row_idx"):
                continue
            if isinstance(val, np.ndarray) and len(val) > 0:
                row[f"{key}_final"] = val[-1]
                row[f"{key}_n_steps"] = len(val)

        # Add emittances from original CSV if available
        if csv_row_idx >= 0 and csv_row_idx < len(input_df):
            orig_row = input_df.iloc[csv_row_idx]
            for col in ["emit_x_241", "emit_y_241", "emit_x_571", "emit_y_571"]:
                if col in orig_row.index:
                    row[col] = orig_row[col]

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "endpoint_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved endpoint summary ({len(summary_df)} samples) to {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract Twiss parameters and emittances vs beamline position from Impact-T archives."
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV (e.g. particles-571.csv) with impact_archive column",
    )
    parser.add_argument(
        "--output-dir",
        default="twiss-output",
        help="Output directory (default: twiss-output)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Process at most N rows (for testing). Default: all.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N rows (default: 100)",
    )
    parser.add_argument(
        "--archive-column",
        default="impact_archive",
        help="Column name for the Impact-T archive path (default: impact_archive)",
    )
    parser.add_argument(
        "--min-alive-particles",
        type=int,
        default=90000,
        help="Skip samples whose n_particles_571 column is below this value "
             "(matches the modeling-571 alive filter). Default: 90000. "
             "Set to 0 to disable.",
    )
    parser.add_argument(
        "--alive-column",
        default="n_particles_571",
        help="Column with the alive-particle count used for filtering. "
             "Default: n_particles_571.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set and output CSV already exists, skip csv_row_idx values "
             "already present in it and append new samples.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(f"Reading {args.input_csv}...")
    input_df = pd.read_csv(args.input_csv)
    print(f"  Total rows: {len(input_df)}")

    if args.archive_column not in input_df.columns:
        raise ValueError(f"Column '{args.archive_column}' not found in {args.input_csv}")

    # Filter to rows with non-empty impact_archive
    has_archive = input_df[args.archive_column].notna() & (input_df[args.archive_column].str.strip() != "")
    valid_indices = input_df.index[has_archive].tolist()
    print(f"  Rows with {args.archive_column}: {len(valid_indices)}")

    # Alive-particle filter (matches modeling-571 cut)
    if args.min_alive_particles and args.min_alive_particles > 0:
        if args.alive_column not in input_df.columns:
            print(f"  WARNING: --alive-column '{args.alive_column}' not found; "
                  f"skipping alive filter")
        else:
            before = len(valid_indices)
            alive_ok = input_df[args.alive_column] >= args.min_alive_particles
            valid_indices = [i for i in valid_indices if alive_ok.iat[i]]
            print(f"  After {args.alive_column} >= {args.min_alive_particles} "
                  f"filter: {len(valid_indices)} ({before - len(valid_indices)} dropped)")

    if args.max_rows is not None:
        valid_indices = valid_indices[:args.max_rows]
        print(f"  Processing first {args.max_rows} rows")

    # Stream rows to CSV instead of accumulating all stats in memory.
    os.makedirs(args.output_dir, exist_ok=True)
    long_path = os.path.join(args.output_dir, "twiss_vs_position.csv")
    summary_path = os.path.join(args.output_dir, "endpoint_summary.csv")
    print(f"Streaming long-form output to {long_path}")
    print(f"Streaming endpoint summary to {summary_path}")

    # Resume support: read already-processed csv_row_idx from existing CSV
    done_csv_row_idx = set()
    next_sample_idx = 0
    long_writer = None
    summary_writer = None
    long_columns = None
    summary_columns = None
    if args.resume and os.path.exists(long_path):
        try:
            existing = pd.read_csv(long_path, usecols=["sample_idx", "csv_row_idx"])
            done_csv_row_idx = set(existing["csv_row_idx"].astype(int).unique())
            if len(existing) > 0:
                next_sample_idx = int(existing["sample_idx"].max()) + 1
            # Recover header / column order so appends line up
            head = pd.read_csv(long_path, nrows=0)
            long_columns = list(head.columns)
            long_writer = True
            print(f"  Resume: {len(done_csv_row_idx)} samples already done in {long_path}")
        except Exception as e:
            print(f"  WARNING: could not parse existing output for resume: {e}")
    if args.resume and os.path.exists(summary_path):
        try:
            head = pd.read_csv(summary_path, nrows=0)
            summary_columns = list(head.columns)
            summary_writer = True
        except Exception as e:
            print(f"  WARNING: could not parse existing summary for resume: {e}")

    if done_csv_row_idx:
        before = len(valid_indices)
        valid_indices = [i for i in valid_indices if i not in done_csv_row_idx]
        print(f"  After resume filter: {len(valid_indices)} remaining "
              f"({before - len(valid_indices)} skipped)")

    n_success = 0
    n_fail = 0
    sample_idx = next_sample_idx
    first_keys = None

    for i, csv_row_idx in enumerate(valid_indices):
        archive_path = input_df.at[csv_row_idx, args.archive_column].strip()

        if (i + 1) % args.progress_every == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(valid_indices)}: {archive_path}")

        if not os.path.exists(archive_path):
            if i < 5:
                print(f"    WARNING: Archive not found: {archive_path}")
            n_fail += 1
            continue

        try:
            stats = extract_stats_from_archive(archive_path)
        except Exception as e:
            if i < 10:
                print(f"    ERROR on row {csv_row_idx}: {e}")
            n_fail += 1
            continue

        if stats is None or "mean_z" not in stats:
            n_fail += 1
            continue

        if first_keys is None:
            first_keys = sorted(
                k for k, v in stats.items()
                if isinstance(v, np.ndarray)
            )
            print("\nAvailable stat keys:")
            for key in first_keys:
                v = stats[key]
                print(f"  {key}: shape={v.shape}, dtype={v.dtype}")
            print()

        n_steps = len(stats["mean_z"])
        arr_keys = [k for k, v in stats.items()
                    if isinstance(v, np.ndarray) and len(v) == n_steps]

        # Build long-form rows for this sample
        long_rows = []
        for step in range(n_steps):
            row = {
                "sample_idx": sample_idx,
                "csv_row_idx": csv_row_idx,
                "step": step,
            }
            for key in arr_keys:
                row[key] = stats[key][step]
            long_rows.append(row)
        long_df = pd.DataFrame(long_rows)

        if long_writer is None:
            long_columns = list(long_df.columns)
            long_df.to_csv(long_path, index=False, mode="w")
            long_writer = True
        else:
            # Reindex to the locked column order; missing cols become NaN
            long_df = long_df.reindex(columns=long_columns)
            long_df.to_csv(long_path, index=False, mode="a", header=False)

        # Build endpoint-summary row
        summary_row = {"sample_idx": sample_idx, "csv_row_idx": csv_row_idx}
        for key in arr_keys:
            summary_row[f"{key}_final"] = stats[key][-1]
            summary_row[f"{key}_n_steps"] = n_steps
        if 0 <= csv_row_idx < len(input_df):
            orig_row = input_df.iloc[csv_row_idx]
            for col in ["emit_x_241", "emit_y_241", "emit_x_571", "emit_y_571",
                        args.alive_column]:
                if col in orig_row.index:
                    summary_row[col] = orig_row[col]
        summary_df = pd.DataFrame([summary_row])
        if summary_writer is None:
            summary_columns = list(summary_df.columns)
            summary_df.to_csv(summary_path, index=False, mode="w")
            summary_writer = True
        else:
            summary_df = summary_df.reindex(columns=summary_columns)
            summary_df.to_csv(summary_path, index=False, mode="a", header=False)

        n_success += 1
        sample_idx += 1

        # Free per-sample memory immediately
        del stats, long_rows, long_df, summary_row, summary_df

    print(f"\nDone: {n_success} succeeded, {n_fail} failed out of {len(valid_indices)}")
    if n_success == 0:
        print("No data extracted. Check that archive paths are accessible.")


if __name__ == "__main__":
    main()
