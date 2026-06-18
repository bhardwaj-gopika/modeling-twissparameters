"""Extract Twiss parameters vs beamline position using Tao 'comb' commands.

For each row in particles-571.csv (that has Bmad settings), this script:
  1. Opens a Tao session with the FACET-II lattice
  2. Applies quad gradients and RF settings from that row
  3. Uses Tao commands to dump Twiss parameters at all lattice elements
  4. Saves results to CSV

This gives beta_x, alpha_x, beta_y, alpha_y, etc. as a function of
beamline position (s) for each set of input parameters.

Usage (on SDF where pytao and facet2-lattice are available):
    python extract_twiss_from_tao.py particles-571.csv \
        --output-dir twiss-tao-output \
        --max-rows 5 \
        --progress-every 1
"""

import argparse
import os
import traceback

import numpy as np
import pandas as pd
from pytao import SubprocessTao


# Bmad setting columns in particles-571.csv and how to apply them in Tao.
# The update_bmad_settings logic from Model_Calibration is replicated here:
#   - "theta0_deg" keys → set ele <name> PHI0 = value/360
#   - "rf_field_scale" keys → set ele <name> VOLTAGE = value
#   - "Q" keys → set ele <name> B1_GRADIENT = value
BMAD_SETTING_COLUMNS = [
    "bmad_L0BF_phase:theta0_deg",
    "bmad_L0BF_scale:rf_field_scale",
    "bmad_QA10361",
    "bmad_QA10371",
    "bmad_QE10425",
    "bmad_QE10441",
    "bmad_QE10511",
    "bmad_QE10525",
]

# Twiss parameters to extract at each element
# Tao python lat_list uses "ele.{mode}.{param}" syntax
TWISS_KEYS = {
    "ele.a.beta":  "beta_a",    # beta_x (Bmad 'a' mode = horizontal)
    "ele.a.alpha": "alpha_a",   # alpha_x
    "ele.a.gamma": "gamma_a",   # gamma_x
    "ele.a.eta":   "eta_a",     # dispersion_x
    "ele.a.etap":  "etap_a",    # dispersion prime_x
    "ele.b.beta":  "beta_b",    # beta_y (Bmad 'b' mode = vertical)
    "ele.b.alpha": "alpha_b",   # alpha_y
    "ele.b.gamma": "gamma_b",   # gamma_y
    "ele.b.eta":   "eta_b",     # dispersion_y
    "ele.b.etap":  "etap_b",    # dispersion prime_y
}


def apply_bmad_settings(tao, row):
    """Apply Bmad settings from a CSV row to a Tao session.

    Replicates the logic from Model_Calibration.utils.simulation_setup.update_bmad_settings.

    Parameters
    ----------
    tao : SubprocessTao
        Active Tao session.
    row : dict-like
        Row from particles-571.csv with bmad_ prefixed columns.
    """
    for col in BMAD_SETTING_COLUMNS:
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue

        # Strip "bmad_" prefix to get the setting key
        key = col.replace("bmad_", "", 1)

        if "theta0_deg" in key:
            ele_name = key.split("_")[0]
            tao.cmd(f"set ele {ele_name} PHI0={float(val) / 360}")
        elif "rf_field_scale" in key:
            ele_name = key.split("_")[0]
            tao.cmd(f"set ele {ele_name} VOLTAGE={float(val)}")
        elif "Q" in key:
            tao.cmd(f"set ele {key} B1_GRADIENT={float(val)}")


def extract_twiss_from_tao(tao):
    """Extract Twiss parameters at all lattice elements using Tao.

    Parameters
    ----------
    tao : SubprocessTao
        Active Tao session with settings already applied.

    Returns
    -------
    pd.DataFrame
        One row per element with columns: ele_name, s, and all Twiss keys.
    """
    # Get element names and s-positions
    ele_names = tao.cmd("python lat_list 1@0>>*|model ele.name")
    s_positions = tao.cmd("python lat_list 1@0>>*|model ele.s")

    # Clean up Tao output format: lines are "idx;value"
    ele_names = [line.split(";")[1].strip() if ";" in line else line.strip()
                 for line in ele_names if line.strip()]
    s_positions = [float(line.split(";")[1]) if ";" in line else float(line.strip())
                   for line in s_positions if line.strip()]

    n_elements = len(ele_names)

    data = {
        "ele_name": ele_names,
        "s": s_positions,
    }

    # Extract each Twiss parameter at all elements
    for tao_attr, col_name in TWISS_KEYS.items():
        raw = tao.cmd(f"python lat_list 1@0>>*|model {tao_attr}")
        values = []
        for line in raw:
            if not line.strip():
                continue
            try:
                val = float(line.split(";")[1]) if ";" in line else float(line.strip())
            except (ValueError, IndexError):
                val = np.nan
            values.append(val)

        if len(values) == n_elements:
            data[col_name] = values
        else:
            print(f"  WARNING: {tao_attr} returned {len(values)} values, expected {n_elements}")
            # Pad or truncate
            padded = values[:n_elements] + [np.nan] * max(0, n_elements - len(values))
            data[col_name] = padded

    return pd.DataFrame(data)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract Twiss parameters vs beamline position using Tao."
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV (e.g. particles-571.csv) with Bmad setting columns",
    )
    parser.add_argument(
        "--output-dir",
        default="twiss-tao-output",
        help="Output directory (default: twiss-tao-output)",
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
        "--lattice-dir",
        default=None,
        help=(
            "Path to facet2-lattice directory. "
            "Default: uses FACET2_LATTICE env var or ./facet2-lattice"
        ),
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

    # Resolve lattice path
    if args.lattice_dir:
        lattice_dir = args.lattice_dir
    elif "FACET2_LATTICE" in os.environ:
        lattice_dir = os.environ["FACET2_LATTICE"]
    else:
        lattice_dir = os.path.join(os.getcwd(), "facet2-lattice")

    tao_init = f"{lattice_dir}/bmad/models/f2_elec/tao.init"
    if not os.path.exists(tao_init):
        raise FileNotFoundError(
            f"Tao init file not found: {tao_init}\n"
            f"Set --lattice-dir or FACET2_LATTICE env var."
        )

    print(f"Reading {args.input_csv}...")
    input_df = pd.read_csv(args.input_csv, low_memory=False)
    print(f"  Total rows: {len(input_df)}")

    # Check that Bmad setting columns exist
    missing_cols = [c for c in BMAD_SETTING_COLUMNS if c not in input_df.columns]
    if missing_cols:
        raise ValueError(f"Missing Bmad setting columns: {missing_cols}")

    # Filter to rows that have at least some Bmad settings (non-null quads)
    has_settings = input_df["bmad_QA10361"].notna()
    valid_indices = input_df.index[has_settings].tolist()
    print(f"  Rows with Bmad settings: {len(valid_indices)}")

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

    os.makedirs(args.output_dir, exist_ok=True)
    long_path = os.path.join(args.output_dir, "twiss_vs_position.csv")
    ele_ref_path = os.path.join(args.output_dir, "element_reference.csv")
    print(f"Streaming long-form output to {long_path}")

    # Resume support: skip csv_row_idx values already present in the output CSV
    done_csv_row_idx = set()
    next_sample_idx = 0
    long_writer = None
    long_columns = None
    if args.resume and os.path.exists(long_path):
        try:
            existing = pd.read_csv(long_path, usecols=["sample_idx", "csv_row_idx"])
            done_csv_row_idx = set(existing["csv_row_idx"].astype(int).unique())
            if len(existing) > 0:
                next_sample_idx = int(existing["sample_idx"].max()) + 1
            head = pd.read_csv(long_path, nrows=0)
            long_columns = list(head.columns)
            long_writer = True
            print(f"  Resume: {len(done_csv_row_idx)} samples already done in {long_path}")
        except Exception as e:
            print(f"  WARNING: could not parse existing output for resume: {e}")

    if done_csv_row_idx:
        before = len(valid_indices)
        valid_indices = [i for i in valid_indices if i not in done_csv_row_idx]
        print(f"  After resume filter: {len(valid_indices)} remaining "
              f"({before - len(valid_indices)} skipped)")

    n_success = 0
    n_fail = 0
    n_elements = None
    sample_idx = next_sample_idx
    first_sample_df = None

    for i, csv_row_idx in enumerate(valid_indices):
        row = input_df.iloc[csv_row_idx]

        if (i + 1) % args.progress_every == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(valid_indices)} (row {csv_row_idx})")

        try:
            with SubprocessTao(
                f"-init {tao_init} -noplot",
                plot=False,
            ) as tao:
                apply_bmad_settings(tao, row)

                # Extract Twiss at all elements
                twiss_df = extract_twiss_from_tao(tao)
                twiss_df["sample_idx"] = sample_idx
                twiss_df["csv_row_idx"] = csv_row_idx

                if n_elements is None:
                    n_elements = len(twiss_df)
                    first_sample_df = twiss_df[["ele_name", "s"]].copy()
                    print(f"  Found {n_elements} elements in lattice")

                if long_writer is None:
                    long_columns = list(twiss_df.columns)
                    twiss_df.to_csv(long_path, index=False, mode="w")
                    long_writer = True
                else:
                    twiss_df = twiss_df.reindex(columns=long_columns)
                    twiss_df.to_csv(long_path, index=False, mode="a", header=False)

                n_success += 1
                sample_idx += 1
                del twiss_df

        except Exception as e:
            if i < 10:
                print(f"    ERROR on row {csv_row_idx}: {e}")
                traceback.print_exc()
            n_fail += 1

    print(f"\nDone: {n_success} succeeded, {n_fail} failed out of {len(valid_indices)}")

    if first_sample_df is not None:
        first_sample_df.to_csv(ele_ref_path, index=False)
        print(f"Saved element reference ({len(first_sample_df)} elements) to {ele_ref_path}")
    else:
        print("No data extracted.")


if __name__ == "__main__":
    main()
