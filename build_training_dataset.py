"""Build the ML training dataset by interpolating beam evolution onto a
uniform s-grid and joining with the 19 machine control parameters.

Preprocessing applied per sample:
  1. Alive-particle filter (n_particles_571 >= 90000, from particles-571.csv).
  2. Input-feature range filter (per supervisor-specified bounds; see
     INPUT_RANGES below).
  3. Target-state range filter at three screens:
       - PR10241  (s = 0.942084 m)  : sigma_x, sigma_y, mean_pz, mean_t
       - L0AFEND  (s = 4.127448 m)  : sigma_x, sigma_y, mean_pz, mean_t
       - PR10571                    : sigma_x, sigma_y (from particles CSV),
                                       mean_pz (derived from mean_kinetic_energy)
     Values at PR10241 and L0AFEND are obtained by interpolating the Impact-T
     stats CSV at the corresponding s.

PR10571's actual lattice position is s = 14.232788 m, but Impact-T tracking
in this dataset stops at L0AFEND (s ~ 4.13 m). Beam properties at PR10571
therefore come from the precomputed columns in particles-571.csv (downstream
Bmad outputs), not from interpolation of the Impact stats.

For each sample that survives all filters:
  - Trim rows to s in [s_min, s_max].
  - Interpolate target columns onto s_grid = arange(s_min, s_max+ds, ds).
  - Emit one row per s-point with [sample_idx, csv_row_idx, 19 inputs, s, targets].

Streaming: processes one sample at a time; constant memory.

Usage:
    python build_training_dataset.py \
        --twiss twiss-impact-output/twiss_vs_position_with_twiss.csv \
        --particles particles-571.csv \
        --output dataset.csv
"""

import argparse

import numpy as np
import pandas as pd

# ---- columns -----------------------------------------------------------

INPUT_COLUMNS = [
    "CQ10121:b1_gradient",
    "GUNF:rf_field_scale",
    "GUNF:theta0_deg",
    "L0AF_phase:theta0_deg",
    "L0AF_scale:rf_field_scale",
    "L0BF_phase:theta0_deg",
    "L0BF_scale:rf_field_scale",
    "QA10361",
    "QA10371",
    "QE10425",
    "QE10441",
    "QE10511",
    "QE10525",
    "SOL10111:solenoid_field_scale",
    "SQ10122:b1_gradient",
    "distgen:VCC",
    "distgen:t_dist:sigma_t:value",
    "distgen:total_charge:value",
    "impact_VCC_Cal",
]

TARGET_COLUMNS = [
    "sigma_x", "sigma_y", "sigma_z",
    "norm_emit_x", "norm_emit_y",
    "emit_geom_x", "emit_geom_y",
    "beta_x", "alpha_x",
    "beta_y", "alpha_y",
    "mean_kinetic_energy",
]

# ---- physical constants ------------------------------------------------

M_E_C = 511e3  # electron rest mass × c, in eV (eV/c for momentum)

# ---- screen s-positions (from Impact lattice in 82.h5) -----------------

S_PR10241 = 0.942084
S_L0AFEND = 4.127448
# PR10571 is at s=14.232788 m in the lattice but Impact tracking stops
# at L0AFEND, so we use particles-571.csv columns for PR10571 quantities.

# ---- supervisor-specified input-feature ranges -------------------------

INPUT_RANGES = {
    "GUNF:theta0_deg": (-81.0, -68.0),
    "GUNF:rf_field_scale": (49e6, 52e6),
    "SOL10111:solenoid_field_scale": (0.25, 0.29),
    "CQ10121:b1_gradient": (-0.05, 0.0),
    "SQ10122:b1_gradient": (-0.05, 0.02),
    "distgen:t_dist:sigma_t:value": (-0.25, 2.0),
    "distgen:total_charge:value": (900.0, 1050.0),
    "L0AF_phase:theta0_deg": (-10.0, 0.0),
    "L0AF_scale:rf_field_scale": (50e6, 53e6),
    "L0BF_phase:theta0_deg": (-10.0, 20.0),
    "L0BF_scale:rf_field_scale": (54e6, 65e6),
    "QA10361": (3.0, 4.0),
    "QA10371": (-4.2, -3.2),
    "QE10425": (3.0, 9.0),
    "QE10441": (-8.0, -5.0),
    "QE10511": (2.0, 4.0),
    "QE10525": (-7.0, 1.0),
}

# ---- supervisor-specified target-state ranges at each screen -----------

TARGET_RANGES = {
    "241": {
        "sigma_x": (0.0, 0.002),
        "sigma_y": (0.0, 0.002),
        "mean_pz": (6e6, 6.3e6),
        "mean_t": (3.17e-9, 3.2e-9),
    },
    "L0AFEND": {
        "sigma_x": (0.0, 0.002),
        "sigma_y": (0.0, 0.002),
        "mean_pz": (98354515.47820634, 111867719.44874424),
        "mean_t": (1.38e-8, 1.39e-8),
    },
    "571": {
        "sigma_x": (0.0, 0.002),
        "sigma_y": (0.0, 0.002),
        "mean_pz": (155e6, 175e6),
        # mean_t at 571 not available; filter dropped per supervisor.
    },
}

# Columns needed from the twiss CSV (in addition to position + TARGET_COLUMNS)
EXTRA_TWISS_COLS = ["t", "mean_kinetic_energy"]  # for screen-target filters
# mean_kinetic_energy is already in TARGET_COLUMNS but listed for clarity.


# ---- helpers -----------------------------------------------------------

def pz_from_ke(ke_eV):
    """Convert kinetic energy in eV to longitudinal momentum in eV/c.
       pz = sqrt((KE + mc^2)^2 - (mc^2)^2)
    """
    ke = np.asarray(ke_eV, dtype=float)
    return np.sqrt(np.maximum((ke + M_E_C) ** 2 - M_E_C ** 2, 0.0))


def interp_at(s_query, s_src, y_src):
    """1-D interpolation at a single s value; nan if out of range."""
    if s_query < s_src.min() or s_query > s_src.max():
        return np.nan
    return float(np.interp(s_query, s_src, y_src))


def in_range(val, lo_hi):
    lo, hi = lo_hi
    return np.isfinite(val) and lo <= val <= hi


def passes_input_ranges(knobs, ranges):
    for col, lim in ranges.items():
        if not in_range(float(knobs[col]), lim):
            return False
    return True


def passes_screen_at_s(sample_df, s_query, ranges_at_s, position_col):
    """Return True if the interpolated beam state at s_query falls within all
    target ranges given for that screen.
    """
    s_src = sample_df[position_col].to_numpy()
    if s_query < s_src.min() or s_query > s_src.max():
        return False

    sx = interp_at(s_query, s_src, sample_df["sigma_x"].to_numpy())
    sy = interp_at(s_query, s_src, sample_df["sigma_y"].to_numpy())
    ke = interp_at(s_query, s_src, sample_df["mean_kinetic_energy"].to_numpy())
    pz = float(pz_from_ke(ke)) if np.isfinite(ke) else np.nan
    t_val = interp_at(s_query, s_src, sample_df["t"].to_numpy()) \
            if "t" in sample_df.columns else np.nan

    for tgt, lim in ranges_at_s.items():
        if tgt == "sigma_x":
            v = sx
        elif tgt == "sigma_y":
            v = sy
        elif tgt == "mean_pz":
            v = pz
        elif tgt == "mean_t":
            v = t_val
        else:
            continue
        if not in_range(v, lim):
            return False
    return True


def passes_571(particle_row, ranges_at_s):
    """571 endpoint check using precomputed columns from particles-571.csv."""
    sx = float(particle_row.get("sigma_x", np.nan))
    sy = float(particle_row.get("sigma_y", np.nan))
    ke = float(particle_row.get("mean_kinetic_energy", np.nan))
    pz = float(pz_from_ke(ke)) if np.isfinite(ke) else np.nan

    for tgt, lim in ranges_at_s.items():
        if tgt == "sigma_x":
            v = sx
        elif tgt == "sigma_y":
            v = sy
        elif tgt == "mean_pz":
            v = pz
        else:
            continue
        if not in_range(v, lim):
            return False
    return True


def interpolate_sample(sample_df, s_grid, position_col):
    sample_df = sample_df.sort_values(position_col)
    s_src = sample_df[position_col].to_numpy()
    if len(s_src) < 2:
        return None
    in_grid = (s_grid >= s_src.min()) & (s_grid <= s_src.max())
    if not in_grid.any():
        return None
    s_out = s_grid[in_grid]
    out = {"s": s_out}
    for col in TARGET_COLUMNS:
        if col not in sample_df.columns:
            out[col] = np.full(len(s_out), np.nan)
            continue
        out[col] = np.interp(s_out, s_src, sample_df[col].to_numpy())
    return pd.DataFrame(out)


# ---- main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twiss", required=True)
    ap.add_argument("--particles", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--s-min", type=float, default=0.001)
    ap.add_argument("--s-max", type=float, default=4.13,
                    help="Max s for the training grid (default 4.13 m ≈ L0AFEND)")
    ap.add_argument("--ds", type=float, default=0.02)
    ap.add_argument("--position-col", default="mean_z")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--min-alive-particles", type=int, default=90_000,
                    help="0 disables alive filter")
    ap.add_argument("--no-input-filter", action="store_true",
                    help="Disable input-feature range filter")
    ap.add_argument("--no-target-filter", action="store_true",
                    help="Disable 241/L0AFEND/571 target range filters")
    args = ap.parse_args()

    print(f"[run] Screen positions: PR10241 s={S_PR10241}, L0AFEND s={S_L0AFEND}")
    print(f"[run] Input filter: {'OFF' if args.no_input_filter else 'ON'} "
          f"({len(INPUT_RANGES)} bounded features)")
    print(f"[run] Target filter: {'OFF' if args.no_target_filter else 'ON'} "
          f"(241, L0AFEND, 571)")

    s_grid = np.arange(args.s_min, args.s_max + args.ds / 2, args.ds)
    print(f"[run] s-grid: {len(s_grid)} points from {s_grid[0]:.4f} to "
          f"{s_grid[-1]:.4f} m (ds={args.ds})")

    # particles-571.csv: load only the columns we need
    needed_p = (set(INPUT_COLUMNS)
                | {"n_particles_571", "sigma_x", "sigma_y",
                   "mean_kinetic_energy"})
    print(f"[run] Loading particles CSV ({len(needed_p)} cols) ...")
    particles = pd.read_csv(args.particles,
                            usecols=lambda c: c in needed_p,
                            low_memory=False)
    missing = [c for c in INPUT_COLUMNS if c not in particles.columns]
    if missing:
        raise SystemExit(f"Missing input columns in {args.particles}: {missing}")
    has_alive = "n_particles_571" in particles.columns
    print(f"  loaded {len(particles)} rows")

    if args.min_alive_particles > 0 and has_alive:
        alive_mask = (particles["n_particles_571"].fillna(0) >=
                      args.min_alive_particles).to_numpy()
        print(f"  alive filter (>= {args.min_alive_particles}): "
              f"{int(alive_mask.sum())} / {len(particles)} pass")
    else:
        alive_mask = np.ones(len(particles), dtype=bool)

    # Streaming over twiss CSV
    print(f"[run] Streaming {args.twiss} in chunks of {args.chunksize}...")
    first_write = True
    kept = 0
    skipped = {"alive": 0, "input": 0, "tgt_241": 0, "tgt_L0AFEND": 0,
               "tgt_571": 0, "other": 0}
    total_rows_out = 0

    buffer = None
    current_id = None

    def flush_sample(buf, sample_id):
        nonlocal first_write, kept, total_rows_out

        if sample_id is None or sample_id < 0 or sample_id >= len(particles):
            skipped["other"] += 1
            return

        # Alive filter
        if not alive_mask[sample_id]:
            skipped["alive"] += 1
            return

        knobs = particles.iloc[sample_id]

        # Input range filter
        if not args.no_input_filter:
            if not passes_input_ranges(knobs, INPUT_RANGES):
                skipped["input"] += 1
                return

        # Target-state filters at 241 and L0AFEND (need interpolation)
        if not args.no_target_filter:
            buf_sorted = buf.sort_values(args.position_col)
            if not passes_screen_at_s(buf_sorted, S_PR10241,
                                      TARGET_RANGES["241"], args.position_col):
                skipped["tgt_241"] += 1
                return
            if not passes_screen_at_s(buf_sorted, S_L0AFEND,
                                      TARGET_RANGES["L0AFEND"],
                                      args.position_col):
                skipped["tgt_L0AFEND"] += 1
                return
            if not passes_571(knobs, TARGET_RANGES["571"]):
                skipped["tgt_571"] += 1
                return

        # Trim to grid range and interpolate
        buf = buf[(buf[args.position_col] >= args.s_min) &
                  (buf[args.position_col] <= args.s_max)]
        if buf.empty:
            skipped["other"] += 1
            return

        interp = interpolate_sample(buf, s_grid, args.position_col)
        if interp is None or interp.empty:
            skipped["other"] += 1
            return

        for c in INPUT_COLUMNS:
            interp[c] = knobs[c]
        interp["csv_row_idx"] = sample_id
        interp["sample_idx"] = kept

        col_order = (["sample_idx", "csv_row_idx"] + INPUT_COLUMNS + ["s"] +
                     TARGET_COLUMNS)
        interp = interp[col_order]

        interp.to_csv(args.output, mode="w" if first_write else "a",
                      header=first_write, index=False)
        first_write = False
        kept += 1
        total_rows_out += len(interp)

        if kept % args.progress_every == 0:
            print(f"  {kept} kept, {total_rows_out:,} rows | "
                  f"skipped: {skipped}")

    needed_twiss = (["csv_row_idx", args.position_col]
                    + TARGET_COLUMNS + EXTRA_TWISS_COLS)
    for chunk in pd.read_csv(args.twiss, chunksize=args.chunksize,
                             usecols=lambda c: c in needed_twiss):
        for sample_id, group in chunk.groupby("csv_row_idx", sort=False):
            if current_id is None:
                current_id = int(sample_id)
                buffer = group
            elif int(sample_id) == current_id:
                buffer = pd.concat([buffer, group], ignore_index=True)
            else:
                flush_sample(buffer, current_id)
                current_id = int(sample_id)
                buffer = group

    if buffer is not None and current_id is not None:
        flush_sample(buffer, current_id)

    print(f"\nDone. {kept} samples kept, {total_rows_out:,} rows written")
    print(f"Skipped: {skipped}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
