"""Compute Twiss parameters from an existing Impact-extracted CSV.

Impact's stats only store sigma_x, cov_x__px (not cov_x__x or cov_px__px),
so the original compute_twiss() in extract_twiss_from_impact.py silently
skips. This script derives Twiss from the columns we do have:

    <x^2>  = sigma_x^2
    p_z    = gamma * beta * m_e * c    (from mean_kinetic_energy)
    <xx'>  = cov_x__px / p_z           (paraxial: x' = px/pz)
    eps_g  = norm_emit_x / (beta*gamma)
    <x'^2> = (eps_g^2 + <xx'>^2) / <x^2>
    beta_t = <x^2> / eps_g
    alpha  = -<xx'> / eps_g
    gamma  = <x'^2> / eps_g

Note: cov_x__px in Impact stats is in units of m * (eV/c), i.e. eV/c after
dividing by p_z [eV/c] you get a dimensionless angle * meter = m.

Usage:
    python compute_twiss_from_impact_csv.py \
        --input twiss-impact-output/twiss_vs_position.csv \
        --output twiss-impact-output/twiss_vs_position_with_twiss.csv
"""

import argparse

import numpy as np
import pandas as pd

# Electron rest mass energy in eV
M_E_C2_EV = 510998.95


def compute_twiss_columns(df):
    """Add Twiss columns to a DataFrame containing Impact stats."""
    ke = df["mean_kinetic_energy"].to_numpy()  # eV
    gamma_rel = 1.0 + ke / M_E_C2_EV
    beta_rel = np.sqrt(np.maximum(1.0 - 1.0 / gamma_rel**2, 0.0))
    betagamma = beta_rel * gamma_rel
    # p_z in eV/c (consistent with Impact's cov_x__px units of m * eV/c)
    pz_eVc = betagamma * M_E_C2_EV

    for plane, mom in [("x", "px"), ("y", "py")]:
        sigma = df[f"sigma_{plane}"].to_numpy()
        cov_xpx = df[f"cov_{plane}__{mom}"].to_numpy()
        norm_emit = df[f"norm_emit_{plane}"].to_numpy()

        x2 = sigma**2  # <x^2>
        # paraxial: <xx'> = <x*px>/p_z
        with np.errstate(divide="ignore", invalid="ignore"):
            xxp = np.where(pz_eVc > 0, cov_xpx / pz_eVc, np.nan)
            eps_geom = np.where(betagamma > 0, norm_emit / betagamma, np.nan)
            xp2 = np.where(x2 > 0, (eps_geom**2 + xxp**2) / x2, np.nan)
            beta_t = np.where(eps_geom > 0, x2 / eps_geom, np.nan)
            alpha_t = np.where(eps_geom > 0, -xxp / eps_geom, np.nan)
            gamma_t = np.where(eps_geom > 0, xp2 / eps_geom, np.nan)

        df[f"emit_geom_{plane}"] = eps_geom
        df[f"beta_{plane}"] = beta_t
        df[f"alpha_{plane}"] = alpha_t
        df[f"gamma_{plane}"] = gamma_t

    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input CSV (from extract_twiss_from_impact.py)")
    ap.add_argument("--output", required=True, help="Output CSV with added Twiss columns")
    ap.add_argument("--chunksize", type=int, default=500_000,
                    help="Rows per chunk (default 500000)")
    args = ap.parse_args()

    print(f"Reading {args.input} in chunks of {args.chunksize}...")
    first = True
    total = 0
    for chunk in pd.read_csv(args.input, chunksize=args.chunksize):
        chunk = compute_twiss_columns(chunk)
        chunk.to_csv(args.output, mode="w" if first else "a",
                     header=first, index=False)
        first = False
        total += len(chunk)
        print(f"  processed {total:,} rows")

    print(f"Done. Wrote {total:,} rows to {args.output}")


if __name__ == "__main__":
    main()
