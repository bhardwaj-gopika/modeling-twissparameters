# AGENTS.md — FACET-II Beam-Evolution Surrogate (Twiss + Beam Parameters)

## What This Codebase Does

This repository builds a **second-generation ML surrogate model** for the FACET-II injector at SLAC. Where the previous model (`modeling-571`) predicts the beam covariance only at screen 571, this model predicts **beam properties as a continuous function of position `s` along the beamline**.

Inputs: 19 machine control parameters + position `s`.
Outputs (target list):
- **From Impact-T tracking**: `norm_emit_x`, `norm_emit_y`, `sigma_x`, `sigma_y`, `sigma_z`, energy / energy spread.
- **From Tao lattice optics**: Twiss `β_a`, `α_a`, `β_b`, `α_b` (and `γ`, `η`, `η'`).

Architecture: `(19 control params, s) → MLP → (target values at that s)` ("architecture B" in earlier discussions).

### Scientific Context

- **Accelerator**: FACET-II electron-beam injector at SLAC.
- **Beam parameters vs s**: A beam's `sigma`, emittance, and Twiss functions evolve continuously through magnets, RF cavities, and drifts. The surrogate captures this full evolution.
- **Two data sources**: Impact-T provides particle-tracking statistics (true beam moments); Tao provides analytic lattice optics (design Twiss). Both are extracted independently per simulation and merged on `s`.
- **Alive-particle filter**: Samples with fewer than 90,000 surviving particles at screen 571 (`n_particles_571 < 90000`) are removed before any analysis. This matches the cut used by the `modeling-571` model and removes ~14% of samples (~9k of ~63k), which would otherwise produce extreme outliers in beam statistics.
- **s-spacing**: A resolution study chose **ds = 0.02 m** (2 cm) as the training-grid spacing. Worst-case peak error <1.3% across 100 random samples, well under the 5% threshold supervisor specified.

## Pipeline Steps

### 1. Impact-T extraction (`extract_twiss_from_impact.py`)

Reads `particles-571.csv`, opens each `.h5` Impact-T archive, and extracts beam statistics vs `mean_z` (`sigma_x/y/z`, `norm_emit_x/y/z`, `mean_kinetic_energy`, etc.). Streams results row-by-row to CSV (constant memory).

```bash
python extract_twiss_from_impact.py particles-571.csv \
    --output-dir twiss-impact-output \
    --min-alive-particles 90000 \
    --progress-every 200 \
    --resume
```

Outputs in `twiss-impact-output/`:
- `twiss_vs_position.csv` — long format, one row per (sample × z-step). Despite the name, this file contains **emittance and beam-size** stats, not Twiss.
- `endpoint_summary.csv` — one row per sample with final-position values.

Note: although `compute_twiss()` is implemented, Impact stats lack `cov_x__x` / `cov_px__px`, so Twiss columns are not actually produced from this source. Twiss comes from Tao (step 2).

### 2. Tao lattice-optics extraction (`extract_twiss_from_tao.py`)

For each row's Bmad settings, opens a Tao session against the FACET-II lattice and uses `python lat_list` to dump Twiss (`β_a`, `α_a`, `γ_a`, `η_a`, `η'_a`, plus `_b` versions) at all ~1,803 lattice elements. Streams results.

```bash
python extract_twiss_from_tao.py particles-571.csv \
    --output-dir twiss-tao-output \
    --lattice-dir $FACET2_LATTICE \
    --min-alive-particles 90000 \
    --progress-every 200 \
    --resume
```

Outputs in `twiss-tao-output/`:
- `twiss_vs_position.csv` — long format, one row per (sample × element), with columns `ele_name`, `s`, `beta_a`, `alpha_a`, etc.
- `element_reference.csv` — element name and s-position lookup.

### 3. Resolution study (`plot_emittance_resolution_study.py`)

Determines how coarsely `s` can be sampled before peak features (in emittance, sigma, etc.) are missed by more than 5%. Tests candidate spacings `ds ∈ {0.05, 0.02, 0.01, 0.005, 0.002, 0.001}` m on a chosen number of random samples and reports worst-case peak error.

```bash
python plot_emittance_resolution_study.py \
    --input twiss-impact-output/twiss_vs_position.csv \
    --output-dir resolution-study \
    --num-samples 100 \
    --z-min 0.001
```

Outputs:
- `evolution_overview.png` — visual of the 4 quantities along z for one sample
- `resolution_<quantity>.png` — per-quantity overlay of true vs reconstructed at each ds
- `peak_errors.csv` — single-sample peak error table
- `peak_errors_per_sample.csv`, `peak_errors_aggregated.csv`, `peak_errors_worst_case.csv` — multi-sample stats
- `worst_case_vs_ds.png` — worst-case peak error vs ds with 5% threshold line

**Result locked in**: ds = 0.02 m (~210 s-points per sample over the injector region z = 0.001 to ~4.2 m). Justified by worst-case <1.3% peak error across 100 samples.

### 4. Build training dataset (planned, not yet implemented)

`build_training_dataset.py` will:
- Build a common s-grid from `z = 0.001` m to `z = 4.2` m at `ds = 0.02` m
- Interpolate each sample's targets from both Impact and Tao outputs onto that grid
- Join with the 19 machine control parameters from `particles-571.csv`
- Output long-format CSV: `[sample_idx, 19 inputs, s, 4 emit/sigma targets, 4 Twiss targets]`

### 5. Train/val/test split (planned)

Adapt the `split_dataset.py` pattern from `modeling-571-moredata` — 70/15/15 split **by `sample_idx`** (not by row) so all s-points of one simulation stay together. Otherwise the model leaks information between train and val sets.

### 6. Training (planned)

Adapt `train.py` from `modeling-571-moredata`. Simpler than the covariance model — no Cholesky tricks. Plain MLP: 20 inputs → N outputs.

## Key Operational Details

### Alive-particle filter

- Both extraction scripts filter using the precomputed `n_particles_571` column already present in `particles-571.csv`. No need to reopen `.h5` files for this.
- Default threshold is 90,000 (configurable via `--min-alive-particles`).
- Pass `--min-alive-particles 0` to disable.

### Streaming output (memory safety)

Both extraction scripts write each successful sample's rows directly to the output CSV **before processing the next sample**. This means:
- Memory use is constant regardless of dataset size.
- If the script crashes mid-run, all completed samples are preserved on disk.

### Resume support

Both scripts accept `--resume`. On restart:
- They read the existing output CSV's `csv_row_idx` column and build a set of already-processed rows.
- They skip those rows and continue with new ones.
- `sample_idx` continues incrementing from the previous max + 1 to stay unique and sequential.

### Sample alignment between Impact and Tao outputs

Both scripts iterate `particles-571.csv` in row order and apply the same alive-particle filter. As long as both runs use the same `--min-alive-particles` value, their `csv_row_idx` columns will correspond 1:1. Joins between the two outputs should be done on `csv_row_idx`, not `sample_idx` (which is just a sequential counter).

### Why z ≥ 0.001 m in the resolution study

Two Impact samples (`csv_row_idx` 4771, 1637 in the 5k test set) have single-point spikes in `sigma_x/y` at `z ≈ 1 μm`, right at the cathode. These are simulation artifacts on the order of 1 μm wide — finer than any practical s-grid can resolve. Filtering to `z ≥ 1 mm` skips them. The surrogate's training s-range starts at 1 mm anyway.

## Directory Structure

```
modeling-twissparameters/
├── extract_twiss_from_impact.py      # Step 1: Impact-T extraction (emittance, sigma)
├── extract_twiss_from_tao.py         # Step 2: Tao extraction (Twiss)
├── plot_emittance_resolution_study.py # Step 3: choose ds for training s-grid
│
├── twiss-impact-output-5000/         # Small Impact test extraction (5k samples)
│   ├── twiss_vs_position.csv
│   └── endpoint_summary.csv
│
├── twiss-impact-output/              # Full Impact extraction (~55k after alive filter)
├── twiss-tao-output/                 # Full Tao extraction (~55k × 1,803 elements)
│
├── resolution-study/                 # ds study, single sample
├── resolution-study-20/              # ds study, 20 samples
├── resolution-study-100/             # ds study, 100 samples (with cathode artifacts)
└── resolution-study-100-clean/       # ds study, 100 samples (z ≥ 1 mm) — final result
```

## Dataset Sizes (after alive filter `n_particles_571 ≥ 90000`)

| Source CSV | Total rows | With archive / settings | After alive filter |
|---|---|---|---|
| `particles-571.csv` | 64,570 | 63,070 (Impact) / 64,570 (Tao) | **54,987** |

This is the working sample count for both extractions.

## Dependencies

- `torch` (for training; not yet used here)
- `pandas`, `numpy`
- `matplotlib`
- `lume-impact` (Impact-T archive loading)
- `pytao` (lattice optics)
- The `facet2-lattice` repository (Bmad lattice files) — pointed to via `--lattice-dir` or `FACET2_LATTICE` env var.

## Conventions

- **Column naming**: Machine knob columns use simulator parameter names (e.g., `CQ10121:b1_gradient`). Bmad-flavoured copies are prefixed `bmad_` (e.g., `bmad_QA10361`).
- **Units**: All positions in meters (Impact `mean_z` and Tao `s`). Sigma in meters, emittance in meter-radians, Twiss `β` in meters, Twiss `α` dimensionless.
- **`sample_idx` vs `csv_row_idx`**: `csv_row_idx` is the row position in `particles-571.csv` and is the canonical join key between Impact and Tao outputs. `sample_idx` is just an incrementing counter assigned at extraction time.
- **Bmad-attribute mapping** (replicated from `Model_Calibration.utils.simulation_setup.update_bmad_settings`):
  - keys containing `theta0_deg` → `set ele <name> PHI0=<value>/360`
  - keys containing `rf_field_scale` → `set ele <name> VOLTAGE=<value>`
  - keys starting with `Q` → `set ele <key> B1_GRADIENT=<value>`
