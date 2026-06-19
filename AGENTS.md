# AGENTS.md — FACET-II Beam-Evolution Surrogate (Twiss + Beam Parameters)

## What This Codebase Does

This repository builds a **second-generation ML surrogate model** for the FACET-II injector at SLAC. Where the previous model (`modeling-571`) predicts the beam covariance only at screen 571, this model predicts **beam properties as a continuous function of position `s` along the beamline**.

Inputs: 19 machine control parameters + position `s`.
Outputs (target list, all from Impact-T tracking):
- **Beam sizes**: `sigma_x`, `sigma_y`, `sigma_z`
- **Normalized emittances**: `norm_emit_x`, `norm_emit_y`, `norm_emit_z`
- **Geometric emittances**: `emit_geom_x`, `emit_geom_y` (derived from `norm_emit / (βγ)`)
- **Twiss parameters** (derived from Impact second moments): `beta_x`, `alpha_x`, `gamma_x`, `beta_y`, `alpha_y`, `gamma_y`
- **Energy**: `mean_kinetic_energy`

Architecture: `(19 control params, s) → MLP → (target values at that s)` ("architecture B" in earlier discussions).

### Scientific Context

- **Accelerator**: FACET-II electron-beam injector at SLAC.
- **Beam parameters vs s**: A beam's `sigma`, emittance, and Twiss functions evolve continuously through magnets, RF cavities, and drifts. The surrogate captures this full evolution.
- **Single data source**: Impact-T provides particle-tracking statistics (true beam moments). Twiss parameters are computed *from* those moments using Courant–Snyder definitions (see below). We previously experimented with extracting design Twiss from Tao, but that approach was dropped: the surrogate should predict the actual beam Twiss (which is what Impact gives via its second moments), not the design lattice Twiss.
- **Alive-particle filter**: Samples with fewer than 90,000 surviving particles at screen 571 (`n_particles_571 < 90000`) are removed before any analysis. This matches the cut used by the `modeling-571` model and removes ~14% of samples (~9k of ~63k), which would otherwise produce extreme outliers in beam statistics.
- **s-spacing**: A resolution study chose **ds = 0.02 m** (2 cm) as the training-grid spacing. Worst-case peak error <1.3% across 100 random samples, well under the 5% threshold supervisor specified.
- **Note on `s` vs `z`**: For the FACET-II injector (no bends until well past z = 4.2 m), `s` (arc length along the design orbit) and `z` (longitudinal Cartesian) are interchangeable. Impact reports `mean_z`; we treat that as `s` in the dataset.

## Twiss Parameter Definitions

The surrogate predicts Courant–Snyder Twiss parameters $\beta$, $\alpha$, $\gamma$. These describe the shape and orientation of the beam ellipse in phase space.

### Courant–Snyder formulation (Wikipedia / Wiedemann ch. 8)

For a beam with second moments $\langle x^2 \rangle$, $\langle x'^2 \rangle$, $\langle xx' \rangle$:

$$
\langle x^2 \rangle = \epsilon\,\beta, \quad
\langle x'^2 \rangle = \epsilon\,\gamma, \quad
\langle xx' \rangle = -\epsilon\,\alpha
$$

with geometric emittance

$$
\epsilon^2 = \langle x^2 \rangle \langle x'^2 \rangle - \langle xx' \rangle^2
$$

and the constraint

$$
\beta\gamma - \alpha^2 = 1.
$$

### What Impact gives us

Impact-T's `output['stats']` dictionary contains:

| Quantity | Symbol | Impact key |
|---|---|---|
| Beam size | $\sigma_x$ | `sigma_x` |
| Position–momentum cross moment | $\langle x \cdot p_x \rangle$ | `cov_x__px` |
| Normalized emittance | $\epsilon_n$ | `norm_emit_x` |
| Mean kinetic energy | KE | `mean_kinetic_energy` |

Impact does **not** store $\langle x^2 \rangle$ or $\langle p_x^2 \rangle$ directly. But we can derive everything we need from the four columns above.

### Derivation (script: `compute_twiss_from_impact_csv.py`)

Step 1 — relativistic factors from KE:

$$
\gamma_{rel} = 1 + \frac{\text{KE}}{m_e c^2}, \quad
\beta_{rel} = \sqrt{1 - \frac{1}{\gamma_{rel}^2}}, \quad
p_z = \beta_{rel}\,\gamma_{rel}\,m_e c
$$

(electron rest energy $m_e c^2 = 510998.95$ eV; $p_z$ in units of eV/c to match Impact's cov units).

Step 2 — second moments:

$$
\langle x^2 \rangle = \sigma_x^2
$$

$$
\langle xx' \rangle = \frac{\langle x \cdot p_x \rangle}{p_z}
\quad\text{(paraxial: } x' = p_x / p_z\text{)}
$$

Step 3 — convert normalized → geometric emittance:

$$
\epsilon_{geom} = \frac{\epsilon_n}{\beta_{rel}\,\gamma_{rel}}
$$

Step 4 — recover $\langle x'^2 \rangle$ from $\epsilon^2 = \langle x^2\rangle\langle x'^2\rangle - \langle xx'\rangle^2$:

$$
\langle x'^2 \rangle = \frac{\epsilon_{geom}^2 + \langle xx' \rangle^2}{\langle x^2 \rangle}
$$

Step 5 — Twiss parameters:

$$
\beta = \frac{\langle x^2 \rangle}{\epsilon_{geom}}, \quad
\alpha = -\frac{\langle xx' \rangle}{\epsilon_{geom}}, \quad
\gamma = \frac{\langle x'^2 \rangle}{\epsilon_{geom}}
$$

(Same for the y-plane using `sigma_y`, `cov_y__py`, `norm_emit_y`.)

### Verification

The script's output satisfies $\beta\gamma - \alpha^2 = 1$ to within $\sim 10^{-11}$ on all 5.4 M test rows — confirming the derivation is self-consistent with the CS definitions.

## Pipeline Steps

### 1. Impact-T extraction (`extract_twiss_from_impact.py`)

Reads `particles-571.csv`, opens each `.h5` Impact-T archive, and extracts beam statistics vs `mean_z` (`sigma_x/y/z`, `cov_x__px`, `cov_y__py`, `norm_emit_x/y/z`, `mean_kinetic_energy`, etc.). Streams results row-by-row to CSV (constant memory).

```bash
python extract_twiss_from_impact.py particles-571.csv \
    --output-dir twiss-impact-output \
    --min-alive-particles 90000 \
    --progress-every 200 \
    --resume
```

Outputs in `twiss-impact-output/`:
- `twiss_vs_position.csv` — long format, one row per (sample × z-step) with all Impact stats columns.
- `endpoint_summary.csv` — one row per sample with final-position values.

The built-in `compute_twiss()` in this script silently skips because Impact stats lack `cov_x__x` and `cov_px__px`. Twiss is added in step 2.

### 2. Twiss derivation (`compute_twiss_from_impact_csv.py`)

Post-processes the Impact CSV to add geometric emittance and Twiss columns using the equations above. Reads in chunks (constant memory) and writes a new CSV with the extra columns.

```bash
python compute_twiss_from_impact_csv.py \
    --input twiss-impact-output/twiss_vs_position.csv \
    --output twiss-impact-output/twiss_vs_position_with_twiss.csv \
    --chunksize 1000000
```

Adds 8 columns: `emit_geom_x`, `emit_geom_y`, `beta_x`, `alpha_x`, `gamma_x`, `beta_y`, `alpha_y`, `gamma_y`. Runs in ~1–2 minutes for a 5 M-row file; no archive reads.

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
- Interpolate each sample's targets from the Twiss-augmented CSV onto that grid
- Join with the 19 machine control parameters from `particles-571.csv`
- Output long-format CSV: `[sample_idx, csv_row_idx, 19 inputs, s, sigma_x/y/z, norm_emit_x/y, emit_geom_x/y, beta_x/y, alpha_x/y, mean_kinetic_energy]`

### 5. Train/val/test split (planned)

Adapt the `split_dataset.py` pattern from `modeling-571-moredata` — 70/15/15 split **by `sample_idx`** (not by row) so all s-points of one simulation stay together. Otherwise the model leaks information between train and val sets.

### 6. Training (planned)

Adapt `train.py` from `modeling-571-moredata`. Simpler than the covariance model — no Cholesky tricks. Plain MLP: 20 inputs → ~12 outputs.

## Key Operational Details

### Alive-particle filter

- The extraction script filters using the precomputed `n_particles_571` column already present in `particles-571.csv`. No need to reopen `.h5` files for this.
- Default threshold is 90,000 (configurable via `--min-alive-particles`).
- Pass `--min-alive-particles 0` to disable.

### Streaming output (memory safety)

The extraction script writes each successful sample's rows directly to the output CSV **before processing the next sample**. This means:
- Memory use is constant regardless of dataset size.
- If the script crashes mid-run, all completed samples are preserved on disk.

`compute_twiss_from_impact_csv.py` is also streaming (chunked read + append write).

### Resume support

`extract_twiss_from_impact.py` accepts `--resume`. On restart:
- It reads the existing output CSV's `csv_row_idx` column and builds a set of already-processed rows.
- It skips those rows and continues with new ones.
- `sample_idx` continues incrementing from the previous max + 1 to stay unique and sequential.

### Why z ≥ 0.001 m in the resolution study

Two Impact samples (`csv_row_idx` 4771, 1637 in the 5k test set) have single-point spikes in `sigma_x/y` at `z ≈ 1 μm`, right at the cathode. These are simulation artifacts on the order of 1 μm wide — finer than any practical s-grid can resolve. Filtering to `z ≥ 1 mm` skips them. The surrogate's training s-range starts at 1 mm anyway.

### Why s ≤ 4.2 m

Screen PR10571 (the diagnostic the previous model targets) sits at s ≈ 4.2 m. This defines the scope of the *injector* surrogate. Beyond this point the beam enters the linac/BC sections, which are out of scope for this model.

## Dataset Sizes (after alive filter `n_particles_571 ≥ 90000`)

| Source CSV | Total rows | With archive | After alive filter |
|---|---|---|---|
| `particles-571.csv` | 64,570 | 63,070 | **54,987** |

This is the working sample count.

## Dependencies

- `torch` (for training; not yet used here)
- `pandas`, `numpy`
- `matplotlib`
- `lume-impact` (Impact-T archive loading)

The `pytao` and `facet2-lattice` dependencies are no longer needed (Tao extraction was dropped).

## Conventions

- **Column naming**: Machine knob columns use simulator parameter names (e.g., `CQ10121:b1_gradient`). Bmad-flavoured copies are prefixed `bmad_` (e.g., `bmad_QA10361`).
- **Units**:
  - Position: meters (`mean_z`, `s`)
  - Beam size $\sigma$: meters
  - Normalized emittance: meter-radians (units of $\epsilon_n$)
  - Geometric emittance: meter-radians ($\epsilon_n / \beta\gamma$)
  - Twiss $\beta$: meters
  - Twiss $\alpha$: dimensionless
  - Twiss $\gamma$: 1/meter
  - Kinetic energy: eV
- **`sample_idx` vs `csv_row_idx`**: `csv_row_idx` is the row position in `particles-571.csv` (canonical identifier). `sample_idx` is an incrementing counter assigned at extraction time.
- **Electron rest mass**: $m_e c^2 = 510998.95$ eV (used to convert KE → $\gamma_{rel}$).
