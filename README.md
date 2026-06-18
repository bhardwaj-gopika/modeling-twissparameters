# modeling-twissparameters

ML surrogate model for the FACET-II injector that predicts beam properties as a function of position `s` along the beamline.

## What it predicts

Given **19 machine control parameters** (quad gradients, RF phases, gun settings, solenoid, etc.) and a beamline position `s`, the model predicts:

- **Beam size**: `sigma_x`, `sigma_y`, `sigma_z`
- **Normalized emittance**: `norm_emit_x`, `norm_emit_y`
- **Twiss parameters**: `β_a`, `α_a`, `β_b`, `α_b`
- **Energy**: `mean_kinetic_energy` (and later energy spread)

Replaces a slow physics simulation (Impact-T particle tracking, minutes per run) with a neural network that runs in milliseconds.

## Why two data sources

| Source | Provides | Method |
|---|---|---|
| **Impact-T** | beam size, emittance, energy | Particle-tracking simulation, ~10⁵ particles |
| **Tao** | Twiss (β, α) | Analytic lattice optics, no particles |

Impact gives the true beam moments. Tao gives the design Twiss functions for that magnet configuration. Combined, you get a complete picture of the beam evolution.

## Pipeline at a glance

1. **Extract** beam stats vs z from Impact-T archives → `twiss-impact-output/twiss_vs_position.csv`
2. **Extract** Twiss vs s from Tao lattice → `twiss-tao-output/twiss_vs_position.csv`
3. **Resolution study** to pick s-grid spacing → chose **ds = 0.02 m** (~210 points per sample, worst-case peak error <1.3%)
4. **Build training dataset** (planned): merge both sources, interpolate to common s-grid, join with control parameters
5. **Train + evaluate** (planned)

## Quick start

### Extract Impact-T stats
```bash
python extract_twiss_from_impact.py particles-571.csv \
    --output-dir twiss-impact-output \
    --min-alive-particles 90000 \
    --resume
```

### Extract Tao Twiss
```bash
python extract_twiss_from_tao.py particles-571.csv \
    --output-dir twiss-tao-output \
    --lattice-dir $FACET2_LATTICE \
    --min-alive-particles 90000 \
    --resume
```

Both scripts **stream output to CSV** (constant memory) and support **`--resume`** (skip already-processed samples on restart).

### Run the resolution study
```bash
python plot_emittance_resolution_study.py \
    --input twiss-impact-output/twiss_vs_position.csv \
    --output-dir resolution-study \
    --num-samples 100 \
    --z-min 0.001
```

## Key design decisions

- **Alive-particle filter**: Samples with fewer than 90,000 surviving particles at screen 571 are removed (matches the cut from the `modeling-571-moredata` model). Drops ~14% of samples (63k → 55k).
- **s-grid spacing**: ds = 0.02 m, chosen by a study that compared worst-case peak error across 100 random samples for spacings from 5 cm to 1 mm. 2 cm passes the 5% threshold with comfortable margin (worst case 1.3%).
- **z-range cutoff**: Resolution study and training use `z ≥ 0.001 m` to skip cathode-region simulation artifacts (single-point sigma spikes at z ≈ 1 μm).
- **Sample alignment**: Both extractions iterate `particles-571.csv` in the same order and apply the same alive filter, so their `csv_row_idx` values match 1:1 for joining.

## Files

| File | Purpose |
|---|---|
| `extract_twiss_from_impact.py` | Load Impact-T archives, extract beam stats vs z |
| `extract_twiss_from_tao.py` | Run Tao with each sample's settings, extract Twiss vs s |
| `plot_emittance_resolution_study.py` | Pick the s-grid spacing |

See [AGENTS.md](AGENTS.md) for detailed pipeline notes, conventions, and operational details.

## Dependencies

- Python ≥ 3.10
- `lume-impact`, `pytao`, `pandas`, `numpy`, `matplotlib`
- `facet2-lattice` repository (path set via `--lattice-dir` or `FACET2_LATTICE` env var)
