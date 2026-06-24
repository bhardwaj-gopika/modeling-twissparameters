# modeling-twissparameters

ML surrogate model for the FACET-II injector that predicts beam properties as a function of position `s` along the beamline.

## What it predicts

Given **19 machine control parameters** (quad gradients, RF phases, gun settings, solenoid, etc.) and a beamline position `s`, the model predicts:

- **Beam size**: `sigma_x`, `sigma_y`, `sigma_z`
- **Normalized emittance**: `norm_emit_x`, `norm_emit_y`
- **Geometric emittance**: `emit_geom_x`, `emit_geom_y`
- **Twiss parameters** (Courant–Snyder): `beta_x`, `alpha_x`, `beta_y`, `alpha_y`
- **Energy**: `mean_kinetic_energy`

Replaces a slow physics simulation (Impact-T particle tracking, minutes per run) with a neural network that runs in milliseconds.

## How Twiss is obtained

All targets come from **Impact-T particle tracking**. Twiss parameters are derived from Impact's second moments using Courant–Snyder definitions:

$$
\beta = \frac{\langle x^2 \rangle}{\epsilon}, \quad
\alpha = -\frac{\langle xx' \rangle}{\epsilon}, \quad
\gamma = \frac{\langle x'^2 \rangle}{\epsilon}, \quad
\epsilon^2 = \langle x^2 \rangle\langle x'^2 \rangle - \langle xx' \rangle^2
$$

Impact stores `sigma_x` (= $\sqrt{\langle x^2\rangle}$), `cov_x__px` (= $\langle x \cdot p_x \rangle$), and `norm_emit_x` (= $\epsilon_n$). We convert these to the CS quantities via:

- $\langle xx' \rangle = \langle x \cdot p_x \rangle / p_z$ (paraxial: $x' = p_x/p_z$)
- $\epsilon_{geom} = \epsilon_n / (\beta_{rel}\gamma_{rel})$
- $p_z = \beta_{rel}\gamma_{rel}\,m_e c$ with $\gamma_{rel} = 1 + \text{KE}/(m_e c^2)$

The identity $\beta\gamma - \alpha^2 = 1$ is verified to ~$10^{-11}$ on output. Full derivation: [AGENTS.md](AGENTS.md).

## Pipeline at a glance

1. **Extract** beam stats vs z from Impact-T archives → `twiss-impact-output/twiss_vs_position.csv`
2. **Derive Twiss** by post-processing the CSV → `twiss-impact-output/twiss_vs_position_with_twiss.csv`
3. **Resolution study** to pick s-grid spacing → chose **ds = 0.02 m** (~210 points per sample, worst-case peak error <1.3%)
4. **Build training dataset**: filter + interpolate → `dataset.csv` (~2.45 M rows, 11,896 samples)
5. **Split** by `sample_idx` → `dataset-train.csv` / `-val.csv` / `-test.csv` (70/15/15)
6. **Train** the 20→12 MLP → `model-output/`
7. **Evaluate** on the test set → `analysis-*/`

## Quick start

### 1. Extract Impact-T stats
```bash
python extract_twiss_from_impact.py particles-571.csv \
    --output-dir twiss-impact-output \
    --min-alive-particles 90000 \
    --resume
```

### 2. Add Twiss columns
```bash
python compute_twiss_from_impact_csv.py \
    --input twiss-impact-output/twiss_vs_position.csv \
    --output twiss-impact-output/twiss_vs_position_with_twiss.csv
```

Both scripts **stream output to CSV** (constant memory). The extraction script supports **`--resume`** (skip already-processed samples on restart).

### 3. Run the resolution study
```bash
python plot_emittance_resolution_study.py \
    --input twiss-impact-output/twiss_vs_position_with_twiss.csv \
    --output-dir resolution-study \
    --num-samples 100 \
    --z-min 0.001
```

### 4. Build the training dataset
```bash
python build_training_dataset.py \
    --twiss twiss-impact-output/twiss_vs_position_with_twiss.csv \
    --particles particles-571.csv \
    --output dataset.csv \
    --progress-every 1000
```

Applies the full filter chain and interpolates each surviving sample onto the uniform 2 cm s-grid:

| Stage | Filter | Source | Typical pass rate |
|---|---|---|---|
| 1 | `n_particles_571 ≥ 90,000` | `particles-571.csv` | 85% (54,987 / 64,570) |
| 2 | 17 control parameters in supervisor-specified ranges | `particles-571.csv` | 56% (30,867 / 54,987) |
| 3a | PR10241 (s = 0.942 m): sigma_x/y ≤ 2 mm, mean_pz ∈ [6.0, 6.3] MeV/c, mean_t ∈ [3.17, 3.20] ns | Interpolated from Impact CSV | >99.9% (5 rejected) |
| 3b | L0AFEND (s = 4.127 m): sigma_x/y ≤ 2 mm, mean_pz ∈ [98.4, 111.9] MeV/c, mean_t ∈ [13.80, 13.90] ns | Interpolated from Impact CSV | 99.8% (54 rejected) |
| 3c | PR10571: sigma_x/y ≤ 2 mm, mean_pz ∈ [155, 175] MeV/c | `particles-571.csv` columns (Impact tracking doesn't reach 571) | 39% (18,912 rejected) |

Net SDF-run yield: **11,896 samples kept** (21.6% overall) producing **2,450,576 rows** in `dataset.csv` (~206 s-grid points per sample).

Output schema (34 cols): `[sample_idx, csv_row_idx, 19 inputs, s, sigma_x, sigma_y, sigma_z, norm_emit_x, norm_emit_y, emit_geom_x, emit_geom_y, beta_x, alpha_x, beta_y, alpha_y, mean_kinetic_energy]`.

Diagnostic flags: `--no-input-filter`, `--no-target-filter`, `--min-alive-particles 0`.

### 5. Split into train / val / test (by `sample_idx`)
```bash
python split_dataset.py dataset.csv \
    --train-fraction 0.70 --val-fraction 0.15 --test-fraction 0.15 \
    --seed 42
```

Splits on **`sample_idx`** (not row), so all ~206 s-grid rows of one simulation stay in the same split — prevents adjacent-s leakage between train and val. Two-pass streaming: pass 1 scans the ID column, pass 2 appends rows in 500k-row chunks. Constant memory.

Result: 8,327 / 1,784 / 1,785 samples → ~1.72 M / 367k / 368k rows.

### 6. Train the surrogate
```bash
python train.py \
    --loss l1 --epochs 200 --patience 40 \
    --batch-size 256 --lr 1e-3 \
    --finetune-batch-sizes 32 8 \
    --finetune-epochs-per-stage 100 \
    --finetune-lr 1e-4 --finetune-lr-decay 0.5 \
    --output-dir model-output
```

Or on SDF: `sbatch gpu_train.sh`.

Architecture (matches the `modeling-571-moredata` covariance surrogate backbone):

```
Linear(20 → 100), ELU
Linear(100 → 200), ELU, Dropout(0.05)
Linear(200 → 200), ELU, Dropout(0.05)
Linear(200 → 300), ELU, Dropout(0.05)
Linear(300 → 300), ELU, Dropout(0.05)
Linear(300 → 200), ELU, Dropout(0.05)
Linear(200 → 100), ELU, Dropout(0.05)
Linear(100 → 100), ELU
Linear(100 → 100), ELU
Linear(100 → 12)               # output head
```

~316k parameters. Inputs and targets are both z-score normalized using statistics computed on the **training split only**.

**Training recipe**:
- **Base run**: 200 epochs max, L1 loss in normalized space, Adam (lr=1e-3), ReduceLROnPlateau (factor 0.5, patience 10), early stopping (patience 40).
- **Staged fine-tuning** (optional): after the base run, reload best checkpoint and re-train at progressively smaller batch sizes (default 32 → 8) with `lr=1e-4` and decay 0.5 per stage. Each stage uses ReduceLROnPlateau (patience 5, min_lr 1e-6).

Outputs in `--output-dir`:
- `model.pt`, `input_transformers.pt`, `output_transformers.pt`
- `training_history.csv` (per-epoch train/val loss, lr, and phase tag)
- `test_metrics.csv` (per-target MAE and MAPE on the held-out test set)

**Practical guidance** — see [AGENTS.md](AGENTS.md) for the long version:
- The base run alone reaches val_loss ≈ 0.10 and is usually overfitting by epoch ~25.
- **Fine-tune stages 1 (bs=32) and 2 (bs=8) deliver ≈30–70% MAPE reduction across all targets**. They take ≈1–2 hours each on an A100.
- **A bs=2 third stage adds essentially nothing for this dataset size** (~1.7M rows): the plateau scheduler collapses LR within 10–20 epochs and val_loss stops improving. Skip it.

### 7. Evaluate the model
```bash
python analyze.py \
    --model-dir model-output \
    --test-csv dataset-test.csv \
    --output-dir analysis
```

Produces in `analysis/`:

| File | What it shows |
|---|---|
| `test_metrics.csv` | Per-target MAE, RMSE, MAPE on the test set |
| `training_curve.png` | Train/val loss vs epoch, log-scaled, color-coded by phase (base / finetune) |
| `mae_per_target.png` | Bar chart of MAE per target (raw units, log y) |
| `mape_per_target.png` | Bar chart of MAPE % per target with overall mean line |
| `scatter_pred_vs_true.png` | 12-panel scatter with per-panel R² (axes clipped to 1–99 percentile) |
| `per_sample_overlay.png` | True vs predicted curves over the first ~1500 test rows |
| `per_sample_zoomed.png` | Same as above but with 5–95 percentile y-zoom (dot version) |
| `sorted_by_magnitude.png` | Rows sorted by \|true\|, helps spot systematic bias across the range |
| `evolution_curves.png` | For 5 random test samples: true (solid) vs predicted (dashed) per target vs `s` |

Diagnostic flags: `--skip-scatter`, `--skip-overlay`, `--skip-sorted`, `--skip-evolution`, `--evolution-num-samples N`.

## Headline results (production SDF run)

Fine-tuned model (`base + bs=32 + bs=8`, 200 + 100 + 100 epochs) on the 1,785-sample held-out test set:

| Target | MAE (raw units) | MAPE | R² |
|---|---:|---:|---:|
| `sigma_x` | 9.23×10⁻⁶ m | 1.44% | 0.999 |
| `sigma_y` | 8.87×10⁻⁶ m | 1.47% | 0.999 |
| `sigma_z` | 5.98×10⁻⁶ m | 0.96% | 0.868 |
| `norm_emit_x` | 2.63×10⁻⁷ m·rad | 3.20% | 1.000 |
| `norm_emit_y` | 2.38×10⁻⁷ m·rad | 3.33% | 1.000 |
| `emit_geom_x` | 1.45×10⁻⁸ m·rad | 3.34% | 1.000 |
| `emit_geom_y` | 1.42×10⁻⁸ m·rad | 3.29% | 1.000 |
| `beta_x` | 0.237 m | 3.20% | 0.998 |
| `beta_y` | 0.201 m | 3.69% | 0.998 |
| `alpha_x` | 0.452 | 21.2%† | 0.968 |
| `alpha_y` | 0.411 | 23.4%† | 0.969 |
| `mean_kinetic_energy` | 4.25×10⁵ eV | 1.19% | 1.000 |

† `alpha` crosses zero, so MAPE blows up near zero-crossings; R² is the honest signal here.

**Improvement from fine-tuning** vs. base-only:
- sigma_x/y: −~29%
- norm_emit_x/y: −~34%
- emit_geom_y: −71%
- beta_x/y: −~61%
- mean_kinetic_energy: −65%
- Stripes near small values in `norm_emit_*`, `emit_geom_*`, and `beta_*` scatters are eliminated.

On `evolution_curves.png`, true (solid) and predicted (dashed) lines are visually indistinguishable for sigma_x/y, the cathode-region emittance spike, the post-L0A beta growth, and the kinetic energy ramp.

## Key design decisions

- **Twiss from Impact (not Tao)**: The surrogate predicts the *actual beam* Twiss derived from Impact's second moments — not design-lattice Twiss. This matches what the beam physically has at each s, including space-charge and tracking effects.
- **Alive-particle filter**: Samples with fewer than 90,000 surviving particles at screen 571 are removed (matches the cut from the `modeling-571-moredata` model). Drops ~14% of samples (63k → 55k).
- **s-grid spacing**: ds = 0.02 m, chosen by a study that compared worst-case peak error across 100 random samples for spacings from 5 cm to 1 mm. 2 cm passes the 5% threshold with comfortable margin (worst case 1.3%).
- **z-range cutoff**: Resolution study and training use `z ≥ 0.001 m` to skip cathode-region simulation artifacts (single-point sigma spikes at z ≈ 1 μm), and `z ≤ 4.13 m` (L0AFEND — the end of Impact-T tracking in this dataset). PR10571 itself sits at s = 14.23 m and is not reached by Impact; its beam state is used only as a filter via precomputed columns in `particles-571.csv`.
- **Filtering**: A multi-stage filter (alive-particle count, supervisor-specified input-feature ranges, and beam-state ranges at PR10241 / L0AFEND / PR10571) is applied by `build_training_dataset.py`. See [AGENTS.md](AGENTS.md) for details.

## Files

| File | Purpose |
|---|---|
| `extract_twiss_from_impact.py` | Load Impact-T archives, extract beam stats vs z |
| `compute_twiss_from_impact_csv.py` | Derive Twiss + geometric emittance from Impact CSV |
| `plot_emittance_resolution_study.py` | Pick the s-grid spacing (chose ds = 0.02 m) |
| `build_training_dataset.py` | Apply filters, interpolate onto 2 cm s-grid, join with controls |
| `split_dataset.py` | Stream-split `dataset.csv` 70/15/15 by `sample_idx` |
| `train.py` | Train the 20→12 MLP (base + optional staged fine-tuning) |
| `analyze.py` | Evaluate on test set; produce metrics CSV + 8 PNG figures |
| `infer_beam_evolution.py` | Inference + LUME-Torch YAML export (sim & machine PV input spaces) |
| `lume_model_utils.py` | Custom transforms for LUME-Torch (OutputDenormTransform, PVToSimWithS) |
| `pv_mapping.py` | Affine mapping between machine PVs and simulator parameters |
| `gpu_train.sh` | SLURM script for training on SDF (A100) |

See [AGENTS.md](AGENTS.md) for detailed pipeline notes, Twiss derivation, conventions, and operational details.

## Inference & LUME-Torch Deployment

### 8. Run inference and export LUME-Torch models
```bash
python infer_beam_evolution.py \
    --model-dir model-output-100h \
    --input-csv dataset-test.csv \
    --train-csv dataset-train.csv \
    --output-dir inference-output
```

This script:
1. Loads the trained model and applies it to the input CSV (auto-detects sim vs machine-PV columns).
2. Exports LUME-Torch YAML model files with `value_range` (min/max from training data) for both input spaces.
3. Validates both LUME models match direct model output (assertion on `np.allclose`).

Outputs:
- `inference-output/predictions.csv` — predicted values with true targets (if available)
- `inference-output/predictions.npy` — raw prediction array
- `lumetorchyaml-sim/` — LUME model accepting 19 sim parameters + `s`
- `lumetorchyaml-machine/` — LUME model accepting 19 machine PVs + `s`

### Using the deployed model

The packaged model lives in `../facet2-model-twissparameters/`:

```python
from facet2_inj_ml_model_twiss import load_model

model = load_model()           # machine PVs + s
model = load_model("sim")      # sim parameters + s

result = model.evaluate({
    "QUAD:IN10:121:BCTRL": 0.022,
    "KLYS:LI10:21:AMPL": 40.0,
    # ... 17 more PVs ...
    "s": 2.0,
})
# result keys: sigma_x, sigma_y, sigma_z, norm_emit_x, norm_emit_y,
#   emit_geom_x, emit_geom_y, beta_x, alpha_x, beta_y, alpha_y, mean_kinetic_energy
```

Install the package:
```bash
cd ../facet2-model-twissparameters && pip install -e .
```

### PV ↔ sim mapping

The machine model applies an affine PV→sim transform on the 19 control channels and passes `s` through unchanged. The mapping is defined in `pv_mapping.py` with per-channel `sim_scaling` and `sim_offset`.

## Dependencies

- Python ≥ 3.10
- `lume-impact`, `pandas`, `numpy`, `matplotlib`, `torch`
- For inference/deployment: `lume-torch`, `botorch`
