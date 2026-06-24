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
- **Note on `s` vs `z`**: For the FACET-II injector (no bends through the L0 linac region), `s` (arc length along the design orbit) and `z` (longitudinal Cartesian) are interchangeable. Impact reports `mean_z`; we treat that as `s` in the dataset.
- **s-range covered**: Impact-T tracking in this dataset runs from the cathode (z ≈ 0) to L0AFEND at **s = 4.127448 m**. Although PR10571 sits at s = 14.232788 m in the Bmad lattice, it is *not* reached by the Impact run — beam state at PR10571 is taken from precomputed columns in `particles-571.csv` (downstream Bmad outputs). The surrogate's grid therefore covers gun → L0AFEND (~4.13 m).

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

### 4. Build training dataset (`build_training_dataset.py`)

Reads the Twiss-augmented CSV and `particles-571.csv`, applies all filters, interpolates targets onto the uniform s-grid, and writes a long-format dataset:

```bash
python build_training_dataset.py \
    --twiss twiss-impact-output/twiss_vs_position_with_twiss.csv \
    --particles particles-571.csv \
    --output dataset.csv
```

Defaults: `--s-min 0.001`, `--s-max 4.13`, `--ds 0.02` (≈ 207 grid points), `--min-alive-particles 90000`.

Filters applied (per sample) in order:

1. **Alive-particle filter** — `n_particles_571 >= 90000` (default; configurable via `--min-alive-particles`, 0 disables). Source: `particles-571.csv`. Pass rate on the working population: 54,987 / 64,570 ≈ 85%.

2. **Input-feature range filter** (`INPUT_RANGES` in script). 17 of the 19 control parameters are bounded; `distgen:VCC` and `impact_VCC_Cal` are kept as model inputs but **not** bounded. Disable with `--no-input-filter`.

   | Control parameter | Lower | Upper |
   |---|---|---|
   | `GUNF:theta0_deg` | −81 | −68 |
   | `GUNF:rf_field_scale` | 4.9 × 10⁷ | 5.2 × 10⁷ |
   | `SOL10111:solenoid_field_scale` | 0.25 | 0.29 |
   | `CQ10121:b1_gradient` | −0.05 | 0.00 |
   | `SQ10122:b1_gradient` | −0.05 | 0.02 |
   | `distgen:t_dist:sigma_t:value` | −0.25 | 2.0 |
   | `distgen:total_charge:value` | 900 | 1050 |
   | `L0AF_phase:theta0_deg` | −10 | 0 |
   | `L0AF_scale:rf_field_scale` | 5.0 × 10⁷ | 5.3 × 10⁷ |
   | `L0BF_phase:theta0_deg` | −10 | 20 |
   | `L0BF_scale:rf_field_scale` | 5.4 × 10⁷ | 6.5 × 10⁷ |
   | `QA10361` | 3.0 | 4.0 |
   | `QA10371` | −4.2 | −3.2 |
   | `QE10425` | 3 | 9 |
   | `QE10441` | −8 | −5 |
   | `QE10511` | 2 | 4 |
   | `QE10525` | −7 | 1 |

   Combined input-filter pass rate on alive samples: **56.1%** (30,867 / 54,987 on the SDF run). Rejects 24,120 samples.

3. **Target-state filters at three screens** (`TARGET_RANGES` in script). Disable with `--no-target-filter`.

   | Screen | s (m) | Source | sigma_x | sigma_y | mean_pz | mean_t |
   |---|---|---|---|---|---|---|
   | **PR10241** | 0.942084 | Interpolated from Impact CSV | ≤ 2 mm | ≤ 2 mm | [6.0, 6.3] MeV/c | [3.17, 3.20] ns |
   | **L0AFEND** | 4.127448 | Interpolated from Impact CSV | ≤ 2 mm | ≤ 2 mm | [98.35, 111.87] MeV/c | [13.80, 13.90] ns |
   | **PR10571** | 14.232788 (lattice) | `particles-571.csv` columns | ≤ 2 mm | ≤ 2 mm | [155, 175] MeV/c | *dropped* |

   Notes:
   - `mean_pz` is computed from `mean_kinetic_energy` via `pz = sqrt((KE + m_e c²)² − (m_e c²)²)`.
   - PR10571 is not reached by Impact, so its `sigma_x`, `sigma_y`, and `mean_kinetic_energy` come from precomputed columns in `particles-571.csv`. `mean_t` at PR10571 is not stored anywhere and the filter is intentionally dropped.
   - SDF-run rejection counts on the 30,867 input-survivors: PR10241 → 5, L0AFEND → 54, PR10571 → 18,912. The 571 cut dominates (~61% of input-survivors); 241 and L0AFEND together remove <0.2%.

**Actual SDF-run yield**: **11,896 samples kept** out of 54,987 alive (**21.6% overall pass rate**), producing **2,450,576 rows** in `dataset.csv` (~206 s-grid points per sample). Full rejection breakdown: `{alive: 0, input: 24120, tgt_241: 5, tgt_L0AFEND: 54, tgt_571: 18912}`.

**Output columns** (34 total): `[sample_idx, csv_row_idx, 19 inputs, s, sigma_x, sigma_y, sigma_z, norm_emit_x, norm_emit_y, emit_geom_x, emit_geom_y, beta_x, alpha_x, beta_y, alpha_y, mean_kinetic_energy]`. One row per (sample × s-grid-point).

**`sample_idx` vs `csv_row_idx`**: `sample_idx` is a sequential counter 0..N over the *kept* samples (useful for stratified splitting). `csv_row_idx` is the original row index in `particles-571.csv` (canonical identifier; preserved for traceability).

**Progress lines** look like:
```
N kept, M rows | skipped: {'alive': a, 'input': i, 'tgt_241': t1, 'tgt_L0AFEND': t2, 'tgt_571': t5, 'other': o}
```
where `kept` is samples written, `M = kept × ~207` s-grid points, and `skipped` are cumulative rejection counts per filter stage.

### 5. Train/val/test split (`split_dataset.py`)

Splits `dataset.csv` 70/15/15 **by `sample_idx`** (not by row) so that all ~206 s-grid rows of one simulation end up in the same split. Splitting by row would put adjacent s-points of the same sample on both sides of the train/val boundary and dramatically overstate generalization.

```bash
python split_dataset.py dataset.csv \
    --train-fraction 0.70 --val-fraction 0.15 --test-fraction 0.15 \
    --seed 42
```

Two-pass streaming for constant memory:
1. **Pass 1**: read only the `sample_idx` column to collect unique IDs (~12k ints).
2. **Pass 2**: read 500k-row chunks; for each row, look up its `sample_idx` in train/val/test ID sets and `to_csv(..., mode="a")` into the matching output file.

Outputs (next to the input by default): `dataset-train.csv`, `dataset-val.csv`, `dataset-test.csv`.

Production split (seed=42): **8,327 / 1,784 / 1,785 samples** → ~1.72 M / 367 k / 368 k rows.

CLI flags: `--seed`, `--output-dir`, `--prefix`, `--chunksize`, `--id-column` (default `sample_idx`).

### 6. Training (`train.py`)

Plain regression MLP. 20 inputs (19 control parameters + `s`) → 12 targets. No Cholesky machinery, no covariance loss — just z-score-normalized L1 regression.

**Architecture** (matches the `modeling-571-moredata` covariance surrogate backbone):

```
Linear(20  → 100), ELU
Linear(100 → 200), ELU, Dropout(0.05)
Linear(200 → 200), ELU, Dropout(0.05)
Linear(200 → 300), ELU, Dropout(0.05)
Linear(300 → 300), ELU, Dropout(0.05)
Linear(300 → 200), ELU, Dropout(0.05)
Linear(200 → 100), ELU, Dropout(0.05)
Linear(100 → 100), ELU
Linear(100 → 100), ELU
Linear(100 →  12)               # output head
```

~316k parameters. The model stores `y_mean`/`y_std` as buffers; `forward()` returns predictions in normalized (z-score) space, and `predict_raw()` is provided for inference in raw units.

**Normalization**: input and output statistics are computed from the **training split only** and saved to `input_transformers.pt` / `output_transformers.pt` for use by `analyze.py` and downstream inference.

**Training recipe**:

| Phase | What happens |
|---|---|
| Base | Up to `--epochs` (default 200) at `--batch-size 256` / `--lr 1e-3`. Adam + `ReduceLROnPlateau(factor=0.5, patience=10)`. Early stop after `--patience` epochs without val improvement (default 40, recommend 15). |
| Finetune stage *k* | Reload best checkpoint. Re-train at progressively smaller batch sizes (`--finetune-batch-sizes 32 8 …`) with `--finetune-lr 1e-4` decayed by `--finetune-lr-decay 0.5` per stage. Each stage uses `ReduceLROnPlateau(patience=5, min_lr=1e-6)`. |

```bash
python train.py \
    --loss l1 --epochs 200 --patience 40 \
    --batch-size 256 --lr 1e-3 \
    --finetune-batch-sizes 32 8 --finetune-epochs-per-stage 100 \
    --finetune-lr 1e-4 --finetune-lr-decay 0.5 \
    --output-dir model-output
```

On SDF: `sbatch gpu_train.sh` (A100, 80 GB, 10 h walltime).

**Outputs** in `--output-dir`:
- `model.pt` — best-val checkpoint (overwritten whenever val_loss improves across any phase)
- `input_transformers.pt` — `{x_mean, x_std, feature_cols}`
- `output_transformers.pt` — `{y_mean, y_std, target_cols}`
- `training_history.csv` — `phase, epoch, train_loss, val_loss, lr` per epoch (across all phases)
- `test_metrics.csv` — per-target `mae` and `mape_percent` evaluated on the test set at the end of training

**Practical lessons from the production run**:

- The **base run** alone converges to val_loss ≈ 0.10 within ~25 epochs and then starts overfitting; if you only care about the base model, set `--patience 15`.
- **Fine-tune stages 1 (bs=32) and 2 (bs=8) do all the real work**: val_loss drops from ~0.10 to ~0.053. This translates to **30–70% reduction in test MAPE** across every target and eliminates the cathode-region "stripes" in scatter plots.
- A **bs=2 third stage is wasteful for this dataset size** (~1.7 M rows): each epoch is ~860k optimizer steps and takes ~35 min on an A100. In our run, stage 3 val_loss bottomed at epoch 1 (0.0535), the plateau scheduler collapsed LR from 2.5e-5 to 3.13e-6 within 19 epochs, and the remaining 281 epochs would have run for ~163 hours of compute with no improvement. **Don't include bs=2** unless you have a much smaller dataset where each "epoch" is cheap.
- L1 loss in normalized target space behaves better than MSE here because several targets (emittances, sigma_z) have small dynamic range and L1 keeps gradients well-scaled across all 12 outputs.

### 7. Analysis (`analyze.py`)

Loads the trained model + transformers from `--model-dir`, runs the test CSV through it, and produces a metrics file plus 8 figures.

```bash
python analyze.py \
    --model-dir model-output \
    --test-csv dataset-test.csv \
    --output-dir analysis
```

| Output | Description |
|---|---|
| `test_metrics.csv` | Per-target MAE, RMSE, MAPE (robust: ignores rows where \|true\| < 1% of column std) |
| `training_curve.png` | Train/val loss vs cumulative epoch, log-y, color-coded by phase from `training_history.csv` |
| `mae_per_target.png` | Bar chart of MAE per target (raw units, log y) |
| `mape_per_target.png` | Bar chart of MAPE % per target with overall-mean dashed line |
| `scatter_pred_vs_true.png` | 12-panel scatter, per-panel R² computed on the 1–99 percentile inlier window |
| `per_sample_overlay.png` | True vs predicted line plot for the first ~1500 test rows (sample order) |
| `per_sample_zoomed.png` | Same as above but with 5–95 percentile y-zoom, dot version |
| `sorted_by_magnitude.png` | Rows sorted by \|true\|: useful for spotting systematic bias across the range |
| `evolution_curves.png` | 5 random test samples plotted as `target vs s`; true solid, predicted dashed, one color per sample |

Diagnostic flags: `--skip-scatter`, `--skip-overlay`, `--skip-sorted`, `--skip-evolution`, `--overlay-max-samples N`, `--evolution-num-samples N`.

**Production headline (fine-tuned model, 1,785 test samples)**:

| Target | MAE | MAPE | R² |
|---|---:|---:|---:|
| `sigma_x` | 9.23e-6 m | 1.44% | 0.999 |
| `sigma_y` | 8.87e-6 m | 1.47% | 0.999 |
| `sigma_z` | 5.98e-6 m | 0.96% | 0.868 |
| `norm_emit_x` | 2.63e-7 m·rad | 3.20% | 1.000 |
| `norm_emit_y` | 2.38e-7 m·rad | 3.33% | 1.000 |
| `emit_geom_x` | 1.45e-8 m·rad | 3.34% | 1.000 |
| `emit_geom_y` | 1.42e-8 m·rad | 3.29% | 1.000 |
| `beta_x` | 0.237 m | 3.20% | 0.998 |
| `beta_y` | 0.201 m | 3.69% | 0.998 |
| `alpha_x` | 0.452 | 21.2% | 0.968 |
| `alpha_y` | 0.411 | 23.4% | 0.969 |
| `mean_kinetic_energy` | 4.25e5 eV | 1.19% | 1.000 |

- `alpha_x/y` MAPE is inflated because alpha crosses zero (range ≈ [−10, +13]); R² is the honest signal.
- `sigma_z` R² = 0.868 looks low but MAPE = 0.96% is the lowest in the table — its dynamic range across the beamline is very narrow (~5.7e−4 to ~6.8e−4 m), so tiny absolute errors eat most of the variance. Practically fine.

### 8. Inference & LUME-Torch Export (`infer_beam_evolution.py`)

Runs inference on a CSV and exports LUME-Torch YAML model files for deployment. Validates both sim-parameter and machine-PV input spaces. Includes `value_range` (training-data min/max) for each input variable in the YAML.

```bash
python infer_beam_evolution.py \
    --model-dir model-output-100h \
    --input-csv dataset-test.csv \
    --train-csv dataset-train.csv \
    --output-dir inference-output
```

Outputs:
- `inference-output/predictions.csv` — flat predictions with true targets
- `inference-output/predictions.npy` — raw (N, 12) prediction array
- `lumetorchyaml-sim/` — LUME-Torch model (sim-parameter + s inputs)
- `lumetorchyaml-machine/` — LUME-Torch model (machine PV + s inputs)

**Architecture of the LUME model pipeline**:

- **Sim model**: `[19 sim params, s]` → `AffineInputTransform` (z-score normalization) → `BeamEvolutionModel` → `OutputDenormTransform` (z-score denorm) → 12 targets
- **Machine model**: `[19 PVs, s]` → `PVToSimWithS` (affine PV→sim on channels 0..18, pass-through on s) → `AffineInputTransform` (z-score normalization) → `BeamEvolutionModel` → `OutputDenormTransform` → 12 targets

**Custom modules** (in `lume_model_utils.py`):
- `OutputDenormTransform(y_mean, y_std)` — applies `pred * y_std + y_mean`
- `PVToSimWithS(pv_to_sim_transform)` — wraps `AffineInputTransform` for the 19 PV channels and passes `s` (index 19) unchanged

**value_range**: Each input variable in the exported YAML includes `value_range: [min, max]` computed from the training split. For the machine model, sim-parameter ranges are first transformed via `sim_to_machine_array()` to get PV-unit ranges.

**Validation**: The script asserts `np.allclose(direct_model_output, lume_model_output, rtol=1e-5, atol=1e-5)` for both input spaces (including float32 roundtrip through PV↔sim conversion for the machine model).

### 9. Package Model Update

Copy LUME model files into the deployable `facet2-model-twissparameters` package:

```bash
cp -r lumetorchyaml-sim/ ../facet2-model-twissparameters/facet2_inj_ml_model_twiss/resources/lumetorchyaml-sim/
cp -r lumetorchyaml-machine/ ../facet2-model-twissparameters/facet2_inj_ml_model_twiss/resources/lumetorchyaml-machine/
cd ../facet2-model-twissparameters && pip install -e . && cd -
```

Usage:
```python
from facet2_inj_ml_model_twiss import load_model

model = load_model()           # machine PVs + s (default)
model = load_model("sim")      # sim parameters + s

result = model.evaluate({"QUAD:IN10:121:BCTRL": 0.022, ..., "s": 2.0})
# Returns: sigma_x, sigma_y, ..., mean_kinetic_energy (12 scalar outputs)
```

The package lives at `../facet2-model-twissparameters/` with:
- `facet2_inj_ml_model_twiss/` — Python package
- `facet2_inj_ml_model_twiss/resources/` — LUME YAML + model artifacts
- `tests/test_model_benchmark.py` — regression tests (12 tests covering loading, evaluation, PV mapping)

Run tests:
```bash
cd ../facet2-model-twissparameters
pip install -e ".[test]"
pytest tests/ -v
```

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

### Why s ≤ 4.13 m

Impact-T tracking in this dataset ends at L0AFEND (s = 4.127448 m, the exit of the L0A accelerating section). PR10571 — the diagnostic the previous covariance model targets — sits at s = 14.232788 m in the Bmad lattice and is *not* reached by this Impact run. The surrogate therefore models beam evolution over gun → L0AFEND. Beam state at PR10571 is available only via precomputed columns in `particles-571.csv` and is used solely as a filter, not a training target.

Screen s-positions (from the Impact lattice in any `82.h5`-style archive):

| Screen | s (m) | Notes |
|---|---|---|
| PR10241 | 0.942084 | Early diagnostic; quantities interpolated from Impact CSV |
| L0AFEND | 4.127448 | End of Impact tracking; quantities interpolated |
| PR10571 | 14.232788 | Lattice position only; not reached by Impact |

## Dataset Sizes (after alive filter `n_particles_571 ≥ 90000`)

| Source CSV | Total rows | With archive | After alive filter |
|---|---|---|---|
| `particles-571.csv` | 64,570 | 63,070 | **54,987** |

This is the working sample count.

## Key Outputs & Transformations

| Artifact | Description |
|----------|-------------|
| `model.pt` | Trained PyTorch model weights (BeamEvolutionModel) |
| `input_transformers.pt` | Input standardization (x_mean, x_std, feature_cols) |
| `output_transformers.pt` | Output standardization (y_mean, y_std, target_cols) |
| `training_history.csv` | Per-epoch train_loss, val_loss, lr, phase |
| `test_metrics.csv` | Per-target MAE and MAPE on test set |
| LUME YAML files | lume-torch model descriptors for deployment (sim & machine input spaces) |

**Transform chain (inference)**:
1. Raw inputs → [PV→sim affine if machine inputs] → z-score normalize (subtract x_mean, divide by x_std)
2. Model forward pass → z-score normalized predictions
3. Destandardize: `pred * y_std + y_mean` → 12 physical-unit targets

## Directory Structure

```
modeling-twissparameters/
├── extract_twiss_from_impact.py        # Step 1: extract beam stats from .h5 archives
├── compute_twiss_from_impact_csv.py    # Step 2: derive Twiss from Impact CSV
├── plot_emittance_resolution_study.py  # Step 3: s-grid resolution study
├── build_training_dataset.py           # Step 4: filter + interpolate → dataset.csv
├── split_dataset.py                    # Step 5: train/val/test split by sample_idx
├── train.py                            # Step 6: train the MLP
├── analyze.py                          # Step 7: test-set evaluation + plots
├── infer_beam_evolution.py             # Step 8: inference + LUME export
├── lume_model_utils.py                 # Custom LUME-Torch transforms
├── pv_mapping.py                       # Machine PV ↔ sim parameter affine mapping
├── gpu_train.sh                        # SLURM job script
│
├── dataset.csv                         # Combined inputs + targets (2.45M rows)
├── dataset-train.csv / -val.csv / -test.csv
│
├── model-output-100h/                  # Trained model checkpoint & transformers
│   ├── model.pt
│   ├── input_transformers.pt
│   ├── output_transformers.pt
│   └── training_history.csv
│
├── analysis-1/                         # Base-only model analysis
├── analysis-2/                         # Fine-tuned model analysis (production)
│
├── lumetorchyaml-sim/                  # LUME-Torch model (sim inputs)
├── lumetorchyaml-machine/              # LUME-Torch model (machine PV inputs)
└── inference-output/                   # Inference result CSVs
```

## Dependencies

- `torch`
- `pandas`, `numpy`
- `matplotlib`
- `lume-impact` (Impact-T archive loading)
- `lume-torch` (LUME model packaging)
- `botorch` (`AffineInputTransform`)

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
