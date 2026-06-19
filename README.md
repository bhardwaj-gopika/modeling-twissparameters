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
4. **Build training dataset** (planned): interpolate to common s-grid, join with control parameters
5. **Train + evaluate** (planned)

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

## Key design decisions

- **Twiss from Impact (not Tao)**: The surrogate predicts the *actual beam* Twiss derived from Impact's second moments — not design-lattice Twiss. This matches what the beam physically has at each s, including space-charge and tracking effects.
- **Alive-particle filter**: Samples with fewer than 90,000 surviving particles at screen 571 are removed (matches the cut from the `modeling-571-moredata` model). Drops ~14% of samples (63k → 55k).
- **s-grid spacing**: ds = 0.02 m, chosen by a study that compared worst-case peak error across 100 random samples for spacings from 5 cm to 1 mm. 2 cm passes the 5% threshold with comfortable margin (worst case 1.3%).
- **z-range cutoff**: Resolution study and training use `z ≥ 0.001 m` to skip cathode-region simulation artifacts (single-point sigma spikes at z ≈ 1 μm), and `z ≤ 4.2 m` (screen PR10571 location) as the injector boundary.

## Files

| File | Purpose |
|---|---|
| `extract_twiss_from_impact.py` | Load Impact-T archives, extract beam stats vs z |
| `compute_twiss_from_impact_csv.py` | Derive Twiss + geometric emittance from Impact CSV |
| `plot_emittance_resolution_study.py` | Pick the s-grid spacing |

See [AGENTS.md](AGENTS.md) for detailed pipeline notes, Twiss derivation, conventions, and operational details.

## Dependencies

- Python ≥ 3.10
- `lume-impact`, `pandas`, `numpy`, `matplotlib`
