"""Run beam-evolution inference and export LUME-Torch models.

The script accepts either simulator-parameter columns or machine-facing PV-unit
columns (auto-detected or forced via --input-space).  Machine inputs are first
mapped into simulator parameter space before normalization.

The 20th input feature, `s` (beamline position), is always passed through
unchanged — it has no PV mapping.

Outputs 12 scalar beam-evolution targets:
  sigma_x, sigma_y, sigma_z, norm_emit_x, norm_emit_y,
  emit_geom_x, emit_geom_y, beta_x, alpha_x, beta_y, alpha_y,
  mean_kinetic_energy

LUME-Torch YAML model files are exported for deployment in both
sim-parameter and machine-PV input spaces.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lume_torch.variables import TorchScalarVariable
from lume_torch.models import TorchModel, TorchModule
from botorch.models.transforms.input import AffineInputTransform

from train import build_model, TARGET_COLUMNS
from pv_mapping import (
    build_pv_to_sim_transform,
    machine_input_names,
    machine_to_sim_array,
    sim_to_machine_array,
    ordered_pv_mapping,
)
from lume_model_utils import OutputDenormTransform, PVToSimWithS


# ── Model loading ────────────────────────────────────────────────────────────

def load_model_and_transformers(model_dir: Path, device: torch.device):
    input_tr = torch.load(model_dir / "input_transformers.pt", map_location="cpu")
    output_tr = torch.load(model_dir / "output_transformers.pt", map_location="cpu")

    feature_cols = list(input_tr["feature_cols"])
    target_cols = list(output_tr["target_cols"])
    y_mean = output_tr["y_mean"].to(device)
    y_std = output_tr["y_std"].to(device)

    model = build_model(len(feature_cols), len(target_cols), y_mean=y_mean, y_std=y_std)
    model.load_state_dict(
        torch.load(model_dir / "model.pt", weights_only=True, map_location=device)
    )
    model.to(device)
    model.eval()
    return model, input_tr, output_tr


# ── Input space detection ────────────────────────────────────────────────────

def resolve_input_space(df: pd.DataFrame, feature_cols: list[str], requested_space: str):
    """Determine whether the CSV has sim-parameter or machine-PV columns.

    The feature_cols list includes 's', which is always present.  Only the
    first 19 (non-s) columns participate in the sim/PV decision.
    """
    sim_cols_no_s = [c for c in feature_cols if c != "s"]
    pv_cols_no_s = machine_input_names(sim_cols_no_s)

    has_sim = all(c in df.columns for c in sim_cols_no_s)
    has_pv = all(c in df.columns for c in pv_cols_no_s)
    has_s = "s" in df.columns

    if not has_s:
        raise SystemExit("Input CSV is missing the 's' (position) column.")

    if requested_space == "sim":
        if not has_sim:
            missing = [c for c in sim_cols_no_s if c not in df.columns]
            raise SystemExit("Input CSV is missing sim columns: " + ", ".join(missing))
        return "sim", sim_cols_no_s, pv_cols_no_s
    if requested_space == "pv":
        if not has_pv:
            missing = [c for c in pv_cols_no_s if c not in df.columns]
            raise SystemExit("Input CSV is missing PV columns: " + ", ".join(missing))
        return "pv", sim_cols_no_s, pv_cols_no_s
    # auto
    if has_pv:
        return "pv", sim_cols_no_s, pv_cols_no_s
    if has_sim:
        return "sim", sim_cols_no_s, pv_cols_no_s
    missing_sim = [c for c in sim_cols_no_s if c not in df.columns]
    missing_pv = [c for c in pv_cols_no_s if c not in df.columns]
    raise SystemExit(
        f"Input CSV does not match either schema. "
        f"Missing sim columns: {missing_sim}. Missing PV columns: {missing_pv}."
    )


# ── LUME-Torch model builders ────────────────────────────────────────────────

def create_lume_torch_sim(model, input_tr, output_tr, value_ranges=None, dump_dir="lumetorchyaml-sim"):
    """Create LUME-torch model that takes simulator-parameter + s inputs."""
    feature_cols = list(input_tr["feature_cols"])
    x_mean = input_tr["x_mean"].to(dtype=torch.float32)
    x_std = input_tr["x_std"].to(dtype=torch.float32)
    target_cols = list(output_tr["target_cols"])
    y_mean = output_tr["y_mean"].to(dtype=torch.float32)
    y_std = output_tr["y_std"].to(dtype=torch.float32)

    input_variables = []
    for idx, col in enumerate(feature_cols):
        kwargs = {"name": col, "default_value": float(x_mean[idx])}
        if value_ranges is not None and col in value_ranges:
            kwargs["value_range"] = value_ranges[col]
        input_variables.append(TorchScalarVariable(**kwargs))
    output_variables = [TorchScalarVariable(name=col) for col in target_cols]

    normalization_transform = AffineInputTransform(
        d=len(feature_cols), coefficient=x_std, offset=x_mean
    )
    denorm_transform = OutputDenormTransform(y_mean, y_std)

    torch_model = TorchModel(
        model=model,
        input_variables=input_variables,
        output_variables=output_variables,
        input_transformers=[normalization_transform],
        output_transformers=[denorm_transform],
        precision="single",
    )

    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    torch_model.dump(f"{dump_dir}/beam_evolution_simulator.yaml")
    return TorchModule(model=torch_model)


def create_lume_torch_machine(model, input_tr, output_tr, value_ranges=None, dump_dir="lumetorchyaml-machine"):
    """Create LUME-torch model that takes machine-PV + s inputs.

    The first 19 inputs are machine PVs; the 20th is `s`.
    A custom PVToSimWithS transform converts PV→sim on channels 0..18
    and passes `s` (channel 19) through unchanged.
    """
    feature_cols = list(input_tr["feature_cols"])
    x_mean = input_tr["x_mean"].to(dtype=torch.float32)
    x_std = input_tr["x_std"].to(dtype=torch.float32)
    target_cols = list(output_tr["target_cols"])
    y_mean = output_tr["y_mean"].to(dtype=torch.float32)
    y_std = output_tr["y_std"].to(dtype=torch.float32)

    sim_cols_no_s = [c for c in feature_cols if c != "s"]
    pv_cols_no_s = machine_input_names(sim_cols_no_s)
    pv_defaults = sim_to_machine_array(
        x_mean[:19].cpu().numpy()[None, :], sim_cols_no_s
    )[0]

    # Machine PV inputs + s
    input_variables = []
    for idx, col in enumerate(pv_cols_no_s):
        kwargs = {"name": col, "default_value": float(pv_defaults[idx])}
        if value_ranges is not None and col in value_ranges:
            kwargs["value_range"] = value_ranges[col]
        input_variables.append(TorchScalarVariable(**kwargs))
    s_kwargs = {"name": "s", "default_value": float(x_mean[19])}
    if value_ranges is not None and "s" in value_ranges:
        s_kwargs["value_range"] = value_ranges["s"]
    input_variables.append(TorchScalarVariable(**s_kwargs))

    output_variables = [TorchScalarVariable(name=col) for col in target_cols]

    pv_to_sim_affine = build_pv_to_sim_transform(sim_cols_no_s)
    pv_to_sim_with_s = PVToSimWithS(pv_to_sim_affine)
    normalization_transform = AffineInputTransform(
        d=len(feature_cols), coefficient=x_std, offset=x_mean
    )
    denorm_transform = OutputDenormTransform(y_mean, y_std)

    torch_model = TorchModel(
        model=model,
        input_variables=input_variables,
        output_variables=output_variables,
        input_transformers=[pv_to_sim_with_s, normalization_transform],
        output_transformers=[denorm_transform],
        precision="single",
    )

    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    torch_model.dump(f"{dump_dir}/beam_evolution_machine.yaml")
    return TorchModule(model=torch_model)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Infer beam-evolution targets from sim-parameter or machine-PV input CSV."
    )
    p.add_argument("--model-dir", default="model-output-100h",
                   help="Directory containing model.pt and transformer files.")
    p.add_argument("--input-csv", default="dataset-test.csv",
                   help="CSV with sim-parameter or machine-PV columns + s.")
    p.add_argument("--train-csv", default="dataset-train.csv",
                   help="Training CSV to compute value_range (min/max) for LUME YAML inputs.")
    p.add_argument("--output-dir", default="inference-output",
                   help="Directory for inference outputs.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--input-space", choices=["auto", "sim", "pv"], default="auto",
                   help="Interpret CSV columns as sim parameters, machine PVs, or auto-detect.")
    p.add_argument("--print-row", type=int, default=0,
                   help="Row index whose predictions to print.")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    model_dir = Path(args.model_dir)
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)
    print(f"[run] Loading model from {model_dir}", flush=True)
    model, input_tr, output_tr = load_model_and_transformers(model_dir, device)

    feature_cols = list(input_tr["feature_cols"])
    target_cols = list(output_tr["target_cols"])
    x_mean = input_tr["x_mean"].cpu().numpy().astype(np.float32)
    x_std = input_tr["x_std"].cpu().numpy().astype(np.float32)
    y_mean = output_tr["y_mean"].cpu().numpy().astype(np.float32)
    y_std = output_tr["y_std"].cpu().numpy().astype(np.float32)

    print(f"[run] Reading input CSV: {input_csv}", flush=True)
    df = pd.read_csv(input_csv, low_memory=False)
    sim_cols_no_s = [c for c in feature_cols if c != "s"]
    input_space, _, pv_cols_no_s = resolve_input_space(df, feature_cols, args.input_space)

    # Build the full 20-column sim-parameter array (19 sim params + s)
    s_values = df["s"].values.astype(np.float32)
    if input_space == "sim":
        X_sim_no_s = df[sim_cols_no_s].values.astype(np.float32)
        print("[run] Detected simulator-parameter input columns", flush=True)
    else:
        X_machine = df[pv_cols_no_s].values.astype(np.float32)
        X_sim_no_s = machine_to_sim_array(X_machine, sim_cols_no_s)
        print("[run] Detected machine-PV input columns; applying PV -> sim transform", flush=True)

    X_sim = np.column_stack([X_sim_no_s, s_values])  # (N, 20)
    X_norm = (X_sim - x_mean) / x_std

    # ── Direct inference ─────────────────────────────────────────────────────
    loader_norm = DataLoader(TensorDataset(torch.from_numpy(X_norm)), batch_size=args.batch_size)
    print(f"[run] Running inference for {len(df)} rows", flush=True)

    pred_norm_batches = []
    with torch.no_grad():
        for (X_batch,) in loader_norm:
            pred = model(X_batch.to(device))
            pred_norm_batches.append(pred.cpu().numpy())
    preds_norm = np.concatenate(pred_norm_batches, axis=0)
    preds_raw = preds_norm * y_std + y_mean

    # ── Compute value ranges from training data ──────────────────────────────
    train_csv = Path(args.train_csv)
    sim_value_ranges = None
    pv_value_ranges = None
    if train_csv.exists():
        print(f"[run] Computing value_range from {train_csv}", flush=True)
        train_df = pd.read_csv(train_csv, usecols=feature_cols)
        sim_value_ranges = {}
        for col in feature_cols:
            sim_value_ranges[col] = [float(train_df[col].min()), float(train_df[col].max())]
        # PV ranges
        train_sim_no_s = train_df[sim_cols_no_s].values.astype(np.float32)
        train_pv = sim_to_machine_array(train_sim_no_s, sim_cols_no_s)
        pv_value_ranges = {}
        for i, col in enumerate(pv_cols_no_s):
            pv_value_ranges[col] = [float(train_pv[:, i].min()), float(train_pv[:, i].max())]
        pv_value_ranges["s"] = sim_value_ranges["s"]
    else:
        print(f"[run] Training CSV not found ({train_csv}); skipping value_range", flush=True)

    # ── LUME-Torch sim-input model ───────────────────────────────────────────
    lume_sim = create_lume_torch_sim(model, input_tr, output_tr, value_ranges=sim_value_ranges)
    loader_sim = DataLoader(TensorDataset(torch.from_numpy(X_sim)), batch_size=args.batch_size)

    pred_lume_sim_batches = []
    with torch.no_grad():
        for (X_batch,) in loader_sim:
            out = lume_sim(X_batch.to(device))
            pred_lume_sim_batches.append(out.cpu().numpy())
    preds_lume_sim = np.concatenate(pred_lume_sim_batches, axis=0)

    if not np.allclose(preds_raw, preds_lume_sim, rtol=1e-5, atol=1e-5):
        max_diff = float(np.max(np.abs(preds_raw - preds_lume_sim)))
        raise AssertionError(
            f"Sim-input LUME model mismatch vs direct model; max abs diff={max_diff:.6e}"
        )
    print("[run] Sim-input LUME-torch model validated successfully", flush=True)

    # ── LUME-Torch machine-input model ───────────────────────────────────────
    X_machine_full = np.column_stack([
        sim_to_machine_array(X_sim_no_s, sim_cols_no_s), s_values
    ])  # (N, 20): 19 PVs + s

    # Reference via roundtrip to match float32 precision
    X_sim_roundtrip_no_s = machine_to_sim_array(X_machine_full[:, :19], sim_cols_no_s)
    X_sim_roundtrip = np.column_stack([X_sim_roundtrip_no_s, s_values])
    X_norm_roundtrip = (X_sim_roundtrip - x_mean) / x_std

    loader_roundtrip = DataLoader(
        TensorDataset(torch.from_numpy(X_norm_roundtrip)), batch_size=args.batch_size
    )
    pred_ref_batches = []
    with torch.no_grad():
        for (X_batch,) in loader_roundtrip:
            pred = model(X_batch.to(device))
            pred_ref_batches.append(pred.cpu().numpy())
    preds_ref_roundtrip = np.concatenate(pred_ref_batches, axis=0)
    preds_ref_roundtrip_raw = preds_ref_roundtrip * y_std + y_mean

    lume_machine = create_lume_torch_machine(model, input_tr, output_tr, value_ranges=pv_value_ranges)
    loader_machine = DataLoader(
        TensorDataset(torch.from_numpy(X_machine_full)), batch_size=args.batch_size
    )
    pred_lume_machine_batches = []
    with torch.no_grad():
        for (X_batch,) in loader_machine:
            out = lume_machine(X_batch.to(device))
            pred_lume_machine_batches.append(out.cpu().numpy())
    preds_lume_machine = np.concatenate(pred_lume_machine_batches, axis=0)

    if not np.allclose(preds_ref_roundtrip_raw, preds_lume_machine, rtol=1e-5, atol=1e-5):
        max_diff = float(np.max(np.abs(preds_ref_roundtrip_raw - preds_lume_machine)))
        raise AssertionError(
            f"Machine-input LUME model mismatch vs direct model; max abs diff={max_diff:.6e}"
        )
    print("[run] Machine-input LUME-torch model validated successfully", flush=True)

    # ── Save predictions ─────────────────────────────────────────────────────
    pred_df = pd.DataFrame(
        {f"pred_{col}": preds_raw[:, idx] for idx, col in enumerate(target_cols)}
    )
    # Include true targets if available
    true_cols_present = [c for c in target_cols if c in df.columns]
    if true_cols_present:
        true_df = df[true_cols_present].copy()
        true_df.columns = [f"true_{c}" for c in true_cols_present]
    else:
        true_df = pd.DataFrame()

    # Include s and sample identifiers if present
    id_cols = [c for c in ["sample_idx", "csv_row_idx", "s"] if c in df.columns]
    base_df = df[id_cols].copy()

    result_df = pd.concat([base_df, true_df, pred_df], axis=1)
    result_df.to_csv(output_dir / "predictions.csv", index=False)
    np.save(output_dir / "predictions.npy", preds_raw)

    # ── Print sample row ─────────────────────────────────────────────────────
    row_idx = args.print_row
    if row_idx < 0 or row_idx >= len(df):
        raise SystemExit(f"--print-row must be between 0 and {len(df) - 1}")

    print(f"\n[run] Saved predictions to {output_dir / 'predictions.csv'}", flush=True)
    print(f"[run] Saved predictions array to {output_dir / 'predictions.npy'}", flush=True)
    print(f"\n[run] Predictions for row {row_idx}:", flush=True)
    for i, col in enumerate(target_cols):
        true_val = f" (true: {df[col].iloc[row_idx]:.6g})" if col in df.columns else ""
        print(f"  {col:>25s} = {preds_raw[row_idx, i]:.6g}{true_val}", flush=True)

    print(
        "\n[run] Flow: inputs -> [PV-to-sim if machine] -> z-score normalization "
        "-> MLP surrogate -> z-score denormalization -> 12 beam-evolution targets.",
        flush=True,
    )
    print(f"[run] LUME YAML exported to lumetorchyaml-sim/ and lumetorchyaml-machine/", flush=True)


if __name__ == "__main__":
    main()
