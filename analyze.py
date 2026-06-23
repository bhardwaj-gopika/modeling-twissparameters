"""Analyze a trained beam-evolution surrogate (12 targets vs s).

Produces, in --output-dir:
  - test_metrics.csv          (MAE, RMSE, MAPE per target)
  - training_curve.png         (train/val loss vs epoch, colored by phase)
  - mae_per_target.png         (bar chart, raw units)
  - mape_per_target.png        (bar chart, %)
  - scatter_pred_vs_true.png   (12-panel scatter with R²)
  - per_sample_overlay.png     (true vs pred curves on first N test rows)
  - per_sample_zoomed.png      (5–95 percentile zoom dot version)
  - sorted_by_magnitude.png    (sorted by |true| value)
  - evolution_curves.png       (random samples: true vs predicted target vs s)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from train import build_model, TARGET_COLUMNS


# ──────────────────────────────────────────────────────────────────────────────
def load_model_and_transformers(model_dir: Path):
    input_tr = torch.load(model_dir / "input_transformers.pt", map_location="cpu")
    output_tr = torch.load(model_dir / "output_transformers.pt", map_location="cpu")

    feature_cols = list(input_tr["feature_cols"])
    target_cols = list(output_tr["target_cols"])
    y_mean_t = output_tr["y_mean"].to(torch.float32)
    y_std_t = output_tr["y_std"].to(torch.float32)

    model = build_model(len(feature_cols), len(target_cols),
                        y_mean=y_mean_t, y_std=y_std_t)
    model.load_state_dict(
        torch.load(model_dir / "model.pt", weights_only=True, map_location="cpu")
    )
    model.eval()
    return model, input_tr, output_tr


def evaluate(model, loader, output_tr, device):
    y_mean = output_tr["y_mean"].cpu().numpy()
    y_std = output_tr["y_std"].cpu().numpy()

    preds_list, tgts_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            pred_norm = model(X_batch.to(device)).cpu().numpy()
            preds_list.append(pred_norm)
            tgts_list.append(y_batch.numpy())
    preds_norm = np.concatenate(preds_list)
    tgts_norm = np.concatenate(tgts_list)
    preds_raw = preds_norm * y_std + y_mean
    tgts_raw = tgts_norm * y_std + y_mean

    abs_err = np.abs(preds_raw - tgts_raw)
    mae = abs_err.mean(axis=0)
    rmse = np.sqrt((abs_err ** 2).mean(axis=0))

    # Robust MAPE: only count rows where |true| > 1% of column std
    target_scale = tgts_raw.std(axis=0)
    mask = np.abs(tgts_raw) > 0.01 * target_scale[None, :]
    pct = np.where(mask, abs_err / np.where(np.abs(tgts_raw) > 0, np.abs(tgts_raw), np.nan) * 100, np.nan)
    mape = np.nanmean(pct, axis=0)

    return preds_raw, tgts_raw, mae, rmse, mape


# ──────────────────────────────────────────────────────────────────────────────
def _grid_axes(n_targets, ncols=4):
    nrows = int(np.ceil(n_targets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)
    return fig, axes, nrows, ncols


def plot_training_curve(history_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    epochs_axis = np.arange(1, len(history_df) + 1)
    if "phase" in history_df.columns:
        for phase, sub in history_df.groupby("phase", sort=False):
            x = epochs_axis[sub.index]
            ax.plot(x, sub["train_loss"], label=f"{phase} train", linewidth=1.2)
            ax.plot(x, sub["val_loss"], label=f"{phase} val", linewidth=1.2, linestyle="--")
    else:
        ax.plot(epochs_axis, history_df["train_loss"], label="train", linewidth=1.2)
        ax.plot(epochs_axis, history_df["val_loss"], label="val", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Cumulative epoch")
    ax.set_ylabel("L1 loss (normalized targets)")
    ax.set_yscale("log")
    ax.set_title("Training curve")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_bar(values, labels, ylabel, title, out_path: Path, overall=None, log=False):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    colors = ["tomato" if np.isfinite(v) and v > 25 and ylabel.endswith("%)") else "steelblue"
              for v in values]
    ax.bar(x, values, color=colors, alpha=0.85)
    if overall is not None:
        ax.axhline(overall, color="black", linestyle="--", linewidth=1.2,
                   label=f"Overall = {overall:.2f}")
        ax.legend()
    if log:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_scatter(preds, tgts, labels, out_path: Path):
    fig, axes, _, _ = _grid_axes(len(labels), ncols=4)
    fig.suptitle("Predicted vs True (test set)", y=1.0)
    for k, name in enumerate(labels):
        ax = axes[k]
        t = tgts[:, k]
        p = preds[:, k]
        all_vals = np.concatenate([t, p])
        lo, hi = np.nanpercentile(all_vals, [1, 99])
        pad = 0.05 * max(hi - lo, 1e-30)
        ax_lo, ax_hi = lo - pad, hi + pad
        inlier = (t >= lo) & (t <= hi) & (p >= lo) & (p <= hi)
        ss_res = float(np.sum((p[inlier] - t[inlier]) ** 2))
        ss_tot = float(np.sum((t[inlier] - t[inlier].mean()) ** 2))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ax.scatter(t, p, s=4, color="steelblue", alpha=0.35)
        ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], "k--", linewidth=1, alpha=0.4)
        ax.set_xlim(ax_lo, ax_hi)
        ax.set_ylim(ax_lo, ax_hi)
        ax.set_title(f"{name}  (R²={r_sq:.3f})", fontsize=9)
        ax.set_xlabel("True", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(len(labels), len(axes)):
        axes[k].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_sample_overlay(preds, tgts, labels, out_path: Path,
                            max_samples: int = 1000, zoomed: bool = False,
                            low_q: float = 5, high_q: float = 95):
    n = preds.shape[0]
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples, dtype=int)
        preds = preds[idx]
        tgts = tgts[idx]
    x = np.arange(preds.shape[0])
    fig, axes, _, _ = _grid_axes(len(labels), ncols=4)
    title = "Per-sample overlay" + (
        f" (zoomed {low_q:.0f}-{high_q:.0f}%)" if zoomed else ""
    )
    fig.suptitle(title, y=1.0)
    for k, name in enumerate(labels):
        ax = axes[k]
        t = tgts[:, k]
        p = preds[:, k]
        if zoomed:
            ax.scatter(x, t, s=4, color="steelblue", alpha=0.55, label="Target")
            ax.scatter(x, p, s=4, color="tomato", alpha=0.55, label="Predicted")
            y_all = np.concatenate([t, p])
            y_lo, y_hi = np.nanpercentile(y_all, [low_q, high_q])
            if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
                pad = 0.08 * (y_hi - y_lo)
                ax.set_ylim(y_lo - pad, y_hi + pad)
        else:
            ax.plot(x, t, color="steelblue", linewidth=0.7, alpha=0.7, label="Target")
            ax.plot(x, p, color="tomato", linewidth=0.7, alpha=0.7, label="Predicted")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc="best")
    for k in range(len(labels), len(axes)):
        axes[k].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_sorted_by_magnitude(preds, tgts, labels, out_path: Path, max_samples=1000):
    n = preds.shape[0]
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples, dtype=int)
        preds = preds[idx]
        tgts = tgts[idx]
    fig, axes, _, _ = _grid_axes(len(labels), ncols=4)
    fig.suptitle("Sorted by |true| magnitude: Target vs Predicted", y=1.0)
    for k, name in enumerate(labels):
        ax = axes[k]
        t = tgts[:, k]
        p = preds[:, k]
        order = np.argsort(np.abs(t))
        x = np.arange(len(t))
        ax.plot(x, t[order], color="steelblue", linewidth=0.7, alpha=0.7, label="Target")
        ax.plot(x, p[order], color="tomato", linewidth=0.7, alpha=0.7, label="Predicted")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc="best")
    for k in range(len(labels), len(axes)):
        axes[k].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_evolution_curves(test_df, feature_cols, target_cols, model, input_tr, output_tr,
                          device, out_path: Path, n_samples: int = 5, seed: int = 0):
    """Pick a few simulations from the test set and plot true vs predicted vs s for each target."""
    if "sample_idx" not in test_df.columns:
        print("[warn] sample_idx not in test CSV; skipping evolution_curves plot.", flush=True)
        return
    x_mean = input_tr["x_mean"].numpy()
    x_std = input_tr["x_std"].numpy()
    y_mean = output_tr["y_mean"].numpy()
    y_std = output_tr["y_std"].numpy()

    rng = np.random.default_rng(seed)
    unique_ids = test_df["sample_idx"].unique()
    pick = rng.choice(unique_ids, size=min(n_samples, len(unique_ids)), replace=False)

    fig, axes, _, _ = _grid_axes(len(target_cols), ncols=4)
    fig.suptitle(
        f"Beam evolution vs s — {len(pick)} random test samples (true solid, predicted dashed)",
        y=1.0,
    )
    cmap = plt.get_cmap("tab10")
    for j, sid in enumerate(pick):
        sub = test_df[test_df["sample_idx"] == sid].sort_values("s")
        if len(sub) == 0:
            continue
        s = sub["s"].to_numpy()
        X = sub[feature_cols].to_numpy(dtype=np.float32)
        X_norm = (X - x_mean) / x_std
        with torch.no_grad():
            preds = model(torch.from_numpy(X_norm).to(device)).cpu().numpy()
        preds_raw = preds * y_std + y_mean
        tgts = sub[target_cols].to_numpy(dtype=np.float32)
        color = cmap(j % 10)
        for k, name in enumerate(target_cols):
            ax = axes[k]
            ax.plot(s, tgts[:, k], color=color, linewidth=1.4, alpha=0.85)
            ax.plot(s, preds_raw[:, k], color=color, linewidth=1.0, linestyle="--", alpha=0.9)
    for k, name in enumerate(target_cols):
        axes[k].set_title(name, fontsize=9)
        axes[k].set_xlabel("s [m]", fontsize=8)
        axes[k].tick_params(labelsize=7)
        axes[k].grid(True, alpha=0.3)
    for k in range(len(target_cols), len(axes)):
        axes[k].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description="Analyze a trained beam-evolution surrogate.")
    p.add_argument("--model-dir", default="model-output-l1")
    p.add_argument("--test-csv", default="dataset-test.csv")
    p.add_argument("--output-dir", default="analysis-l1")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--overlay-max-samples", type=int, default=1500)
    p.add_argument("--evolution-num-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-scatter", action="store_true")
    p.add_argument("--skip-overlay", action="store_true")
    p.add_argument("--skip-sorted", action="store_true")
    p.add_argument("--skip-evolution", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)

    # Training curve
    history_path = model_dir / "training_history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        print(f"[run] Loaded {len(history_df)} epochs of history", flush=True)
        plot_training_curve(history_df, out_dir / "training_curve.png")
        print(f"[run] Saved training_curve.png", flush=True)

    print(f"[run] Loading model from {model_dir} ...", flush=True)
    model, input_tr, output_tr = load_model_and_transformers(model_dir)
    model.to(device)
    feature_cols = input_tr["feature_cols"]
    target_cols = output_tr["target_cols"]
    print(f"[run] Features ({len(feature_cols)}), Targets ({len(target_cols)}): {target_cols}",
          flush=True)

    print(f"[run] Loading test data from {args.test_csv} ...", flush=True)
    test_df = pd.read_csv(args.test_csv, low_memory=False)
    print(f"[run] Test rows: {len(test_df):,}", flush=True)

    x_mean = input_tr["x_mean"].numpy()
    x_std = input_tr["x_std"].numpy()
    y_mean = output_tr["y_mean"].numpy()
    y_std = output_tr["y_std"].numpy()
    X = test_df[feature_cols].values.astype(np.float32)
    y = test_df[target_cols].values.astype(np.float32)
    X_norm = (X - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    test_ds = TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(y_norm))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    print("[run] Evaluating on test set ...", flush=True)
    preds_raw, tgts_raw, mae, rmse, mape = evaluate(model, test_loader, output_tr, device)

    print("\n[results] Per-target metrics:", flush=True)
    print(f"  {'target':24s}  {'MAE':>14s}  {'RMSE':>14s}  {'MAPE (%)':>10s}", flush=True)
    for name, a, r, m in zip(target_cols, mae, rmse, mape):
        print(f"  {name:24s}  {a:14.6e}  {r:14.6e}  {m:10.3f}", flush=True)

    metrics_df = pd.DataFrame({
        "target": target_cols,
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
    })
    metrics_df.to_csv(out_dir / "test_metrics.csv", index=False)
    print(f"\n[run] test_metrics.csv saved", flush=True)

    plot_bar(mae, target_cols, "MAE (raw units)",
             "Test MAE per target", out_dir / "mae_per_target.png", log=True)
    plot_bar(mape, target_cols, "MAPE (%)",
             "Test MAPE per target", out_dir / "mape_per_target.png",
             overall=float(np.nanmean(mape)))
    print("[run] Bar plots saved", flush=True)

    if not args.skip_scatter:
        plot_scatter(preds_raw, tgts_raw, target_cols, out_dir / "scatter_pred_vs_true.png")
        print("[run] scatter_pred_vs_true.png saved", flush=True)

    if not args.skip_overlay:
        plot_per_sample_overlay(preds_raw, tgts_raw, target_cols,
                                out_dir / "per_sample_overlay.png",
                                max_samples=args.overlay_max_samples, zoomed=False)
        plot_per_sample_overlay(preds_raw, tgts_raw, target_cols,
                                out_dir / "per_sample_zoomed.png",
                                max_samples=args.overlay_max_samples, zoomed=True)
        print("[run] per-sample overlays saved", flush=True)

    if not args.skip_sorted:
        plot_sorted_by_magnitude(preds_raw, tgts_raw, target_cols,
                                 out_dir / "sorted_by_magnitude.png",
                                 max_samples=args.overlay_max_samples)
        print("[run] sorted_by_magnitude.png saved", flush=True)

    if not args.skip_evolution:
        plot_evolution_curves(test_df, feature_cols, target_cols, model,
                              input_tr, output_tr, device,
                              out_dir / "evolution_curves.png",
                              n_samples=args.evolution_num_samples,
                              seed=args.seed)
        print("[run] evolution_curves.png saved", flush=True)

    print(f"\n[run] Done. Outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
