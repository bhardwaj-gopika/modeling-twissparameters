"""Train the beam-evolution MLP on the prepared dataset splits.

Inputs (20):  19 control parameters + s
Outputs (12): sigma_x, sigma_y, sigma_z, norm_emit_x, norm_emit_y,
              emit_geom_x, emit_geom_y, beta_x, alpha_x, beta_y, alpha_y,
              mean_kinetic_energy

Architecture matches the modeling-571 covariance surrogate backbone:
    Linear(N -> 100), ELU
    Linear(100 -> 200), ELU, Dropout
    Linear(200 -> 200), ELU, Dropout
    Linear(200 -> 300), ELU, Dropout
    Linear(300 -> 300), ELU, Dropout
    Linear(300 -> 200), ELU, Dropout
    Linear(200 -> 100), ELU, Dropout
    Linear(100 -> 100), ELU
    Linear(100 -> 100), ELU
A single 12-dim linear head produces normalized targets.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Column definitions ────────────────────────────────────────────────────────
ID_COLUMNS = ["sample_idx", "csv_row_idx"]
TARGET_COLUMNS = [
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "norm_emit_x",
    "norm_emit_y",
    "emit_geom_x",
    "emit_geom_y",
    "beta_x",
    "alpha_x",
    "beta_y",
    "alpha_y",
    "mean_kinetic_energy",
]


def get_feature_target_columns(df: pd.DataFrame):
    target_cols = [c for c in TARGET_COLUMNS if c in df.columns]
    missing = set(TARGET_COLUMNS) - set(target_cols)
    if missing:
        raise SystemExit(f"Dataset is missing target columns: {sorted(missing)}")
    drop = set(ID_COLUMNS) | set(target_cols)
    feature_cols = [c for c in df.columns if c not in drop]
    if "s" not in feature_cols:
        raise SystemExit("Dataset is missing the 's' (position) feature column.")
    return feature_cols, target_cols


# ── Model ─────────────────────────────────────────────────────────────────────
class BeamEvolutionModel(nn.Module):
    """MLP: (19 controls + s) -> 12 beam-evolution targets."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        y_mean: torch.Tensor | None = None,
        y_std: torch.Tensor | None = None,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.backbone = nn.Sequential(
            nn.Linear(n_inputs, 100),
            nn.ELU(),
            nn.Linear(100, 200),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(200, 300),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(300, 200),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
        )
        self.head = nn.Linear(100, n_outputs)

        if y_mean is None:
            y_mean = torch.zeros(n_outputs, dtype=torch.float32)
        if y_std is None:
            y_std = torch.ones(n_outputs, dtype=torch.float32)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predictions in normalized (z-score) space."""
        return self.head(self.backbone(x))

    def predict_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predictions in original physical units."""
        return self.forward(x) * self.y_std + self.y_mean


def build_model(n_inputs, n_outputs, y_mean=None, y_std=None) -> BeamEvolutionModel:
    return BeamEvolutionModel(n_inputs, n_outputs, y_mean=y_mean, y_std=y_std)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_split(
    path: Path,
    feature_cols,
    target_cols,
    x_mean,
    x_std,
    y_mean,
    y_std,
) -> TensorDataset:
    df = pd.read_csv(path, low_memory=False)
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


# ── Training ──────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    n_samples = 0
    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(X_batch)
            n_samples += len(X_batch)
    return total_loss / n_samples


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the beam-evolution MLP.")
    p.add_argument("--train-csv", default="dataset-train.csv")
    p.add_argument("--val-csv", default="dataset-val.csv")
    p.add_argument("--test-csv", default="dataset-test.csv")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--loss",
        choices=["mse", "l1"],
        default="l1",
        help="Per-target normalized regression loss (default: l1).",
    )
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stopping patience in epochs (0 disables).")
    p.add_argument("--output-dir", default="model-output",
                   help="Where to save model.pt and transformer files.")
    # Optional staged fine-tuning (mirrors modeling-571-moredata convention).
    p.add_argument("--finetune-batch-sizes", type=int, nargs="+", default=None)
    p.add_argument("--finetune-epochs-per-stage", type=int, default=0)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--finetune-lr-decay", type=float, default=0.5)
    p.add_argument("--finetune-plateau-patience", type=int, default=5)
    p.add_argument("--finetune-min-lr", type=float, default=1e-6)
    return p


def main() -> None:
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute scalers from training set only ────────────────────────────────
    print(f"[run] Reading training CSV: {args.train_csv}", flush=True)
    train_df = pd.read_csv(args.train_csv, low_memory=False)
    feature_cols, target_cols = get_feature_target_columns(train_df)
    n_inputs = len(feature_cols)
    n_outputs = len(target_cols)
    print(
        f"[run] Features ({n_inputs}): {feature_cols}",
        flush=True,
    )
    print(f"[run] Targets ({n_outputs}): {target_cols}", flush=True)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_cols].values.astype(np.float32)

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0] = 1.0

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std == 0] = 1.0

    input_transformers = {
        "x_mean": torch.from_numpy(x_mean),
        "x_std": torch.from_numpy(x_std),
        "feature_cols": feature_cols,
    }
    output_transformers = {
        "y_mean": torch.from_numpy(y_mean),
        "y_std": torch.from_numpy(y_std),
        "target_cols": target_cols,
    }
    torch.save(input_transformers, output_dir / "input_transformers.pt")
    torch.save(output_transformers, output_dir / "output_transformers.pt")
    print(f"[run] Saved transformers to {output_dir}", flush=True)

    # ── Datasets / loaders ────────────────────────────────────────────────────
    train_ds = load_split(Path(args.train_csv), feature_cols, target_cols,
                          x_mean, x_std, y_mean, y_std)
    val_ds = load_split(Path(args.val_csv), feature_cols, target_cols,
                        x_mean, x_std, y_mean, y_std)
    test_ds = load_split(Path(args.test_csv), feature_cols, target_cols,
                         x_mean, x_std, y_mean, y_std)
    print(
        f"[run] Rows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
        flush=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    y_mean_t = torch.from_numpy(y_mean).to(device)
    y_std_t = torch.from_numpy(y_std).to(device)
    model = build_model(n_inputs, n_outputs, y_mean=y_mean_t, y_std=y_std_t).to(device)
    print(f"[run] Model:\n{model}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run] Parameter count: {n_params:,}", flush=True)

    criterion = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    print(f"[run] Loss: {args.loss} in z-score-normalized target space", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"phase": [], "epoch": [], "train_loss": [], "val_loss": [], "lr": []}

    print(f"\n[run] Training for up to {args.epochs} epochs ...", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["phase"].append("base")
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:04d}/{args.epochs}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "model.pt")
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            print(
                f"[run] Early stopping at epoch {epoch} "
                f"(no improvement for {args.patience} epochs)",
                flush=True,
            )
            break

    # ── Optional staged fine-tuning ───────────────────────────────────────────
    do_finetune = (
        args.finetune_batch_sizes is not None
        and args.finetune_epochs_per_stage > 0
        and len(args.finetune_batch_sizes) > 0
    )
    if do_finetune:
        print(
            f"\n[run] Fine-tuning stages: batch_sizes={args.finetune_batch_sizes} "
            f"epochs_per_stage={args.finetune_epochs_per_stage} "
            f"initial_lr={args.finetune_lr:.2e}",
            flush=True,
        )
        model.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))
        stage_lr = args.finetune_lr

        for stage_idx, stage_bs in enumerate(args.finetune_batch_sizes, start=1):
            stage_train_loader = DataLoader(train_ds, batch_size=stage_bs, shuffle=True)
            stage_val_loader = DataLoader(val_ds, batch_size=stage_bs)
            stage_optimizer = torch.optim.Adam(model.parameters(), lr=stage_lr)
            stage_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                stage_optimizer,
                mode="min",
                factor=0.5,
                patience=args.finetune_plateau_patience,
                min_lr=args.finetune_min_lr,
            )
            print(
                f"[finetune stage {stage_idx}] batch_size={stage_bs} "
                f"lr={stage_lr:.2e} epochs={args.finetune_epochs_per_stage}",
                flush=True,
            )
            for stage_epoch in range(1, args.finetune_epochs_per_stage + 1):
                t0 = time.time()
                train_loss = run_epoch(
                    model, stage_train_loader, criterion, stage_optimizer, device, train=True
                )
                val_loss = run_epoch(
                    model, stage_val_loader, criterion, stage_optimizer, device, train=False
                )
                stage_scheduler.step(val_loss)

                history["phase"].append(f"finetune_bs{stage_bs}")
                history["epoch"].append(stage_epoch)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["lr"].append(stage_optimizer.param_groups[0]["lr"])

                elapsed = time.time() - t0
                print(
                    f"[finetune {stage_idx}:{stage_epoch:03d}/"
                    f"{args.finetune_epochs_per_stage}] "
                    f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                    f"lr={stage_optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s",
                    flush=True,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), output_dir / "model.pt")

            stage_lr = max(stage_lr * args.finetune_lr_decay, args.finetune_min_lr)

    # ── Save history ──────────────────────────────────────────────────────────
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    print(f"[run] History saved to {output_dir}/training_history.csv", flush=True)

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\n[run] Loading best checkpoint for test evaluation ...", flush=True)
    model.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))
    test_loss = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"[run] Test loss ({args.loss}, normalized): {test_loss:.6f}", flush=True)

    # Per-target MAE in original (raw) units.
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred_norm = model(X_batch).cpu().numpy()
            preds_list.append(pred_norm)
            targets_list.append(y_batch.numpy())
    preds_norm = np.concatenate(preds_list)
    targets_norm = np.concatenate(targets_list)
    preds_raw = preds_norm * y_std + y_mean
    targets_raw = targets_norm * y_std + y_mean

    abs_err = np.abs(preds_raw - targets_raw)
    mae_per_target = abs_err.mean(axis=0)
    denom = np.where(np.abs(targets_raw) > 1e-30, np.abs(targets_raw), np.nan)
    mape_per_target = np.nanmean(abs_err / denom, axis=0) * 100.0

    print("\n[run] Test MAE / MAPE per target (raw units):", flush=True)
    print(f"  {'target':24s}  {'MAE':>14s}  {'MAPE (%)':>10s}", flush=True)
    for name, mae, mape in zip(target_cols, mae_per_target, mape_per_target):
        print(f"  {name:24s}  {mae:14.6e}  {mape:10.3f}", flush=True)

    metrics_df = pd.DataFrame(
        {"target": target_cols, "mae": mae_per_target, "mape_percent": mape_per_target}
    )
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    print(f"\n[run] Test metrics saved to {output_dir}/test_metrics.csv", flush=True)
    print(f"[run] Best val_loss: {best_val_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
