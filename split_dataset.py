"""Split the long-format training dataset into train / val / test by sample_idx.

CRITICAL: We split on `sample_idx`, not on row, so that all s-grid rows belonging
to one simulation stay in the same split. Splitting by row would leak the same
sample's neighboring s-points between train and val, vastly overstating
performance.

Streams the input CSV in chunks so it works on the full ~2.5 M row dataset
without loading everything into memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split long-format dataset into train/val/test by sample_idx."
    )
    p.add_argument("input_csv", help="Path to dataset.csv (output of build_training_dataset.py)")
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--prefix", default=None, help="Output filename prefix (default: input stem)")
    p.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Rows per streaming read chunk (default: 500k)",
    )
    p.add_argument(
        "--id-column",
        default="sample_idx",
        help="Column used to define a 'sample' (default: sample_idx)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    fracs = [args.train_fraction, args.val_fraction, args.test_fraction]
    if any(f <= 0 for f in fracs):
        raise ValueError("All split fractions must be positive.")
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {sum(fracs):.6f}")

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_csv.stem
    id_col = args.id_column

    # Pass 1: collect unique sample IDs (cheap: ~12k uint values).
    print(f"[run] Scanning {input_csv} for unique {id_col} values...", flush=True)
    unique_ids: set[int] = set()
    total_rows = 0
    for chunk in pd.read_csv(input_csv, usecols=[id_col], chunksize=args.chunksize):
        unique_ids.update(chunk[id_col].unique().tolist())
        total_rows += len(chunk)
    ids = np.array(sorted(unique_ids))
    n_samples = len(ids)
    print(f"[run] {total_rows:,} rows across {n_samples:,} unique samples", flush=True)

    # Shuffle sample IDs and split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_samples)
    ids_shuffled = ids[perm]

    n_train = int(n_samples * args.train_fraction)
    n_val = int(n_samples * args.val_fraction)
    train_ids = set(ids_shuffled[:n_train].tolist())
    val_ids = set(ids_shuffled[n_train : n_train + n_val].tolist())
    test_ids = set(ids_shuffled[n_train + n_val :].tolist())
    print(
        f"[run] Samples split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}",
        flush=True,
    )

    out_paths = {
        "train": output_dir / f"{prefix}-train.csv",
        "val": output_dir / f"{prefix}-val.csv",
        "test": output_dir / f"{prefix}-test.csv",
    }
    id_sets = {"train": train_ids, "val": val_ids, "test": test_ids}

    # Wipe any pre-existing output files so append mode starts clean.
    for path in out_paths.values():
        if path.exists():
            path.unlink()

    written = {"train": 0, "val": 0, "test": 0}
    header_written = {"train": False, "val": False, "test": False}

    # Pass 2: stream-write rows to the right split.
    print(f"[run] Streaming rows in chunks of {args.chunksize:,}...", flush=True)
    for chunk_idx, chunk in enumerate(
        pd.read_csv(input_csv, chunksize=args.chunksize, low_memory=False)
    ):
        for split, id_set in id_sets.items():
            mask = chunk[id_col].isin(id_set)
            if not mask.any():
                continue
            sub = chunk.loc[mask]
            sub.to_csv(
                out_paths[split],
                index=False,
                mode="a",
                header=not header_written[split],
            )
            header_written[split] = True
            written[split] += len(sub)
        if (chunk_idx + 1) % 10 == 0 or chunk_idx == 0:
            done = sum(written.values())
            print(
                f"[run]   chunk {chunk_idx + 1}: cumulative train={written['train']:,} "
                f"val={written['val']:,} test={written['test']:,} ({done:,} / {total_rows:,})",
                flush=True,
            )

    print("[run] Done.", flush=True)
    for split, path in out_paths.items():
        print(f"[run]   {split}: {written[split]:,} rows -> {path}", flush=True)
    total_out = sum(written.values())
    if total_out != total_rows:
        print(
            f"[warn] Row total mismatch: wrote {total_out:,} vs scanned {total_rows:,}",
            flush=True,
        )


if __name__ == "__main__":
    main()
