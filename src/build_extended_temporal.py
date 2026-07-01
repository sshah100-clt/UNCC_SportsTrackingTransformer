"""
Build the extended temporal dataset for the additional-feature experiments.

Produces data/datasets/temporal_ext/{train,val,test}_dataset.pkl, identical to the
standard temporal dataset (data/datasets/temporal/) except that each per-player vector
carries EXTENDED_RAW_FEATURES (15 channels) instead of RAW_FEATURES (8):

    [x_rel, y_rel, vx, vy, ox, oy, side, is_ball_carrier,   # channels 0-7  (== RAW_FEATURES)
     ax, ay, delta_ox, delta_oy,                            # channels 8-11 (engineered dynamics)
     s, a, dis]                                             # channels 12-14 (raw kinematics)

Because the first 8 channels are RAW_FEATURES in the same order, slicing a temporal_ext
window [..., :8] reproduces the standard temporal window exactly. This reuses the project's
real precompute + on-the-fly windowing path (no hand-rolled windowing), so the dynamics and
kinematics experiment arms stay perfectly aligned with the 4.09 hybrid_ts baseline.

Usage:
    uv run python src/build_extended_temporal.py            # build missing splits
    uv run python src/build_extended_temporal.py --check    # build, then verify [:8] == temporal
"""

from __future__ import annotations

import pickle
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import polars as pl

from datasets import (
    DATASET_DIR,
    EXTENDED_RAW_FEATURES,
    PREPPED_DATA_DIR,
    RAW_FEATURE_COUNT,
    BDB2024_Dataset,
    load_datasets,
)

EXT_NAME = "temporal_ext"
SPLITS = ["test", "val", "train"]


def build():
    """Build each split's extended temporal pickle, skipping any that already exist."""
    out_dir = DATASET_DIR / EXT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        out_path = out_dir / f"{split}_dataset.pkl"
        if out_path.exists():
            print(f"Skipping existing dataset: {out_path}")
            continue
        print(f"Building {EXT_NAME} ({split}) with {len(EXTENDED_RAW_FEATURES)} features...")
        tic = time.time()
        feature_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_features.parquet")
        tgt_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_targets.parquet")
        # model_type is any temporal type (selects the temporal/windowing code path);
        # feature_list overrides which per-player columns are stored.
        dataset = BDB2024_Dataset("hybrid_ts", feature_df, tgt_df, feature_list=EXTENDED_RAW_FEATURES)
        with open(out_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"  -> {out_path}  ({(time.time() - tic) / 60:.1f} min)")


def check():
    """Verify temporal_ext[..., :8] reproduces the standard temporal dataset exactly."""
    print("\nSanity check: temporal_ext[:, :8] == temporal, on the test split...")
    base = load_datasets("hybrid_ts", split="test", window_length=15)  # standard temporal (8 feat)
    ext_path = DATASET_DIR / EXT_NAME / "test_dataset.pkl"
    with open(ext_path, "rb") as f:
        ext = pickle.load(f)
    ext.window_length = 15

    assert len(base) == len(ext), f"length mismatch: {len(base)} vs {len(ext)}"
    assert ext.feature_len == len(EXTENDED_RAW_FEATURES), f"ext feature_len={ext.feature_len}"
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(base), size=200, replace=False)
    max_abs = 0.0
    for i in idxs:
        fb, tb = base[int(i)]
        fe, te = ext[int(i)]
        assert fb.shape[:-1] == fe.shape[:-1], f"window shape mismatch at {i}: {fb.shape} vs {fe.shape}"
        assert fe.shape[-1] == len(EXTENDED_RAW_FEATURES)
        max_abs = max(max_abs, float(np.abs(fb - fe[..., :RAW_FEATURE_COUNT]).max()))
        assert np.allclose(tb, te), f"target mismatch at {i}"
    print(f"  checked {len(idxs)} windows | max |temporal - temporal_ext[:, :8]| = {max_abs:.2e}")
    assert max_abs == 0.0, "temporal_ext first 8 channels do not match temporal exactly"
    print("  PASS: extended dataset is a strict superset of the baseline windows.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Build the extended temporal dataset (15 per-player features)")
    parser.add_argument("--check", action="store_true", help="after building, verify [:8] == temporal")
    args = parser.parse_args()
    build()
    if args.check:
        check()
