"""
Tackler Identity Data Prep — NFL Big Data Bowl 2024

Standalone script, run AFTER prep_data.py's main(). Augments the already-prepped
location-target parquet files with tackler identity labels, without touching or
re-running any of prep_data.py's existing pipeline.

Reads:
    data/split_prepped_data/{split}_features.parquet
    data/split_prepped_data/{split}_targets.parquet
    tackles.csv (via prep_data.INPUT_DATA_DIR)

Writes:
    data/split_prepped_data/{split}_features_tackler.parquet
    data/split_prepped_data/{split}_targets_tackler.parquet

These are a strict subset of rows from the originals: plays with no recorded
primary tackler (out of bounds, touchdown, etc.) are dropped.
"""

from pathlib import Path

import polars as pl

from prep_data import get_tackler_target_df  # reuses the function already added there

PREPPED_DATA_DIR = Path("data/split_prepped_data/")


def build_tackler_split(split: str) -> None:
    features_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_features.parquet")
    targets_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_targets.parquet")

    tackler_df = get_tackler_target_df(features_df)  # needs "mirrored" col -- features_df has it

    tackler_targets_df = targets_df.join(
        tackler_df, on=["gameId", "playId", "mirrored"], how="inner"  # drops plays w/o a primary tackler
    )
    valid_plays = tackler_targets_df.select(["gameId", "playId", "mirrored"]).unique()
    tackler_features_df = features_df.join(valid_plays, on=["gameId", "playId", "mirrored"], how="inner")

    print(
        f"[{split}] {features_df.n_unique(['gameId', 'playId', 'mirrored'])} plays -> "
        f"{tackler_features_df.n_unique(['gameId', 'playId', 'mirrored'])} plays with a primary tackler "
        f"({tackler_targets_df.height} target rows)"
    )

    sort_keys_f = ["gameId", "playId", "mirrored", "frameId"]
    tackler_features_df.sort(sort_keys_f).write_parquet(PREPPED_DATA_DIR / f"{split}_features_tackler.parquet")
    tackler_targets_df.sort(["gameId", "playId", "mirrored"]).write_parquet(
        PREPPED_DATA_DIR / f"{split}_targets_tackler.parquet"
    )


def main():
    for split in ["train", "val", "test"]:
        build_tackler_split(split)


if __name__ == "__main__":
    main()