"""
Naive baseline: predict the tackler as whichever defender is currently closest
to the ball carrier.

No model involved -- pure geometry, for comparison against trained models.

The prediction uses the LAST frame of each window, matching the frame used
by the STGNN adjacency construction.
"""

import numpy as np
import polars as pl

from datasets import load_datasets, BDB2024_Dataset


# RAW_FEATURES:
#   0 = x_rel
#   1 = y_rel
#   2 = vx
#   3 = vy
#   4 = ox
#   5 = oy
#   6 = side
#   7 = is_ball_carrier
IDX_X = 0
IDX_Y = 1
IDX_SIDE = 6
IDX_BC = 7


def nearest_defender_baseline(
    model_type: str,
    window_length: int,
    split: str,
) -> pl.DataFrame:
    """
    For every frame in the split, predict the tackler as the defender
    closest to the ball carrier.

    Uses the LAST frame of the input window.

    Returns:
        gameId
        playId
        mirrored
        frameId
        tacklerNflId
        pred_tackler_nflId
        correct
    """

    print(f"Loading {split} dataset...")

    ds: BDB2024_Dataset = load_datasets(
        model_type,
        split=split,
        window_length=window_length,
        target_type="tackler",
    )

    print(f"Loaded {len(ds):,} frames.")

    # Build nflId lookup once
    index_df = ds.feature_df_partition.index.to_frame(index=False)

    nfl_ids_by_key = {}

    for row in index_df.itertuples(index=False):
        key = (
            row.gameId,
            row.playId,
            row.mirrored,
            row.frameId,
        )

        nfl_ids_by_key.setdefault(key, []).append(row.nflId)

    print("Built nflId lookup.")

    rows = []

    for i, key in enumerate(ds.keys):

        if i % 100_000 == 0:
            print(f"Processing {i:,} / {len(ds):,} frames...")

        # Get features for this frame.
        feat = ds.feature_arrays[key]

        # [22, 2]
        pos = feat[:, [IDX_X, IDX_Y]]

        # Player side
        side = feat[:, IDX_SIDE]

        # Ball-carrier indicator
        bc = feat[:, IDX_BC]

        # Find ball carrier.
        bc_mask = bc == 1

        if not np.any(bc_mask):
            continue

        bc_pos = pos[bc_mask][0]

        # Find defenders.
        defender_mask = side == -1

        def_pos = pos[defender_mask]

        if len(def_pos) == 0:
            continue

        # Use pre-built nflId lookup.
        nfl_ids = np.asarray(nfl_ids_by_key[key])

        def_ids = nfl_ids[defender_mask]

        # Euclidean distance from each defender to ball carrier.
        distances = np.linalg.norm(
            def_pos - bc_pos,
            axis=1,
        )

        nearest_idx = np.argmin(distances)

        nearest_id = def_ids[nearest_idx]

        # True tackler.
        target = ds.tgt_df_partition.loc[
            key,
            "tacklerNflId",
        ]

        rows.append(
            {
                "gameId": key[0],
                "playId": key[1],
                "mirrored": key[2],
                "frameId": key[3],
                "tacklerNflId": target,
                "pred_tackler_nflId": nearest_id,
                "correct": nearest_id == target,
            }
        )

    print(f"Finished {split}: {len(rows):,} predictions.")

    return pl.DataFrame(rows)


def run_baseline(
    model_type: str = "windowed_transformer",
    window_length: int = 1,
):
    """
    Run the nearest-defender baseline on train, validation, and test.
    """

    results = {}

    for split in ["train", "val", "test"]:

        print()
        print("=" * 60)
        print(f"Running nearest-defender baseline: {split}")
        print("=" * 60)

        df = nearest_defender_baseline(
            model_type=model_type,
            window_length=window_length,
            split=split,
        )

        accuracy = df["correct"].mean()

        print(
            f"{split}: nearest-defender baseline accuracy = "
            f"{accuracy * 100:.2f}% "
            f"(n={len(df):,})"
        )

        results[split] = df

    return results


if __name__ == "__main__":
    run_baseline(
        model_type="windowed_transformer",
        window_length=1,
    )

