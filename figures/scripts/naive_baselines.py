"""Naive (non-learned) baselines for tackle-location prediction.

Mirrors the evaluation in src/generate_results_summary.py exactly:
  - test split, mirrored=False only
  - ADE = mean euclidean distance on absolute (tackle_x, tackle_y)
  - frame buckets = tackle_frameId - frameId, cut at range(0,31,5)
"""

import numpy as np
import polars as pl

DATA = "data/split_prepped_data"
FPS = 10.0  # BDB tracking is 10 Hz; s is yards/sec


def ball_carrier_frames(split: str) -> pl.DataFrame:
    """One row per (game, play, frame): ball carrier state + anchor + target."""
    feats = (
        pl.scan_parquet(f"{DATA}/{split}_features.parquet")
        .filter((~pl.col("mirrored")) & (pl.col("is_ball_carrier") == 1))
        .select(["gameId", "playId", "frameId", "x", "y", "vx", "vy", "ax", "ay", "anchor_x", "anchor_y"])
        .collect()
    )
    tgts = (
        pl.read_parquet(f"{DATA}/{split}_targets.parquet")
        .filter(~pl.col("mirrored"))
        .select(["gameId", "playId", "frameId", "tackle_frameId", "tackle_x", "tackle_y", "tackle_event"])
        .unique(subset=["gameId", "playId", "frameId"])
    )
    return feats.join(tgts, on=["gameId", "playId", "frameId"], how="inner")


def ade(df: pl.DataFrame, px: str, py: str) -> float:
    d = np.sqrt((df[px].to_numpy() - df["tackle_x"].to_numpy()) ** 2 + (df[py].to_numpy() - df["tackle_y"].to_numpy()) ** 2)
    return float(d.mean())


def fit_global_dt(train: pl.DataFrame) -> float:
    """Single global horizon dt (seconds) minimising ADE of constant-velocity extrapolation."""
    grid = np.arange(0.0, 4.01, 0.01)
    x, y = train["x"].to_numpy(), train["y"].to_numpy()
    vx, vy = train["vx"].to_numpy(), train["vy"].to_numpy()
    tx, ty = train["tackle_x"].to_numpy(), train["tackle_y"].to_numpy()
    best, best_dt = np.inf, 0.0
    for dt in grid:
        d = np.sqrt((x + vx * dt - tx) ** 2 + (y + vy * dt - ty) ** 2).mean()
        if d < best:
            best, best_dt = d, dt
    return float(best_dt)


def add_buckets(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        fd=(pl.col("tackle_frameId") - pl.col("frameId")),
    ).with_columns(
        bucket=pl.col("fd").cut(
            breaks=range(0, 31, 5),
            labels=["after tackle", "0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"],
            left_closed=True,
        )
    )


def main() -> None:
    train = ball_carrier_frames("train")
    test = ball_carrier_frames("test")
    print(f"train frames: {len(train):,}   test frames: {len(test):,}")
    print(f"test plays:   {test.select(['gameId','playId']).n_unique():,}")

    dt_hat = fit_global_dt(train)
    print(f"\nfitted global horizon dt = {dt_hat:.2f} s ({dt_hat*FPS:.1f} frames)")

    # ---- baselines -------------------------------------------------------
    test = test.with_columns(
        # B0: play-start ball carrier position (the anchor)
        b0_x=pl.col("anchor_x"), b0_y=pl.col("anchor_y"),
        # B1: current ball carrier position
        b1_x=pl.col("x"), b1_y=pl.col("y"),
        # B2: constant velocity, single fitted global horizon
        b2_x=pl.col("x") + pl.col("vx") * dt_hat,
        b2_y=pl.col("y") + pl.col("vy") * dt_hat,
        # B2-oracle: constant velocity with the TRUE remaining time (upper bound on physics)
        b2o_x=pl.col("x") + pl.col("vx") * ((pl.col("tackle_frameId") - pl.col("frameId")) / FPS),
        b2o_y=pl.col("y") + pl.col("vy") * ((pl.col("tackle_frameId") - pl.col("frameId")) / FPS),
        # B2-oracle + acceleration (constant-accel kinematics with true remaining time)
        b2a_x=pl.col("x") + pl.col("vx") * ((pl.col("tackle_frameId") - pl.col("frameId")) / FPS)
        + 0.5 * pl.col("ax") * ((pl.col("tackle_frameId") - pl.col("frameId")) / FPS) ** 2,
        b2a_y=pl.col("y") + pl.col("vy") * ((pl.col("tackle_frameId") - pl.col("frameId")) / FPS)
        + 0.5 * pl.col("ay") * ((pl.col("tackle_frameId") - pl.col("frameId")) / FPS) ** 2,
    )

    names = {
        "b0": "B0  anchor (play-start BC position)",
        "b1": "B1  current BC position",
        "b2": f"B2  const-velocity, global dt={dt_hat:.2f}s",
        "b2o": "B2* const-velocity, ORACLE dt",
        "b2a": "B3* const-accel, ORACLE dt",
    }

    print("\n" + "=" * 62)
    print("OVERALL TEST ADE (yards), mirrored=False, all frames")
    print("=" * 62)
    overall = {}
    for k, label in names.items():
        overall[k] = ade(test, f"{k}_x", f"{k}_y")
        print(f"  {label:42s} {overall[k]:6.3f}")
    print(f"  {'--- hybrid_ts (published this work)':42s} {4.09:6.3f}")
    print(f"  {'--- stgnn_ts_knn_hybrid_hub (best)':42s} {4.077:6.3f}")
    print(f"  {'--- SportsTransformer [3]':42s} {4.61:6.3f}")

    # ---- by horizon bucket ----------------------------------------------
    tb = add_buckets(test)
    print("\n" + "=" * 96)
    print("ADE BY FRAMES BEFORE TACKLE")
    print("=" * 96)
    hdr = f"{'bucket':>14s} {'n_frames':>9s} {'%mass_b1':>9s}  " + "".join(f"{k:>9s}" for k in names)
    print(hdr)
    print("-" * len(hdr))
    order = ["after tackle", "0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"]
    rows = []
    for b in order:
        sub = tb.filter(pl.col("bucket") == b)
        if len(sub) == 0:
            continue
        vals = {k: ade(sub, f"{k}_x", f"{k}_y") for k in names}
        mass = vals["b1"] * len(sub)
        rows.append((b, len(sub), mass, vals))
    total_mass = sum(r[2] for r in rows)
    for b, n, mass, vals in rows:
        print(f"{b:>14s} {n:>9,d} {100*mass/total_mass:>8.1f}%  " + "".join(f"{vals[k]:>9.3f}" for k in names))

    # error mass share for the model, for comparison
    print("\nNote: %mass_b1 is B1's share of total error mass, not the model's.")
    print(f"Total test frames evaluated: {len(test):,}")


if __name__ == "__main__":
    main()
