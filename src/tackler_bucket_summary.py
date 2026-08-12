"""
Time-bucketed evaluation for tackler-ID predictions

Pools predictions into fixed time-before-tackle bins and reports, per bin:
    - accuracy: percentage of how often the model's top pick was the true tackler
    - credited_accuracy: percentage of how often the top pick was EITHER the true
      tackler OR a defender credited with an assist on that play
    - avg_true_tackler_prob: how much probability, on average, the model assigned
      to the actual tackler, regardless of whether that was the model's top pick

Usage:
    python tackler_bucket_summary.py path/to/results.parquet
    python tackler_bucket_summary.py     # defaults to most recent file in tackler_results/
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from prep_data import INPUT_DATA_DIR

TACKLER_RESULTS_DIR = Path("tackler_results")

BUCKET_EDGES = [5, 10, 15, 20, 25, 30]  # frames from tackle (10 frames per second)
BUCKET_LABELS = ["0-0.5s", "0.5-1s", "1-1.5s", "1.5-2s", "2-2.5s", "2.5-3s", "3s+"]


def get_assist_nfl_ids_df() -> pl.DataFrame:
    """Per-play list of nflIds credited with an assist (NOT the primary tackler)."""
    tackles_df = pl.read_csv(INPUT_DATA_DIR / "tackles.csv")
    return (
        tackles_df.filter(pl.col("assist") == 1)
        .group_by(["gameId", "playId"])
        .agg(assist_nfl_ids=pl.col("nflId"))
    )


def add_assist_involvement(preds_df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds:
        was_assist_pick: model's top-1 pick was an assist-credited defender
        credited_pick: top pick was EITHER the primary tackler OR an assist-credited defender
    """
    assist_df = get_assist_nfl_ids_df()
    df = preds_df.join(assist_df, on=["gameId", "playId"], how="left")
    df = df.with_columns(pl.col("assist_nfl_ids").fill_null([]))
    return df.with_columns(
        was_assist_pick=(
            pl.col("pred_tackler_nflId").is_in(pl.col("assist_nfl_ids")) & ~pl.col("correct")
        ),
        credited_pick=(pl.col("correct") | pl.col("pred_tackler_nflId").is_in(pl.col("assist_nfl_ids"))),
    )


def add_true_tackler_prob(df: pl.DataFrame) -> pl.DataFrame:
    """Probability the model assigned to the actual tackler."""

    def _lookup(row: dict) -> float:
        ids = row["nfl_ids"]
        idx = ids.index(row["tacklerNflId"])
        return float(row["probs"][idx])

    return df.with_columns(
        pl.struct(["nfl_ids", "probs", "tacklerNflId"])
        .map_elements(_lookup, return_dtype=pl.Float32)
        .alias("true_tackler_prob")
    )


def add_time_bucket(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(frames_to_tackle=(pl.col("tackle_frameId") - pl.col("frameId")))
    bucket_expr = pl.lit(BUCKET_LABELS[-1])  # default: "3s+"
    bucket_order_expr = pl.lit(len(BUCKET_LABELS) - 1)
    for i in range(len(BUCKET_EDGES) - 1, -1, -1):  # step backwards to lowest bucket edge
        bucket_expr = (
            pl.when(pl.col("frames_to_tackle") < BUCKET_EDGES[i])
            .then(pl.lit(BUCKET_LABELS[i]))
            .otherwise(bucket_expr)
        )
        bucket_order_expr = (
            pl.when(pl.col("frames_to_tackle") < BUCKET_EDGES[i])
            .then(pl.lit(i))
            .otherwise(bucket_order_expr)
        )
    return df.with_columns(time_bucket=bucket_expr, bucket_order=bucket_order_expr)


def summarize_by_time_bucket(preds_df: pl.DataFrame, split: str = "test") -> pl.DataFrame:
    df = preds_df.filter(pl.col("dataset_split") == split)
    df = add_true_tackler_prob(df)
    df = add_time_bucket(df)
    df = add_assist_involvement(df)

    summary = (
        df.group_by(["time_bucket", "bucket_order"])
        .agg(
            accuracy=pl.col("correct").mean(),
            credited_accuracy=pl.col("credited_pick").mean(),
            avg_true_tackler_prob=pl.col("true_tackler_prob").mean(),
            n=pl.len(),
        )
        .sort("bucket_order")
        .drop("bucket_order")
        .with_columns(
            accuracy=(pl.col("accuracy") * 100).round(2),
            credited_accuracy=(pl.col("credited_accuracy") * 100).round(2),
            avg_true_tackler_prob=(pl.col("avg_true_tackler_prob") * 100).round(2),
        )
        .rename(
            {
                "accuracy": "accuracy_%",
                "credited_accuracy": "credited_accuracy_%",
                "avg_true_tackler_prob": "confidence_in_correct_player_%",
            }
        )
    )
    return summary


def plot_summary(summary: pl.DataFrame, out_path: str = "tackler_accuracy_by_time.png") -> None:
    labels = summary["time_bucket"].to_list()
    acc = summary["accuracy_%"].to_list()
    conf = summary["confidence_in_correct_player_%"].to_list()
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, acc, width=0.6, alpha=0.7, label="True Tackler Prediction Accuracy (%)")
    ax.plot(x, conf, marker="o", color="darkorange", label="Confidence in True Tackler (%)")
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("Time before tackle")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.set_title("Tackler Prediction Performance vs Time Before Contact")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")

def add_topk_hits(df: pl.DataFrame, ks: list[int] = [1, 3, 5]) -> pl.DataFrame:
    """
    Adds top_{k}_hit boolean columns: was the true tackler among the model's
    top-k most-probable defenders (by predicted probability)?
    """
    def _rank_of_true(row: dict) -> int:
        ids = row["nfl_ids"]
        probs = row["probs"]
        true_idx = ids.index(row["tacklerNflId"])
        true_prob = probs[true_idx]
        # rank = how many defenders have strictly higher prob than the true tackler
        # (ties broken conservatively -- true tackler counts as "found" only once
        # strictly-more-confident picks are exhausted)
        return sum(1 for p in probs if p > true_prob)

    df = df.with_columns(
        pl.struct(["nfl_ids", "probs", "tacklerNflId"])
        .map_elements(_rank_of_true, return_dtype=pl.Int64)
        .alias("true_tackler_rank")  # 0 = model's #1 pick was correct
    )
    for k in ks:
        df = df.with_columns((pl.col("true_tackler_rank") < k).alias(f"top_{k}_hit"))
    return df


def summarize_overall(preds_df: pl.DataFrame, split: str = "test", ks: list[int] = [1, 3, 5]) -> pl.DataFrame:
    """Single-row summary: overall top-1/top-k accuracy and confidence, no time bucketing."""
    df = preds_df.filter(pl.col("dataset_split") == split)
    df = add_true_tackler_prob(df)
    df = add_topk_hits(df, ks=ks)

    agg_exprs = [pl.col("correct").mean().alias("accuracy_%") * 100]
    agg_exprs += [pl.col(f"top_{k}_hit").mean().alias(f"top_{k}_accuracy_%") * 100 for k in ks]
    agg_exprs += [pl.col("true_tackler_prob").mean().alias("avg_confidence_in_correct_%") * 100]
    agg_exprs += [pl.len().alias("n")]

    return df.select(agg_exprs).with_columns(pl.exclude("n").round(2))

def main():
    results_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else max(TACKLER_RESULTS_DIR.glob("*.results.parquet"), key=lambda p: p.stat().st_mtime)
    )
    preds_df = pl.read_parquet(results_path)

    for split in ["train", "val", "test"]:
        print(f"\n=== {split} ===")
        print("--- Overall ---")
        print(summarize_overall(preds_df, split=split))
        print("--- By time bucket ---")
        summary = summarize_by_time_bucket(preds_df, split=split)
        print(summary)
        if split == "test":
            plot_summary(summary, out_path=str(TACKLER_RESULTS_DIR / f"tackler_accuracy_by_time_{split}2.png"))

if __name__ == "__main__":
    main()