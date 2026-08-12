"""
Tackler-ID Model Comparison

Evaluates every tackler prediction results file in tackler_results/
using ONLY the test split.

For each model, reports:
    - top-1 accuracy
    - top-3 accuracy
    - top-5 accuracy
    - average probability assigned to the true tackler
    - number of test examples

Results are saved to:
    tackler_results/model_tables/tackler_model_comparison.csv
    tackler_results/model_tables/tackler_model_comparison.parquet

Usage:
    python tackler_model_summary.py

The script automatically finds all:
    tackler_results/*.results.parquet
"""

from pathlib import Path

import polars as pl


TACKLER_RESULTS_DIR = Path("tackler_results")
MODEL_TABLES_DIR = TACKLER_RESULTS_DIR / "model_tables"

MODEL_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def add_true_tackler_prob(df: pl.DataFrame) -> pl.DataFrame:
    """Add probability assigned to the actual tackler."""

    def _lookup(row: dict) -> float:
        ids = row["nfl_ids"]
        probs = row["probs"]

        true_id = row["tacklerNflId"]

        # The true tackler should always be one of the 22 players.
        try:
            idx = ids.index(true_id)
        except ValueError:
            return 0.0

        return float(probs[idx])

    return df.with_columns(
        pl.struct(["nfl_ids", "probs", "tacklerNflId"])
        .map_elements(
            _lookup,
            return_dtype=pl.Float32,
        )
        .alias("true_tackler_prob")
    )


def add_topk_hits(
    df: pl.DataFrame,
    ks: list[int] = [1, 3, 5],
) -> pl.DataFrame:
    """
    Determine whether the true tackler is in the model's
    top-k predictions.
    """

    def _rank_of_true(row: dict) -> int:
        ids = row["nfl_ids"]
        probs = row["probs"]
        true_id = row["tacklerNflId"]

        try:
            true_idx = ids.index(true_id)
        except ValueError:
            # Should not happen, but make the row a guaranteed miss.
            return len(ids)

        true_prob = probs[true_idx]

        # 0 = highest probability / top-1
        return sum(1 for p in probs if p > true_prob)

    df = df.with_columns(
        pl.struct(["nfl_ids", "probs", "tacklerNflId"])
        .map_elements(
            _rank_of_true,
            return_dtype=pl.Int64,
        )
        .alias("true_tackler_rank")
    )

    for k in ks:
        df = df.with_columns(
            (pl.col("true_tackler_rank") < k).alias(f"top_{k}_hit")
        )

    return df


def summarize_model(
    results_path: Path,
    split: str = "test",
) -> dict:
    """
    Calculate the test-set summary for one model results file.
    """

    print(f"\nEvaluating: {results_path.name}")

    df = pl.read_parquet(results_path)

    # Only evaluate the requested split.
    df = df.filter(pl.col("dataset_split") == split)

    if df.is_empty():
        raise ValueError(
            f"No rows found for split='{split}' in {results_path}"
        )

    # Add metrics needed for evaluation.
    df = add_true_tackler_prob(df)
    df = add_topk_hits(df)

    summary = (
        df.select(
            [
                (pl.col("correct").mean() * 100)
                .alias("accuracy_%"),

                (pl.col("top_1_hit").mean() * 100)
                .alias("top_1_accuracy_%"),

                (pl.col("top_3_hit").mean() * 100)
                .alias("top_3_accuracy_%"),

                (pl.col("top_5_hit").mean() * 100)
                .alias("top_5_accuracy_%"),

                (pl.col("true_tackler_prob").mean() * 100)
                .alias("avg_confidence_in_correct_%"),

                pl.len().alias("n"),
            ]
        )
        .with_columns(
            pl.col("accuracy_%").round(2),
            pl.col("top_1_accuracy_%").round(2),
            pl.col("top_3_accuracy_%").round(2),
            pl.col("top_5_accuracy_%").round(2),
            pl.col("avg_confidence_in_correct_%").round(2),
        )
    )

    # There is only one row in the summary.
    row = summary.to_dicts()[0]

    # Pull model information directly from the results parquet.1\
    model_metadata = {}

    metadata_columns = [
        "model_type",
        "model_dim",
        "num_layers",
        "window_length",
        "topology",
        "edge_features",
        "knn_k",
        "learning_rate",
        "task",
    ]

    for column in metadata_columns:
        if column in df.columns:
            value = df[column][0]
            model_metadata[column] = value
        else:
            model_metadata[column] = None

    # Add filename so there is always an unambiguous reference
    # to the exact results file/checkpoint.
    model_metadata["results_file"] = results_path.name

    # Add the evaluation split.
    model_metadata["split"] = split

    # Put model metadata first, followed by metrics.
    return {
        **model_metadata,
        **row,
    }


def print_model_summary(summary_df: pl.DataFrame) -> None:
    """Print a readable comparison table."""

    print("\n" + "=" * 100)
    print("TACKLER-ID TEST SET MODEL COMPARISON")
    print("=" * 100)

    # Show the most useful comparison columns first.
    display_columns = [
        "model_type",
        "model_dim",
        "num_layers",
        "window_length",
        "topology",
        "edge_features",
        "knn_k",
        "accuracy_%",
        "top_3_accuracy_%",
        "top_5_accuracy_%",
        "avg_confidence_in_correct_%",
        "n",
    ]

    display_columns = [
        c for c in display_columns
        if c in summary_df.columns
    ]

    print(summary_df.select(display_columns))


def main():
    # Find every model result file.
    result_files = sorted(
        TACKLER_RESULTS_DIR.glob("*.results.parquet")
    )

    if not result_files:
        raise FileNotFoundError(
            f"No .results.parquet files found in {TACKLER_RESULTS_DIR}"
        )

    print(f"Found {len(result_files)} model result file(s).")

    # Evaluate every model.
    summaries = []

    for results_path in result_files:
        try:
            summary = summarize_model(
                results_path,
                split="test",
            )
            summaries.append(summary)

        except Exception as e:
            print(
                f"WARNING: Could not evaluate "
                f"{results_path.name}: {e}"
            )

    if not summaries:
        raise RuntimeError(
            "No model results could be evaluated."
        )

    # Create combined comparison table.
    comparison_df = pl.DataFrame(summaries)

    # Sort by top-1 accuracy, best model first.
    comparison_df = comparison_df.sort(
        "accuracy_%",
        descending=True,
    )

    # Save results.
    csv_path = MODEL_TABLES_DIR / "tackler_model_comparison.csv"
    parquet_path = MODEL_TABLES_DIR / "tackler_model_comparison.parquet"

    comparison_df.write_csv(csv_path)
    comparison_df.write_parquet(parquet_path)

    # Print results.
    print_model_summary(comparison_df)

    print("\nSaved model comparison:")
    print(f"  CSV:     {csv_path}")
    print(f"  Parquet: {parquet_path}")


if __name__ == "__main__":
    main()