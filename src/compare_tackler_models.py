"""
Compare tackler-ID accuracy across multiple trained models, overlaid by time-before-tackle bucket.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from tackler_bucket_summary import summarize_by_time_bucket, TACKLER_RESULTS_DIR

# label -> results file
RUNS = {
    "stgnn_ts (knn_hybrid_hub)": "stgnn_ts_M128_L8_W10_knn_hybrid_hub_K4_LR1e-04_epoch=10-val_loss=1.173.results.parquet",
    "stgnn_st (knn_hybrid_hub)": "stgnn_st_M128_L8_W10_knn_hybrid_hub_K4_LR1e-04_epoch=4-val_loss=1.316.results.parquet",
    "hybrid_ts": "hybrid_ts_M128_L8_W10_LR1e-04_epoch=12-val_loss=1.194.results.parquet",
    "hybrid_st": "hybrid_st_M128_L8_W10_LR1e-04_epoch=4-val_loss=1.349.results.parquet",
    "windowed_transformer": "windowed_transformer_M128_L8_W10_LR1e-04_epoch=15-val_loss=1.260.results.parquet",
    "pure_gru": "pure_gru_M128_L8_W10_LR1e-04_epoch=46-val_loss=1.766.results.parquet"
}
BASELINE_ACCURACY = [
    60.21,  # 0-0.5s
    72.44,  # 0.5-1s
    59.17,  # 1-1.5s
    42.23,  # 1.5-2s
    30.93,  # 2-2.5s
    20.34,  # 2.5-3s
    11.01,  # 3s+
]

SPLIT = "test"


def main():
    summaries = {}

    for label, filename in RUNS.items():
        path = TACKLER_RESULTS_DIR / filename
        df = pl.read_parquet(path)
        summaries[label] = summarize_by_time_bucket(df, split=SPLIT)

    # Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = next(iter(summaries.values()))["time_bucket"].to_list()
    x = range(len(labels))

    # Plot trained models
    for label, summary in summaries.items():
        ax.plot(
            x,
            summary["accuracy_%"].to_list(),
            marker="o",
            label=label,
        )

    # Plot closest-defender baseline
    ax.plot(
        x,
        BASELINE_ACCURACY,
        marker="o",
        linestyle="--",
        label="closest defender baseline",
    )

    ax.set_xticks(list(x), labels)
    ax.set_xlabel("Time before tackle")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Tackler prediction accuracy by model ({SPLIT} split)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        TACKLER_RESULTS_DIR / "model_comparison_accuracy.png",
        dpi=150,
    )
    print("Saved model_comparison_accuracy.png")

    # Confidence-in-correct-player comparison.
    # Baseline is NOT included here because it has no probability/
    # confidence metric.
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, summary in summaries.items():
        ax.plot(
            x,
            summary["confidence_in_correct_player_%"].to_list(),
            marker="o",
            label=label,
        )

    ax.set_xticks(list(x), labels)
    ax.set_xlabel("Time before tackle")
    ax.set_ylabel("Avg. probability assigned to true tackler (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Model confidence in true tackler by time ({SPLIT} split)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        TACKLER_RESULTS_DIR / "model_comparison_confidence.png",
        dpi=150,
    )
    print("Saved model_comparison_confidence.png")

    # Print raw tables
    for label, summary in summaries.items():
        print(f"\n=== {label} ===")
        print(summary)

    print("\n=== closest defender baseline ===")
    baseline_df = pl.DataFrame(
        {
            "time_bucket": labels,
            "baseline_accuracy_%": BASELINE_ACCURACY,
        }
    )
    print(baseline_df)


if __name__ == "__main__":
    main()