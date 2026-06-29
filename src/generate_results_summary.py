"""
Generate Results Summary and Analysis

This module performs comprehensive analysis of trained model performance, generating
publication-ready figures, tables, and metrics for comparing different architectures.

Key Analyses:
1. Overall Performance Comparison: Calculates Average Displacement Error (ADE) across
   all data splits for each model architecture.

2. Event-Type Breakdown: Analyzes model performance at different game moments
   (snap, handoff, tackle, etc.) to understand where models excel or struggle.

3. Temporal Analysis: Examines how prediction accuracy changes as plays progress,
   measuring performance at different frames before the tackle event.

4. Model Scaling Analysis: Compares all trained models across different configurations
   to understand how each architecture responds to increased model capacity.

5. Computational Efficiency: Calculates FLOPs for inference to compare computational
   costs across different model sizes and architectures.

Outputs:
- results/results.csv: Comprehensive metrics table for all analyses
- results/model_comparison.json: Details of all trained model configurations
- results/frame_difference_plot.png: Temporal performance visualization
- results/model_scaling_plot.png: Model capacity vs. performance comparison

Usage:
    uv run python src/generate_results_summary.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from calflops import calculate_flops

from datasets import TEMPORAL_MODEL_TYPES
from models import LitModel
from train import get_epoch_val_loss_from_ckpt

# All model types that may have trained checkpoints / best models on disk.
ALL_MODEL_TYPES = ["zoo", "transformer"] + TEMPORAL_MODEL_TYPES
STGNN_TYPES = ["stgnn_ts", "stgnn_st"]


def calculate_ade(
    x: pl.Series | np.ndarray,
    y: pl.Series | np.ndarray,
    x_pred: pl.Series | np.ndarray,
    y_pred: pl.Series | np.ndarray,
) -> float:
    """
    Calculate Average Displacement Error (ADE).

    ADE = mean Euclidean distance between predicted and true (x, y) locations.
    Standard metric for trajectory prediction, pose estimation, and spatial tasks.

    Formula: mean(sqrt((x_pred - x)² + (y_pred - y)²))

    Args:
        x: True x coordinates
        y: True y coordinates
        x_pred: Predicted x coordinates
        y_pred: Predicted y coordinates

    Returns:
        Average displacement error in the same units as input coordinates (yards)
    """
    if isinstance(x, pl.Series):
        x = x.to_numpy()
    if isinstance(y, pl.Series):
        y = y.to_numpy()
    if isinstance(x_pred, pl.Series):
        x_pred = x_pred.to_numpy()
    if isinstance(y_pred, pl.Series):
        y_pred = y_pred.to_numpy()

    distances = np.sqrt((x_pred - x) ** 2 + (y_pred - y) ** 2)
    return float(np.mean(distances))


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
TOPOLOGY_RESULTS_DIR = Path("topology_results")
TOPOLOGY_RESULTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models/best_models")


def _best_model_result_paths() -> dict[str, Path]:
    """Map each model type with a best-model results parquet on disk to its path."""
    paths = {}
    for model_type in ALL_MODEL_TYPES:
        p = MODELS_DIR / model_type / "best_model_results.parquet"
        if p.exists():
            paths[model_type] = p
    return paths


def load_results() -> pl.DataFrame:
    """Load and combine results from every best model present, plus per-frame events."""
    print("Loading model results...")
    result_paths = _best_model_result_paths()
    if not result_paths:
        raise FileNotFoundError(f"No best_model_results.parquet found under {MODELS_DIR}")
    print(f"  Found best-model results for: {', '.join(result_paths)}")
    results_df = pl.concat(
        [pl.read_parquet(p) for p in result_paths.values()],
        how="diagonal",
    )

    # Load tracking data to get per-frame events
    print("Loading tracking data for per-frame events...")
    tracking_df = pl.read_parquet("data/split_prepped_data/*_features.parquet")

    # Join with tracking data to get per-frame events
    results_df = results_df.join(
        tracking_df.filter(pl.col("is_ball_carrier") == 1)
        .select(["x", "y", "gameId", "playId", "frameId", "mirrored", "event"])
        .rename({"x": "ball_carrier_x", "y": "ball_carrier_y"}),
        on=["gameId", "playId", "frameId", "mirrored"],
        how="inner",
    )

    # Filter to mirrored=False to avoid double-counting predictions
    #
    # During training, we augment data by horizontally flipping each play (data augmentation).
    # This gives us 2× more training examples from the same data, helping the model generalize.
    #
    # Example: Original play has ball carrier running right → tackle at x=30, y=25
    #          Mirrored play has ball carrier running left → tackle at x=70, y=25 (x is flipped)
    #
    # When evaluating, we only count each unique play once to avoid inflating our metrics.
    # Both the original and mirrored versions produce predictions, but we only evaluate
    # the original (mirrored=False) to get true performance on unique plays.
    results_df = results_df.filter(~pl.col("mirrored"))
    print(f"  Loaded {len(results_df):,} predictions (mirrored=False only)")
    return results_df


def _calculate_ade_for_df(df: pl.DataFrame) -> float:
    """
    Helper to calculate ADE from a DataFrame with prediction columns.

    Args:
        df: DataFrame with columns: tackle_x, tackle_y, tackle_x_pred, tackle_y_pred

    Returns:
        ADE in yards, rounded to 2 decimal places
    """
    ade = df.select(
        pl.map_groups(
            exprs=["tackle_x", "tackle_y", "tackle_x_pred", "tackle_y_pred"],
            function=lambda ls: calculate_ade(*ls),
            returns_scalar=True,
        )
    ).item()
    return round(ade, 2)


def _calculate_improvement_metrics(zoo_ade: float, transformer_ade: float) -> tuple[float, float]:
    """
    Calculate improvement metrics comparing Zoo baseline to Transformer model.

    Args:
        zoo_ade: ADE for Zoo model (baseline)
        transformer_ade: ADE for Transformer model

    Returns:
        Tuple of (improvement_pct, improvement_yards)
        - improvement_pct: Percentage improvement (positive = better)
        - improvement_yards: Absolute yards improvement
    """
    improvement_pct = round((zoo_ade - transformer_ade) / zoo_ade * 100, 1)
    improvement_yards = round(zoo_ade - transformer_ade, 2)
    return improvement_pct, improvement_yards


def _present_model_types(results_df: pl.DataFrame) -> list[str]:
    """Model types actually present in the results, in a stable canonical order."""
    present = set(results_df["model_type"].unique().to_list())
    return [m for m in ALL_MODEL_TYPES if m in present]


def _add_model_type_ades(row: dict, df: pl.DataFrame, model_types: list[str]) -> dict:
    """Add a per-model-type ADE column to `row`; add zoo/transformer improvement if both present."""
    for model_type in model_types:
        model_df = df.filter(pl.col("model_type") == model_type)
        if len(model_df) > 0:
            row[model_type] = _calculate_ade_for_df(model_df)
    if "zoo" in row and "transformer" in row:
        row["improvement_pct"], row["improvement_yards"] = _calculate_improvement_metrics(
            row["zoo"], row["transformer"]
        )
    return row


def calculate_results(results_df: pl.DataFrame) -> list[dict]:
    """Calculate results for all splits and events."""
    print("\nCalculating results...")

    results = []
    model_types = _present_model_types(results_df)

    # Main splits
    for split in ["train", "val", "test"]:
        split_df = results_df.filter(pl.col("dataset_split") == split)

        row = {"split": split, "metric": "ade_yards"}
        _add_model_type_ades(row, split_df, model_types)
        row["n_plays"] = split_df.select(pl.struct(["gameId", "playId"]).n_unique()).item()
        row["n_frames"] = split_df.select(pl.len()).item()

        results.append(row)

    # Test event breakdowns (using per-frame events)
    print("Calculating test set event breakdowns (per-frame events)...")
    test_df = results_df.filter(pl.col("dataset_split") == "test")
    events = sorted(test_df.filter(pl.col("event").is_not_null())["event"].unique().to_list())

    event_results = []
    for event in events:
        # Use per-frame events (all frames with this event)
        event_df = test_df.filter(pl.col("event") == event)

        # Skip events with too few plays
        n_plays = event_df.select(pl.struct(["gameId", "playId"]).n_unique()).item()
        if n_plays < 100:
            continue

        row = {"split": f"test-event-{event}", "metric": "ade_yards"}
        _add_model_type_ades(row, event_df, model_types)
        row["n_plays"] = n_plays
        row["n_frames"] = event_df.select(pl.len()).item()
        row["_avg_frameId"] = round(event_df["frameId"].mean(), 1)  # For sorting only

        event_results.append(row)

    # Sort event results by avg_frameId, then remove the sorting key
    event_results.sort(key=lambda x: x["_avg_frameId"])
    for row in event_results:
        del row["_avg_frameId"]
    results.extend(event_results)

    return results


def calculate_frame_difference_results(results_df: pl.DataFrame) -> tuple[list[dict], pl.DataFrame]:
    """
    Calculate results by frame difference from tackle (test set only).

    Returns:
        tuple: (list of result dicts, DataFrame for plotting)
    """
    print("\nCalculating frame-difference breakdown (test set only)...")

    frame_diff_df = (
        results_df.with_columns(
            frame_difference_from_tackle=(pl.col("tackle_frameId") - pl.col("frameId")),
        )
        .with_columns(
            frame_difference_from_tackle_cat=(
                pl.col("frame_difference_from_tackle").cut(
                    breaks=range(0, 31, 5),
                    labels=["after tackle", "0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"],
                    left_closed=True,
                )
            )
        )
        .filter(pl.col("dataset_split") == "test")
        .group_by(["model_type", "frame_difference_from_tackle_cat"])
        .agg(
            order=pl.col("frame_difference_from_tackle").mean() * -1,
            n_frames=pl.len(),
            n_plays=pl.struct(["gameId", "playId"]).n_unique(),
            ade_yards=pl.map_groups(
                exprs=["tackle_x", "tackle_y", "tackle_x_pred", "tackle_y_pred"],
                function=lambda ls: round(calculate_ade(*ls), 2),
                returns_scalar=True,
            ),
        )
        .sort("frame_difference_from_tackle_cat")
    )

    # Convert to results format for JSON
    results = []
    categories = sorted(frame_diff_df["frame_difference_from_tackle_cat"].unique().to_list())
    model_types = _present_model_types(results_df)

    for category in categories:
        cat_df = frame_diff_df.filter(pl.col("frame_difference_from_tackle_cat") == category)

        row = {"split": f"test-frames-before-tackle-{category}", "metric": "ade_yards"}

        for model_type in model_types:
            model_data = cat_df.filter(pl.col("model_type") == model_type)
            if len(model_data) > 0:
                row[model_type] = model_data["ade_yards"].item()

        if "zoo" in row and "transformer" in row:
            row["improvement_pct"], row["improvement_yards"] = _calculate_improvement_metrics(
                row["zoo"], row["transformer"]
            )

        # Get n_plays and n_frames (should be same for both models)
        first_row = cat_df.row(0, named=True)
        row["n_plays"] = int(first_row["n_plays"])
        row["n_frames"] = int(first_row["n_frames"])

        results.append(row)

    return results, frame_diff_df


def generate_frame_difference_plot(frame_diff_df: pl.DataFrame) -> None:
    """Generate and save frame-difference plot."""
    print("\nGenerating frame-difference plot...")

    # Convert to pandas for plotting
    frame_diff_df_pandas = frame_diff_df.to_pandas()

    # Create the line plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=frame_diff_df_pandas,
        x="frame_difference_from_tackle_cat",
        y="ade_yards",
        hue="model_type",
        marker="o",
    )

    # Flip the x-axis
    plt.gca().invert_xaxis()

    # Customize the plot
    plt.title("Model Performance by Frames Before Tackle", fontsize=16)
    plt.xlabel("Frames Before Tackle", fontsize=12)
    plt.ylabel("Average Displacement Error (yards)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model Type", title_fontsize="12", fontsize="10")

    # Adjust layout and save
    plt.tight_layout()
    plot_path = RESULTS_DIR / "frame_difference_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {plot_path}")


def find_all_model_checkpoints() -> list[dict]:
    """
    Find all model checkpoints and group by configuration.

    Returns:
        list[dict]: List of config dicts with model_type, model_dim, num_layers, and best checkpoint path.
    """
    models_base = Path("models")
    configs = []

    # Non-temporal: M128_L2_LR1e-04. Temporal: M128_L2_W10_LR1e-04.
    # STGNN (topology probe): M128_L4_W10_bipartite_K4_LR5e-04.
    pattern = re.compile(r"M(\d+)_L(\d+)(?:_W(\d+))?(?:_([a-z_]+)_K(\d+))?_LR")

    for model_type in ALL_MODEL_TYPES:
        model_dir = models_base / model_type
        if not model_dir.exists():
            continue

        # Find all config directories. Non-temporal: M128_L2_LR1e-04.
        # Temporal: M128_L2_W10_LR1e-04 (W{T} is the window length).
        for config_dir in model_dir.iterdir():
            if not config_dir.is_dir() or not config_dir.name.startswith("M"):
                continue

            # Parse config from directory name (window length optional)
            match = pattern.match(config_dir.name)
            if not match:
                continue

            model_dim = int(match.group(1))
            num_layers = int(match.group(2))
            window_length = int(match.group(3)) if match.group(3) else 1
            topology = match.group(4)
            knn_k = int(match.group(5)) if match.group(5) else None

            # Find best checkpoint (lowest val_loss)
            checkpoints_dir = config_dir / "checkpoints"
            if not checkpoints_dir.exists():
                continue

            checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
            if not checkpoint_files:
                continue

            # Parse val_loss from filename and find best
            best_checkpoint = None
            best_val_loss = float("inf")

            for ckpt_file in checkpoint_files:
                # Parse: epoch=X-val_loss=Y.YYY.ckpt
                val_loss_match = re.search(r"val_loss=([\d.]+?)\.ckpt", ckpt_file.name)
                if val_loss_match:
                    val_loss = float(val_loss_match.group(1))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint = ckpt_file

            if best_checkpoint:
                # Find corresponding results file
                results_file = best_checkpoint.with_suffix(".results.parquet")
                if results_file.exists():
                    configs.append(
                        {
                            "model_type": model_type,
                            "model_dim": model_dim,
                            "num_layers": num_layers,
                            "window_length": window_length,
                            "topology": topology,
                            "knn_k": knn_k,
                            "checkpoint_path": str(best_checkpoint),
                            "results_path": str(results_file),
                            "val_loss": best_val_loss,
                        }
                    )

    return configs

def calculate_stgnn_topology_results() -> list[dict]:
    """
    Build the per (ordering, topology) comparison table for the STGNN probe.

    Reads directly from each variant's checkpoint/results files (no best_models/
    selection step needed, since each (model_type, topology) combo only trains
    one fixed config).

    Returns:
        list[dict]: One row per (model_type, topology) with val/test ADE, params, best epoch.
    """
    print("\nCalculating STGNN topology probe results...")

    configs = find_all_model_checkpoints()
    stgnn_configs = [c for c in configs if c["model_type"] == "stgnn_ts" and c["topology"] is not None]

    if not stgnn_configs:
        print("  No STGNN topology checkpoints found, skipping.")
        return []

    rows = []
    for config in stgnn_configs:
        lit_model = LitModel.load_from_checkpoint(config["checkpoint_path"], map_location="cpu")
        params = int(lit_model.hparams["params"])
        epoch, val_loss = get_epoch_val_loss_from_ckpt(Path(config["checkpoint_path"]))

        preds_df = pl.read_parquet(config["results_path"])
        val_df = preds_df.filter((pl.col("dataset_split") == "val") & ~pl.col("mirrored"))
        test_df = preds_df.filter((pl.col("dataset_split") == "test") & ~pl.col("mirrored"))

        val_ade = _calculate_ade_for_df(val_df)
        test_ade = _calculate_ade_for_df(test_df)

        rows.append(
            {
                "variant": f"{config['model_type']}_{config['topology']}",
                "ordering": "ts" if config["model_type"] == "stgnn_ts" else "st",
                "topology": config["topology"],
                "knn_k": config["knn_k"],
                "val_ade": val_ade,
                "test_ade": test_ade,
                "params": params,
                "best_epoch": epoch,
                "val_loss": round(config["val_loss"], 4),
            }
        )

    rows.sort(key=lambda r: r["test_ade"])
    return rows


def calculate_config_sweep_results() -> list[dict]:
    """
    Build the config sweep results table for stgnn_ts / knn_hybrid_hub.
    One row per (model_dim, num_layers, window_length) config.
    """
    print("\nCalculating config sweep results...")

    configs = find_all_model_checkpoints()
    sweep_configs = [
        c for c in configs
        if c["model_type"] == "stgnn_ts"
        and c["topology"] == "knn_hybrid_hub"
        and c["knn_k"] == 4
    ]

    if not sweep_configs:
        print("  No config sweep checkpoints found, skipping.")
        return []

    rows = []
    for config in sweep_configs:
        lit_model = LitModel.load_from_checkpoint(config["checkpoint_path"], map_location="cpu")
        params = int(lit_model.hparams["params"])
        epoch, val_loss = get_epoch_val_loss_from_ckpt(Path(config["checkpoint_path"]))

        preds_df = pl.read_parquet(config["results_path"])
        val_df = preds_df.filter((pl.col("dataset_split") == "val") & ~pl.col("mirrored"))
        test_df = preds_df.filter((pl.col("dataset_split") == "test") & ~pl.col("mirrored"))

        rows.append({
            "model_dim": config["model_dim"],
            "num_layers": config["num_layers"],
            "window_length": config["window_length"],
            "knn_k": config["knn_k"],
            "val_ade": _calculate_ade_for_df(val_df),
            "test_ade": _calculate_ade_for_df(test_df),
            "params": params,
            "best_epoch": epoch,
            "val_loss": round(config["val_loss"], 4),
        })

    rows.sort(key=lambda r: r["test_ade"])
    return rows


def compute_model_metrics(checkpoint_path: str, model_type: str) -> dict:
    """
    Compute params and FLOPs for a single model checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint file.
        model_type (str): 'zoo', 'transformer', or a temporal type.

    Returns:
        dict: Metrics including params and inference_flops.
    """
    # Load model
    lit_model = LitModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model = lit_model.model
    model.eval()

    # Create dummy input shape using feature_len from checkpoint hparams
    feature_len = int(lit_model.hparams["feature_len"])
    if model_type in TEMPORAL_MODEL_TYPES:
        window_length = int(lit_model.hparams.get("window_length", 1))
        input_shape = (1, window_length, 22, feature_len)
    elif model_type == "transformer":
        input_shape = (1, 22, feature_len)
    else:  # zoo
        input_shape = (1, 10, 11, feature_len)

    # Calculate params
    params = int(lit_model.hparams["params"])

    # Calculate FLOPs using calflops
    # Note: We use calflops instead of fvcore because it properly counts
    # transformer attention operations (scaled_dot_product_attention),
    # which are critical for accurate FLOP comparison between models.
    try:
        flops, macs, _ = calculate_flops(
            model=model,
            input_shape=input_shape,
            print_results=False,
            output_as_string=False,
        )
        inference_flops = int(flops)
    except Exception:
        inference_flops = None

    return {"params": params, "inference_flops": inference_flops}


def compute_test_ade(results_path: str) -> float:
    """
    Compute test set ADE from results parquet file.

    Args:
        results_path (str): Path to results parquet file.

    Returns:
        float: Test set ADE in yards.
    """
    df = pl.read_parquet(results_path)
    test_df = df.filter((pl.col("dataset_split") == "test") & ~pl.col("mirrored"))

    ade = test_df.select(
        pl.map_groups(
            exprs=["tackle_x", "tackle_y", "tackle_x_pred", "tackle_y_pred"],
            function=lambda ls: calculate_ade(*ls),
            returns_scalar=True,
        )
    ).item()

    return float(ade)


def compute_model_comparison() -> list[dict]:
    """
    Compute comprehensive comparison of all trained models.

    Returns:
        list[dict]: List of model configs with params, FLOPs, and test ADE.
    """
    print("\nComputing comprehensive model comparison...")

    # Find all checkpoints
    configs = find_all_model_checkpoints()
    print(f"  Found {len(configs)} model configurations")

    results = []

    for i, config in enumerate(configs, 1):
        label = f"M{config['model_dim']}_L{config['num_layers']}"
        if config["topology"]:
            label += f"_{config['topology']}"
        print(
            f"  [{i}/{len(configs)}] Processing {config['model_type']} {label} "
        )

        # Compute metrics
        metrics = compute_model_metrics(config["checkpoint_path"], config["model_type"])
        test_ade = compute_test_ade(config["results_path"])

        results.append(
            {
                "model_type": config["model_type"],
                "model_dim": config["model_dim"],
                "num_layers": config["num_layers"],
                "window_length": config["window_length"],
                "topology": config["topology"],
                "knn_k": config["knn_k"],
                "params": metrics["params"],
                "inference_flops": metrics["inference_flops"],
                "test_ade_yards": round(test_ade, 2),
                "val_loss": round(config["val_loss"], 3),
            }
        )

    # Sort by model_type, then params
    results.sort(key=lambda x: (x["model_type"], x["params"]))

    return results


def generate_model_scaling_plot(model_comparison: list[dict]) -> None:
    """
    Generate model scaling plot showing Test ADE vs FLOPs.

    This visualization supports the "Model Selection and Architectural Scaling"
    section of the paper, demonstrating how Zoo and Transformer architectures
    scale with computational budget.
    """
    print("\nGenerating model scaling plot...")

    # Convert to DataFrame for easier manipulation
    df = pl.DataFrame(model_comparison).to_pandas()

    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Preferred styles for the original two architectures; others fall back to a palette.
    colors = {"zoo": "#FF7F0E", "transformer": "#1F77B4"}
    markers = {"zoo": "s", "transformer": "o"}
    fallback_palette = sns.color_palette("husl", len(ALL_MODEL_TYPES))
    fallback_markers = ["^", "D", "v", "P", "X", "*"]

    # Plot Test ADE vs FLOPs for every model type present, in canonical order.
    present = [m for m in ALL_MODEL_TYPES if (df["model_type"] == m).any()]
    for i, model_type in enumerate(present):
        data = df[df["model_type"] == model_type].sort_values("inference_flops")
        ax.plot(
            data["inference_flops"],
            data["test_ade_yards"],
            marker=markers.get(model_type, fallback_markers[i % len(fallback_markers)]),
            markersize=8,
            linewidth=2,
            label=model_type.replace("_", " ").title(),
            color=colors.get(model_type, fallback_palette[i]),
            alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference FLOPs (log scale)", fontsize=12)
    ax.set_ylabel("Test ADE (yards) - Lower is Better", fontsize=12)
    ax.set_title("Model Scaling: Test ADE vs FLOPs", fontsize=14, fontweight="bold")
    ax.legend(title="Architecture", fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Adjust layout and save
    plt.tight_layout()
    plot_path = RESULTS_DIR / "model_scaling_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {plot_path}")


def main():
    """Generate results summary."""
    print("=" * 60)
    print("GENERATING RESULTS")
    print("=" * 60)

    # STGNN topology probe table
    stgnn_rows = calculate_stgnn_topology_results()
    desired_topologies = {
        "knn_hybrid_hub",
        "knn_same_hub",
        "knn_hub",
        "knn_cross_hub",
    }
    desired_ks = {2, 4, 6, 8}

    stgnn_rows = [
        r
        for r in stgnn_rows
        if r["topology"] in desired_topologies
        and r["knn_k"] in desired_ks
    ]
    if stgnn_rows:
        topology_df = (
            pl.DataFrame(stgnn_rows)
            .select(
                [
                    "variant",
                    "ordering",
                    "topology",
                    "knn_k",
                    "val_ade",
                    "test_ade",
                    "params",
                    "best_epoch",
                    "val_loss",
                ]
            )
            .sort(["topology", "knn_k"])
        )
        output_path = TOPOLOGY_RESULTS_DIR / "topology_knn_sweep.csv"
        topology_df.write_csv(output_path)

        print(f"\n✓ Saved: {output_path}")
        print(topology_df)
    
    # Config sweep results table
    config_rows = calculate_config_sweep_results()
    if config_rows:
        config_df = pl.DataFrame(config_rows).sort(["model_dim", "num_layers", "window_length"])
        config_output_path = TOPOLOGY_RESULTS_DIR / "config_sweep_results.csv"
        config_df.write_csv(config_output_path)
        print(f"\n✓ Saved: {config_output_path} ({len(config_rows)} configs)")
        print(config_df)
        
    # rusn if best_models/ populated
    if not MODELS_DIR.exists() or not any(MODELS_DIR.iterdir()):
        print("\nNo models/best_models/ found -skipping standard results pipeline.")
        return
    try:
        results_df = load_results()
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("Skipping standard results pipeline.")
        return

    results = calculate_results(results_df)

    # Add frame-difference results (test only)
    frame_diff_results, frame_diff_df = calculate_frame_difference_results(results_df)
    results.extend(frame_diff_results)

    # Generate frame-difference plot
    generate_frame_difference_plot(frame_diff_df)

    # Save results CSV
    results_csv_path = RESULTS_DIR / "results.csv"
    results_pl_df = pl.DataFrame(results)
    results_pl_df.write_csv(results_csv_path)
    print(f"\n✓ Saved: {results_csv_path}")

    # Compute and save comprehensive model comparison
    model_comparison = compute_model_comparison()
    comparison_path = RESULTS_DIR / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(model_comparison, f, indent=2)
    print(f"\n✓ Saved: {comparison_path} ({len(model_comparison)} models)")

    # Generate model scaling plot
    generate_model_scaling_plot(model_comparison)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    # Print test set summary: per-model-type test ADE for whatever models are present.
    test_row = next(r for r in results if r["split"] == "test")
    present = [m for m in ALL_MODEL_TYPES if m in test_row]
    print("\nTest Set Overall (ADE yards):")
    for model_type in present:
        print(f"  {model_type.replace('_', ' ').title():20s}: {test_row[model_type]:.2f}")
    if "improvement_pct" in test_row:
        print(f"  Transformer vs Zoo: {test_row['improvement_yards']:.2f} yards ({test_row['improvement_pct']:.1f}%)")

    if any("improvement_pct" in r for r in results if r["split"].startswith("test-event-")):
        print("\nTest Set Events (Transformer vs Zoo):")
        for row in results:
            if row["split"].startswith("test-event-") and "improvement_pct" in row:
                event_name = row["split"].replace("test-event-", "")
                print(f"  {event_name:20s}: {row['improvement_pct']:5.1f}% improvement")


if __name__ == "__main__":
    main()
