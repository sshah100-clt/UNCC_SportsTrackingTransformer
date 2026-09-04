"""
Dataset Module for NFL Big Data Bowl 2024

This module handles data loading and preprocessing for tackle prediction models.
It implements two distinct feature engineering approaches:

1. Transformer Model: Minimal feature engineering, providing raw player features
   to leverage self-attention for learning spatial relationships end-to-end.

2. Zoo Model: Complex pairwise feature engineering creating a 10x11 grid of
   offensive-defensive player interactions, following the architecture that won
   the 2020 NFL Big Data Bowl.

The key insight is that Transformer models can learn these interaction patterns
automatically, while Zoo models require manual feature engineering.

Classes:
    BDB2024_Dataset: Custom PyTorch dataset class for NFL tracking data

Functions:
    load_datasets: Load preprocessed datasets from disk
    main: Precompute and cache datasets for all splits and model types

Usage:
    dataset = load_datasets('transformer', 'train')
    features, targets = dataset[0]  # Get first sample
"""

import multiprocessing as mp
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from torch.utils.data import Dataset
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

PREPPED_DATA_DIR = Path("data/split_prepped_data/")
DATASET_DIR = Path("data/datasets/")

TRANSFORMER_FEATURES = [
    "x_rel",
    "y_rel",
    "vx",
    "vy",
    "ax",
    "ay",
    "ox",
    "oy",
    "delta_ox",
    "delta_oy",
    "side",
    "is_ball_carrier",
]
ZOO_INTERACTION_FEATURE_COUNT = 28

# Raw (non-engineered) per-player features used by the temporal sequence models.
# These are sensor-measured / positional quantities only -- no backward-difference
# features (ax, ay, delta_ox, delta_oy), because recurrent models learn dynamics
# from the frame sequence itself rather than from hand-computed derivatives.
RAW_FEATURES = [
    "x_rel",
    "y_rel",
    "vx",
    "vy",
    "ox",
    "oy",
    "side",
    "is_ball_carrier",
]
RAW_FEATURE_COUNT = len(RAW_FEATURES)

# Sequence model types. All consume a window of T frames, shape (T, 22, F), built
# on RAW_FEATURES, and share a single precomputed dataset (DATASET_DIR / "temporal").
TEMPORAL_MODEL_TYPES = ["windowed_transformer", "pure_gru", "hybrid_ts", "hybrid_st", "stgnn_ts", "stgnn_st"]
TEMPORAL_DATASET_NAME = "temporal"


class BDB2024_Dataset(Dataset):
    """
    Custom dataset class for NFL tracking data.

    This class preprocesses and stores NFL tracking data for use in machine learning models.
    It supports both 'transformer' and 'zoo' model types.

    Attributes:
        model_type (str): Type of model ('transformer' or 'zoo')
        keys (list): List of unique identifiers for each data point
        feature_df_partition (pd.DataFrame): Preprocessed feature data
        tgt_df_partition (pd.DataFrame): Preprocessed target data
        tgt_arrays (dict): Precomputed target arrays
        feature_arrays (dict): Precomputed feature arrays
    """

    def __init__(
        self,
        model_type: str,
        feature_df: pl.DataFrame,
        tgt_df: pl.DataFrame,
    ):
        """
        Initialize the dataset.

        Args:
            model_type (str): Type of model ('transformer' or 'zoo')
            feature_df (pl.DataFrame): DataFrame containing feature data
            tgt_df (pl.DataFrame): DataFrame containing target data

        Raises:
            ValueError: If an invalid model_type is provided
        """
        valid_types = ["transformer", "zoo"] + TEMPORAL_MODEL_TYPES
        if model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")

        self.model_type = model_type
        # Temporal models use the raw feature set; transformer uses the engineered set; zoo builds its own grid.
        if model_type in TEMPORAL_MODEL_TYPES:
            self.feature_list = RAW_FEATURES
            self.feature_len = RAW_FEATURE_COUNT
        elif model_type == "transformer":
            self.feature_list = TRANSFORMER_FEATURES
            self.feature_len = len(TRANSFORMER_FEATURES)
        else:  # zoo
            self.feature_list = None
            self.feature_len = ZOO_INTERACTION_FEATURE_COUNT

        # Window length for temporal models, set after loading via load_datasets().
        # window_length == 1 yields the original single-frame (22, F) behavior.
        self.window_length = 1

        # Sort keys to ensure deterministic ordering across runs
        self.keys = sorted(feature_df.select(["gameId", "playId", "mirrored", "frameId"]).unique().rows())

        # Per-play ordered frame index for on-the-fly windowing. Keys are sorted as
        # (gameId, playId, mirrored, frameId) tuples, so frameIds within each play are
        # already ascending. play_frames maps a play to its ordered frameIds; key_pos
        # maps each full key to its position within that play.
        self.play_frames: dict[tuple, list] = {}
        self.key_pos: dict[tuple, int] = {}
        for g, p, m, f in self.keys:
            self.play_frames.setdefault((g, p, m), []).append(f)
        for (g, p, m), frames in self.play_frames.items():
            for i, f in enumerate(frames):
                self.key_pos[(g, p, m, f)] = i

        # Convert to pandas form with index for quick row retrieval
        self.feature_df_partition = (
            feature_df.to_pandas(use_pyarrow_extension_array=True)
            .set_index(["gameId", "playId", "mirrored", "frameId", "nflId"])
            .sort_index()
        )
        self.tgt_df_partition = (
            tgt_df.to_pandas(use_pyarrow_extension_array=True)
            .set_index(["gameId", "playId", "mirrored", "frameId"])
            .sort_index()
        )

        # Precompute features and store in dicts
        # Note: Using pool.map() preserves input order, ensuring deterministic dictionary construction
        self.tgt_arrays: dict[tuple, np.ndarray] = {}
        self.feature_arrays: dict[tuple, np.ndarray] = {}
        with mp.Pool(processes=min(8, mp.cpu_count())) as pool:
            results = pool.map(
                self.process_key,
                tqdm(self.keys, desc="Pre-computing feature transforms", total=len(self.keys)),
            )
            # Unpack results in the same order as self.keys (pool.map guarantees order)
            for key, tgt_array, feature_array in results:
                self.tgt_arrays[key] = tgt_array
                self.feature_arrays[key] = feature_array

    def process_key(self, key: tuple) -> tuple[tuple, np.ndarray, np.ndarray]:
        """
        Process a single key to generate target and feature arrays.

        Args:
            key (tuple): Key (gameId, playId, mirrored, frameId) identifying a specific data point

        Returns:
            tuple[tuple, np.ndarray, np.ndarray]: Processed key, target array, and feature array
        """
        tgt_array = self.transform_target_df(self.tgt_df_partition.loc[key])
        feature_array = self.transform_input_frame_df(self.feature_df_partition.loc[key])
        return key, tgt_array, feature_array

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            tuple[np.ndarray, np.ndarray]: Feature array and target array for the specified index

        Raises:
            IndexError: If the index is out of range
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        key = self.keys[idx]

        # getattr fallback keeps datasets pickled before windowing existed (which lack
        # the window_length attribute) working as single-frame (22, F) datasets.
        if getattr(self, "window_length", 1) > 1:
            return self._get_window(key), self.tgt_arrays[key]
        return self.feature_arrays[key], self.tgt_arrays[key]

    def _get_window(self, key: tuple) -> np.ndarray:
        """
        Build a temporal window of the T frames ending at `key`, shape (T, 22, F).

        Frames are gathered from the same (gameId, playId, mirrored) play, ordered
        oldest -> newest (index -1 is the current frame). When fewer than T frames
        precede the current frame (start of a play), the earliest available frame is
        repeated (edge padding), so no out-of-play or fake-zero frames are introduced.
        """
        g, p, m, _ = key
        frames = self.play_frames[(g, p, m)]
        pos = self.key_pos[key]
        window = []
        for offset in range(self.window_length - 1, -1, -1):
            src_pos = max(pos - offset, 0)  # edge-pad with the earliest frame
            window.append(self.feature_arrays[(g, p, m, frames[src_pos])])
        return np.stack(window, axis=0)

    def transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        """
        Transform input frame DataFrame to numpy array based on model type.

        Args:
            frame_df (pd.DataFrame): Input frame DataFrame

        Returns:
            np.ndarray: Transformed input features

        Raises:
            ValueError: If an unknown model type is specified
        """
        if self.model_type == "zoo":
            return self.zoo_transform_input_frame_df(frame_df)
        # transformer and all temporal model types use the per-player (22, F) layout
        return self.transformer_transform_input_frame_df(frame_df)

    def transform_target_df(self, tgt_df: pd.DataFrame) -> np.ndarray:
        """
        Transform target DataFrame to numpy array.

        Args:
            tgt_df (pd.DataFrame): Target DataFrame

        Returns:
            np.ndarray: Transformed target values

        Raises:
            AssertionError: If the output shape is not as expected
        """
        y = tgt_df[["tackle_x_rel", "tackle_y_rel"]].to_numpy(dtype=np.float32).squeeze()
        assert y.shape == (2,), f"Expected shape (2,), got {y.shape}"
        return y

    def transformer_transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        """
        Transform input frame DataFrame for transformer model.

        Args:
            frame_df (pd.DataFrame): Input frame DataFrame

        Returns:
            np.ndarray: Transformed input features for transformer model

        Raises:
            AssertionError: If the output shape is not as expected
        """
        features = self.feature_list
        x = frame_df[features].to_numpy(dtype=np.float32)
        assert x.shape == (22, len(features)), f"Expected shape (22, {len(features)}), got {x.shape}"
        return x

    def zoo_transform_input_frame_df(self, frame_df: pd.DataFrame) -> np.ndarray:
        """
        Transform input frame DataFrame for zoo model.

        Args:
            frame_df (pd.DataFrame): Input frame DataFrame

        Returns:
            np.ndarray: Transformed input features for zoo model

        Raises:
            AssertionError: If the output shape is not as expected
        """
        # Isolate offensive and defensive players
        ball_carrier = frame_df[frame_df["is_ball_carrier"] == 1]
        off_plyrs = frame_df[(frame_df["side"] == 1) & (frame_df["is_ball_carrier"] == 0)]
        def_plyrs = frame_df[frame_df["side"] == -1]

        # Position & velocity (original features)
        ball_carr_mvmt = ball_carrier[["x_rel", "y_rel", "vx", "vy"]].to_numpy(dtype=np.float32).squeeze()
        off_mvmt = off_plyrs[["x_rel", "y_rel", "vx", "vy"]].to_numpy(dtype=np.float32)
        def_mvmt = def_plyrs[["x_rel", "y_rel", "vx", "vy"]].to_numpy(dtype=np.float32)

        # Acceleration (temporal)
        ball_carr_accel = ball_carrier[["ax", "ay"]].to_numpy(dtype=np.float32).squeeze()
        off_accel = off_plyrs[["ax", "ay"]].to_numpy(dtype=np.float32)
        def_accel = def_plyrs[["ax", "ay"]].to_numpy(dtype=np.float32)

        # Orientation (static)
        ball_carr_orient = ball_carrier[["ox", "oy"]].to_numpy(dtype=np.float32).squeeze()
        off_orient = off_plyrs[["ox", "oy"]].to_numpy(dtype=np.float32)
        def_orient = def_plyrs[["ox", "oy"]].to_numpy(dtype=np.float32)

        # Orientation change rate (temporal)
        ball_carr_dorient = ball_carrier[["delta_ox", "delta_oy"]].to_numpy(dtype=np.float32).squeeze()
        off_dorient = off_plyrs[["delta_ox", "delta_oy"]].to_numpy(dtype=np.float32)
        def_dorient = def_plyrs[["delta_ox", "delta_oy"]].to_numpy(dtype=np.float32)

        # Zoo interaction features — 3-tier pattern per quantity:
        #   Tier 1: Defender raw (tiled across offense)
        #   Tier 2: Defender − ball carrier (tiled across offense)
        #   Tier 3: Offense − defense (pairwise)
        x = [
            # --- Original: velocity & position (10 features) ---
            # def_vx, def_vy
            np.tile(def_mvmt[:, 2:], (10, 1, 1)),
            # def_pos - ball_pos
            np.tile(def_mvmt[None, :, :2] - ball_carr_mvmt[None, None, :2], (10, 1, 1)),
            # def_vel - ball_vel
            np.tile(def_mvmt[None, :, 2:] - ball_carr_mvmt[None, None, 2:], (10, 1, 1)),
            # off_pos - def_pos
            off_mvmt[:, None, :2] - def_mvmt[None, :, :2],
            # off_vel - def_vel
            off_mvmt[:, None, 2:] - def_mvmt[None, :, 2:],
            # --- Acceleration interactions (6 features) ---
            np.tile(def_accel, (10, 1, 1)),
            np.tile(def_accel[None, :] - ball_carr_accel[None, None, :], (10, 1, 1)),
            off_accel[:, None, :] - def_accel[None, :, :],
            # --- Orientation interactions (6 features) ---
            np.tile(def_orient, (10, 1, 1)),
            np.tile(def_orient[None, :] - ball_carr_orient[None, None, :], (10, 1, 1)),
            off_orient[:, None, :] - def_orient[None, :, :],
            # --- Orientation change interactions (6 features) ---
            np.tile(def_dorient, (10, 1, 1)),
            np.tile(def_dorient[None, :] - ball_carr_dorient[None, None, :], (10, 1, 1)),
            off_dorient[:, None, :] - def_dorient[None, :, :],
        ]

        x = np.concatenate(x, dtype=np.float32, axis=-1)

        assert x.shape == (
            10,
            11,
            ZOO_INTERACTION_FEATURE_COUNT,
        ), f"Expected shape (10, 11, {ZOO_INTERACTION_FEATURE_COUNT}), got {x.shape}"
        return x


def load_datasets(model_type: str, split: str, window_length: int = 1) -> BDB2024_Dataset:
    """
    Load datasets for a specific model type and data split.

    Args:
        model_type (str): Type of model ('transformer', 'zoo', or a temporal type).
        split (str): Data split ('train', 'val', or 'test')
        window_length (int): Temporal window length T to apply when serving samples.
            Only meaningful for temporal model types; ignored (left at 1) otherwise.

    Returns:
        BDB2024_Dataset: Loaded dataset for the specified model type and split

    Raises:
        ValueError: If an unknown split is specified
        FileNotFoundError: If the dataset file is not found
    """
    # All temporal model types share one precomputed dataset built on RAW_FEATURES.
    ds_name = TEMPORAL_DATASET_NAME if model_type in TEMPORAL_MODEL_TYPES else model_type
    ds_dir = DATASET_DIR / ds_name
    file_path = ds_dir / f"{split}_dataset.pkl"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, "rb") as f:
        dataset = pickle.load(f)

    if model_type in TEMPORAL_MODEL_TYPES:
        dataset.window_length = window_length
    return dataset


def main():
    """
    Main function to create and save datasets for different model types and splits.

    Builds the zoo and transformer datasets (engineered/12-feature) and a single
    shared temporal dataset (RAW_FEATURES, served as windows by the sequence models).
    Existing pickles are skipped so re-running only builds what is missing.
    """
    # (model_type used to build, output directory name)
    build_specs = [
        ("zoo", "zoo"),
        ("transformer", "transformer"),
        # One shared raw windowed dataset for all temporal model types. The build-time
        # model_type only selects RAW_FEATURES; window_length is applied later at load.
        ("windowed_transformer", TEMPORAL_DATASET_NAME),
    ]
    for split in ["test", "val", "train"]:
        feature_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_features.parquet")
        tgt_df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_targets.parquet")
        for model_type, out_name in build_specs:
            out_dir = DATASET_DIR / out_name
            out_path = out_dir / f"{split}_dataset.pkl"
            if out_path.exists():
                print(f"Skipping existing dataset: {out_path}")
                continue
            print(f"Creating dataset for {out_name=} ({model_type=}), {split=}...")
            tic = time.time()
            dataset = BDB2024_Dataset(model_type, feature_df, tgt_df)
            out_dir.mkdir(exist_ok=True, parents=True)
            with open(out_path, "wb") as f:
                pickle.dump(dataset, f)
            print(f"Took {(time.time() - tic)/60:.1f} mins")


if __name__ == "__main__":
    main()
