"""
Experiment 1: per-player feature arms on the best hybrid_ts (M128/L4/W15).

Each arm appends per-player information to the 8 RAW_FEATURES and trains the plain
HybridTS at the resulting feature_len. Nothing is late-fused here: per-player signal flows
through the GRU/attention exactly like the existing physical (weight/height) experiment.

Arms (all vs the same 8-feature `off` baseline, paired by seed):
  off          8   baseline (existing 4.09 config, byte-identical to hybrid_ts off)
  position    27   + one-hot roster position (19), per-player static, broadcast over frames
  age          9   + age in years (1), per-player static, broadcast over frames
  dynamics    12   + ax, ay, delta_ox, delta_oy (engineered backward-diffs), PER-FRAME
  kinematics  11   + s, a, dis (raw kinematic scalars; s/dis ~redundant -> redundancy control), PER-FRAME
  position_bmi 29  + position (19) + weight_Z, height_Z (2): does size help in combination with role?
  dynamics_bmi 14  + dynamics (4) + weight_Z, height_Z (2): does size help in combination with dynamics?

The two *_bmi arms are targeted interaction tests: weight/height were dead alone in the physical
experiment, so these check whether they come alive combined with a new per-player feature.

Mechanism by feature kind:
  * Static arms (position, age) are play-constant, so a (22, k) lookup is broadcast across
    the T window frames and concatenated, just like the physical experiment.
  * Per-frame arms (dynamics, kinematics) vary every frame, so they cannot be broadcast.
    They are read from the extended temporal dataset (temporal_ext, 15 channels, built by
    build_extended_temporal.py) by SLICING channels -- which reuses the exact windowing
    behind the 4.09 baseline. temporal_ext channel order:
      0-7 RAW_FEATURES | 8 ax 9 ay 10 delta_ox 11 delta_oy | 12 s 13 a 14 dis
    dynamics = channels [0..11]; kinematics = channels [0..7, 12, 13, 14].

Config fixed at M128/L4/W15 (the 4.09 winner); only the features change. Same recipe as the
originals (AdamW, lr 1e-4, batch 256, dropout 0.3, SmoothL1, patience 10, 200 epochs).

Usage (one arm per GPU; seeds run sequentially in-process):
    uv run python src/run_player_features.py --arm off        --device 0
    uv run python src/run_player_features.py --arm position    --device 1
    uv run python src/run_player_features.py --arm age         --device 2
    uv run python src/run_player_features.py --arm dynamics    --device 3
    uv run python src/run_player_features.py --arm kinematics  --device 0
    uv run python src/run_player_features.py --arm position_bmi --device 1
    uv run python src/run_player_features.py --arm dynamics_bmi --device 2
    uv run python src/run_player_features.py --aggregate
"""

from __future__ import annotations

import json
import pickle
import re
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import lightning.pytorch.callbacks as callbacks
import numpy as np
import polars as pl
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from datasets import DATASET_DIR, PREPPED_DATA_DIR, RAW_FEATURE_COUNT, BDB2024_Dataset, load_datasets
from models import LitModel

CONFIG = dict(model_dim=128, num_layers=4, window=15)  # the 4.09 hybrid_ts winner

# Per-frame arms slice channels out of the extended temporal dataset (15 channels).
DYNAMICS_CHANNELS = list(range(12))  # raw 8 + ax, ay, delta_ox, delta_oy
KINEMATICS_CHANNELS = list(range(8)) + [12, 13, 14]  # raw 8 + s, a, dis

# Static per-player components that can be appended (broadcast over frames) and their widths.
STATIC_COMPONENT_DIM = {"position": 19, "age": 1, "bmi": 2}  # bmi = weight_Z, height_Z

# Each arm declares its base dataset, which per-frame channels to slice (None = standard 8
# raw features), and which static per-player components to append. The two *_bmi combos test
# whether weight/height -- dead alone in the physical experiment -- help in combination with a
# new per-player feature (the size x role / size x dynamics interaction).
ARM_SPEC = {
    "off":          dict(base="temporal", channels=None,               static=[]),
    "position":     dict(base="temporal", channels=None,               static=["position"]),
    "age":          dict(base="temporal", channels=None,               static=["age"]),
    "dynamics":     dict(base="ext",      channels=DYNAMICS_CHANNELS,   static=[]),
    "kinematics":   dict(base="ext",      channels=KINEMATICS_CHANNELS, static=[]),
    "position_bmi": dict(base="temporal", channels=None,               static=["position", "bmi"]),
    "dynamics_bmi": dict(base="ext",      channels=DYNAMICS_CHANNELS,   static=["bmi"]),
}
ARMS = tuple(ARM_SPEC)

# Feature columns recorded per arm (for the JSON; not all live in the prepped parquet).
ARM_FEATURES = {
    "off": [],
    "position": ["position_onehot(19)"],
    "age": ["age_years"],
    "dynamics": ["ax", "ay", "delta_ox", "delta_oy"],
    "kinematics": ["s", "a", "dis"],
    "position_bmi": ["position_onehot(19)", "weight_Z", "height_Z"],
    "dynamics_bmi": ["ax", "ay", "delta_ox", "delta_oy", "weight_Z", "height_Z"],
}

RESULTS_DIR = Path("results/player_features")
# Shared baseline: the off arm (plain hybrid_ts, 8 feat, M128/L4/W15) is identical across all
# three experiments, so it is trained ONCE and written here; every experiment's aggregator reads
# its paired deltas against this single baseline. Run off from any one script; the others skip it.
BASELINE_DIR = Path("results/baseline_off")
MODELS_DIR = Path("models")

PLAYERS_CSV = Path("data/bdb_2024/players.csv")
GAMES_CSV = Path("data/bdb_2024/games.csv")


# --------------------------------------------------------------------------------------
# Extended temporal loader (the per-frame arms read from temporal_ext, sliced by channel)
# --------------------------------------------------------------------------------------
def load_extended_temporal(split: str, window_length: int) -> BDB2024_Dataset:
    path = DATASET_DIR / "temporal_ext" / f"{split}_dataset.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Build it first: uv run python src/build_extended_temporal.py --check"
        )
    with open(path, "rb") as f:
        ds = pickle.load(f)
    ds.window_length = window_length
    return ds


# --------------------------------------------------------------------------------------
# Static per-player lookups (position one-hot, age), aligned to the dataset's player order
# --------------------------------------------------------------------------------------
# The complete set of 19 roster positions in players.csv (every tracked player has one).
# Hardcoded so importing this module does not depend on the CWD / CSV being present.
_POSITIONS = ["C", "CB", "DB", "DE", "DT", "FB", "FS", "G", "ILB", "LS",
              "MLB", "NT", "OLB", "QB", "RB", "SS", "T", "TE", "WR"]
_POS_INDEX = {p: i for i, p in enumerate(_POSITIONS)}


def _player_static_maps() -> tuple[dict[int, int], dict[int, float], dict[int, float], float]:
    """Return (nflId->position index), (nflId->age-at-game placeholder is per-play), helpers.

    Age depends on the game date, so this returns nflId->birth ordinal and gameId->game ordinal;
    the per-play age is computed in build_static_lookup. Missing birthDates fall back to the mean.
    """
    players = pl.read_csv(PLAYERS_CSV)
    pos_map = {row["nflId"]: _POS_INDEX[row["position"]] for row in players.iter_rows(named=True)}

    def _parse_birth(s):
        # players.csv mixes two date formats (YYYY-MM-DD for ~71%, MM/DD/YYYY for the rest)
        # plus literal "NA". Try both; genuinely missing dates fall back to the mean age.
        for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(s, fmt).toordinal()
            except (TypeError, ValueError):
                continue
        return None

    birth_map = {row["nflId"]: _parse_birth(row["birthDate"]) for row in players.iter_rows(named=True)}
    known = [v for v in birth_map.values() if v is not None]
    mean_birth = float(np.mean(known)) if known else datetime(1990, 1, 1).toordinal()

    games = pl.read_csv(GAMES_CSV)
    game_date = {
        row["gameId"]: datetime.strptime(row["gameDate"], "%m/%d/%Y").toordinal()
        for row in games.iter_rows(named=True)
    }
    return pos_map, birth_map, game_date, mean_birth


def build_static_lookup(base: BDB2024_Dataset, components: list[str]) -> dict[tuple, np.ndarray]:
    """Map (gameId, playId, mirrored) -> float32 (22, sum(component dims)) for the requested
    static per-player components ('position', 'age', 'bmi'), concatenated in order.

    Read the 22 nflIds in the SAME sorted order the feature array uses (the partition is
    sort_index'd on ..., nflId), so the appended rows align to the player rows exactly.
    """
    pos_map, birth_map, game_date, mean_birth = _player_static_maps()
    fp = base.feature_df_partition  # MultiIndex (gameId, playId, mirrored, frameId, nflId)
    lut: dict[tuple, np.ndarray] = {}
    for (g, p, m), frames in base.play_frames.items():
        sub = fp.loc[(g, p, m, frames[0])]  # 22 rows, nflId-sorted (same order as features)
        nflids = list(sub.index)
        assert len(nflids) == 22, f"expected 22 players, got {len(nflids)} for {(g, p, m)}"
        parts = []
        for comp in components:
            if comp == "position":
                arr = np.zeros((22, len(_POSITIONS)), dtype=np.float32)
                for i, nid in enumerate(nflids):
                    arr[i, pos_map[nid]] = 1.0
            elif comp == "age":
                gd = game_date[g]
                ages = [(gd - (birth_map.get(nid) or mean_birth)) / 365.25 for nid in nflids]
                arr = np.array(ages, dtype=np.float32).reshape(22, 1)
            elif comp == "bmi":
                arr = sub[["weight_Z", "height_Z"]].to_numpy(dtype=np.float32)
                assert arr.shape == (22, 2) and not np.isnan(arr).any(), f"bad weight/height for {(g, p, m)}"
            else:
                raise ValueError(f"unknown static component {comp!r}")
            parts.append(arr)
        lut[(g, p, m)] = np.concatenate(parts, axis=1)
    return lut


# --------------------------------------------------------------------------------------
# Dataset wrapper: append a static lookup, OR slice per-frame channels (one or the other)
# --------------------------------------------------------------------------------------
class PlayerFeatureDataset(Dataset):
    """Wrap a base temporal dataset and produce the arm's per-player feature tensor.

    static append: lut is set, channels is None -> concat broadcast (22, k) onto (T, 22, F).
    per-frame slice: channels is set, lut is None -> select channels from (T, 22, 15).
    off: both None -> passthrough.
    """

    def __init__(self, base: BDB2024_Dataset, lut: dict[tuple, np.ndarray] | None, channels: list[int] | None):
        self.base = base
        self.lut = lut
        self.channels = channels
        self.keys = base.keys

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        feat, tgt = self.base[idx]
        if self.channels is not None:
            feat = feat[..., self.channels]
        if self.lut is not None:
            g, p, m, _ = self.base.keys[idx]
            add = self.lut[(g, p, m)]  # (22, k)
            if feat.ndim == 3:  # (T, 22, F)
                t, players, _ = feat.shape
                add = np.broadcast_to(add, (t, players, add.shape[1]))
            feat = np.concatenate([feat, add], axis=-1)
        return feat.astype(np.float32), tgt


# --------------------------------------------------------------------------------------
# Data prep (once per process), then train/eval one seed
# --------------------------------------------------------------------------------------
def prepare_data(arm: str):
    """Load the FULL datasets once and wrap them for this arm, per ARM_SPEC.

    Returns (train_ds, val_ds, eval_ds, test_base, feature_len)."""
    spec = ARM_SPEC[arm]
    wl = CONFIG["window"]
    if spec["base"] == "ext":  # per-frame arms read sliced channels from temporal_ext
        train_base = load_extended_temporal("train", wl)
        val_base = load_extended_temporal("val", wl)
        test_base = load_extended_temporal("test", wl)
    else:  # off / static arms use the standard temporal dataset (8 raw channels)
        train_base = load_datasets("hybrid_ts", split="train", window_length=wl)
        val_base = load_datasets("hybrid_ts", split="val", window_length=wl)
        test_base = load_datasets("hybrid_ts", split="test", window_length=wl)

    channels = spec["channels"]
    base_len = len(channels) if channels is not None else RAW_FEATURE_COUNT
    feature_len = base_len + sum(STATIC_COMPONENT_DIM[c] for c in spec["static"])

    def wrap(base):
        lut = build_static_lookup(base, spec["static"]) if spec["static"] else None
        return PlayerFeatureDataset(base, lut, channels)

    return wrap(train_base), wrap(val_base), wrap(test_base), test_base, feature_len


def _val_loss_from_ckpt(ckpt_path: Path) -> float:
    m = re.search(r"val_loss=([\d.]+)", Path(ckpt_path).name)
    return float(m.group(1).rstrip(".")) if m else float("nan")


def evaluate_test_ade(model, best_ckpt, eval_ds, base, device, num_workers):
    """Test ADE in yards over NON-mirrored frames only (anchor-relative == absolute ADE)."""
    loader = DataLoader(eval_ds, batch_size=1024, shuffle=False, num_workers=num_workers)
    pred_trainer = Trainer(accelerator="gpu", devices=[device], logger=False, enable_model_summary=False)
    preds = pred_trainer.predict(model, dataloaders=loader, ckpt_path=str(best_ckpt))
    preds = torch.cat(preds, dim=0).cpu().numpy()

    keys = base.keys
    mirrored = np.array([k[2] for k in keys], dtype=bool)
    targets = np.stack([base.tgt_arrays[k] for k in keys]).astype(np.float32)
    assert preds.shape[0] == targets.shape[0]

    mask = ~mirrored
    dist = np.sqrt(((preds[mask] - targets[mask]) ** 2).sum(axis=-1))
    return float(dist.mean()), int(mask.sum()), preds, keys, targets


def train_one_seed(arm, seed, device, num_workers, batch_size, patience, max_epochs, data):
    train_ds, val_ds, eval_ds, test_base, feature_len = data
    seed_everything(seed, workers=True)

    model = LitModel(
        model_type="hybrid_ts",
        batch_size=batch_size,
        model_dim=CONFIG["model_dim"],
        num_layers=CONFIG["num_layers"],
        feature_len=feature_len,
        learning_rate=1e-4,
        dropout=0.3,
        window_length=CONFIG["window"],
    )

    version = f"{arm}_S{seed}"
    logger = TensorBoardLogger(save_dir=MODELS_DIR, name="player_features", version=version, log_graph=False)

    # drop_last=True matches the originals (keeps samples-per-epoch comparable across arms).
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True, num_workers=num_workers)

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[device],
        sync_batchnorm=True,
        logger=logger,
        enable_model_summary=True,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=patience),
            callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, filename="{epoch}-{val_loss:.3f}"),
        ],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt = Path(trainer.checkpoint_callback.best_model_path)
    val_loss = _val_loss_from_ckpt(best_ckpt)
    test_ade, n_eval, preds, keys, targets = evaluate_test_ade(model, best_ckpt, eval_ds, test_base, device, num_workers)

    # off goes to the shared baseline dir (trained once, reused by all experiments); on-arms here.
    out_dir = BASELINE_DIR if arm == "off" else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    record = dict(
        model="hybrid_ts",
        arm=arm,
        seed=seed,
        feature_len=feature_len,
        added_features=ARM_FEATURES[arm],
        model_dim=CONFIG["model_dim"],
        num_layers=CONFIG["num_layers"],
        window=CONFIG["window"],
        params=model.num_params,
        val_loss=val_loss,
        test_ade=round(test_ade, 4),
        n_test_nonmirrored=n_eval,
        best_ckpt=str(best_ckpt),
    )
    (out_dir / f"{arm}_S{seed}.json").write_text(json.dumps(record, indent=2))

    keys_arr = np.array(keys)
    pl.DataFrame(
        {
            "gameId": keys_arr[:, 0],
            "playId": keys_arr[:, 1],
            "mirrored": keys_arr[:, 2].astype(bool),
            "frameId": keys_arr[:, 3],
            "pred_x_rel": preds[:, 0],
            "pred_y_rel": preds[:, 1],
            "tgt_x_rel": targets[:, 0],
            "tgt_y_rel": targets[:, 1],
        }
    ).write_parquet(best_ckpt.with_suffix(".pf_test_preds.parquet"))

    print(f"\n[DONE] player_features arm={arm} seed={seed} | val_loss={val_loss:.4f} "
          f"test_ADE={test_ade:.4f} yd over {n_eval} frames | feature_len={feature_len} params={model.num_params:,}")
    return record


def aggregate():
    """Collate per-run JSONs into results/player_features_experiment.csv with paired deltas vs off."""
    on_rows = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    off_rows = [json.loads(p.read_text()) for p in sorted(BASELINE_DIR.glob("off_S*.json"))]  # shared baseline
    if not on_rows and not off_rows:
        print("No per-run JSONs found in", RESULTS_DIR, "or", BASELINE_DIR)
        return
    import statistics as st

    if on_rows:  # CSV holds this experiment's own on-arms; off lives in the shared baseline dir
        pl.DataFrame([{k: v for k, v in r.items() if k != "added_features"} for r in on_rows]).write_csv(
            "results/player_features_experiment.csv"
        )
    rows = on_rows + off_rows  # on-arms + shared off, for the paired-delta summary below
    by_arm = {}
    for arm in ARMS:
        a = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["seed"])
        by_arm[arm] = {r["seed"]: r["test_ade"] for r in a}
        if a:
            ades = [r["test_ade"] for r in a]
            print(f"  {arm:11s} seeds={[r['seed'] for r in a]} mean={st.mean(ades):.4f} "
                  f"std={st.pstdev(ades) if len(ades) > 1 else 0:.4f}")
    off = by_arm.get("off", {})
    for arm in ARMS:
        if arm == "off":
            continue
        seeds = sorted(set(off) & set(by_arm.get(arm, {})))
        if seeds:
            deltas = [by_arm[arm][s] - off[s] for s in seeds]
            print(f"  paired ({arm}-off) seeds {seeds}: mean delta = {st.mean(deltas):+.4f}  (negative = helps)")
    print("\nWrote results/player_features_experiment.csv")


if __name__ == "__main__":
    parser = ArgumentParser(description="Experiment 1: per-player feature arms on hybrid_ts")
    parser.add_argument("--arm", choices=ARMS, help="which per-player feature arm to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="seeds to run sequentially")
    parser.add_argument("--device", type=int, default=0, help="GPU index")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--aggregate", action="store_true", help="collate per-run JSONs into a CSV and exit")
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
    else:
        assert args.arm, "provide --arm (or --aggregate)"
        data = prepare_data(args.arm)  # load full data + build lookup/slice ONCE
        for seed in args.seeds:
            print(f"\n===== player_features arm={args.arm} seed={seed} (device {args.device}) =====")
            train_one_seed(
                args.arm, seed, args.device,
                args.num_workers, args.batch_size, args.patience, args.max_epochs, data,
            )
