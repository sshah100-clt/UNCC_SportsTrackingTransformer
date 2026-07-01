"""
Experiment 3: full power -- every non-leaky feature at once on hybrid_ts (M128/L4/W15).

This is the maximalist "unleash the black box" run: throw the entire available feature set
at the best architecture and see whether more signal helps or just adds noise. It composes
all three fusion mechanisms from Experiments 1 and 2 in a single model:

  player channels (37, through the GRU/attention):
      8  RAW_FEATURES
      4  engineered dynamics  (ax, ay, delta_ox, delta_oy)      } per-frame, from temporal_ext
      3  raw kinematics       (s, a, dis)                       }
     19  position one-hot                                       } static, broadcast over frames
      1  age                                                    }
      2  weight_Z, height_Z                                     }
  late-fused per-play context (18, fused after pooling):
     11  context   (offenseFormation 7, defendersInTheBox, passProbability, quarter, gameClock)
      4  situation (possession_score, defense_score, possession win-prob, expectedPoints)
      3  downyards (down, yardsToGo, distanceToGoal)            } the gamestate features, so
                                                                  everything is literally all

Model: GameStateHybridTS(feature_len=37, game_state_dim=18) -- the per-frame + static channels
flow through the encoder; the per-play context is split off and late-fused, exactly as in the
gamestate/context experiments. Excludes only the post-snap leakage fields and the redundant
o/dir/event channels (see DATASETS.md and the experiment plan).

The `combined` arm (winners-only forward selection) is intentionally NOT here yet: it is defined
by the Experiment 1 + 2 results, so it is built once those verdicts exist. This script runs the
`everything` and `off` arms now, at the fixed config.

Usage (one arm per GPU; seeds sequential in-process):
    uv run python src/run_full_power.py --arm off        --device 0
    uv run python src/run_full_power.py --arm everything  --device 1
    uv run python src/run_full_power.py --aggregate
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch.callbacks as callbacks
import numpy as np
import polars as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

# BDB2024_Dataset is imported so it resolves in this script's __main__ namespace when the
# temporal pickle (created with a __main__.BDB2024_Dataset class reference) is unpickled.
from datasets import RAW_FEATURE_COUNT, BDB2024_Dataset, load_datasets  # noqa: F401
from run_player_features import _POSITIONS, _player_static_maps, load_extended_temporal
from run_presnap_context import (
    CONTEXT_DIM,
    SITUATION_DIM,
    GameStateLitModel,
    _val_loss_from_ckpt,
    build_downyards_lookup,
    build_play_vectors,
    evaluate_test_ade,
)

CONFIG = dict(model_dim=128, num_layers=4, window=15)  # the 4.09 hybrid_ts winner

# Player-channel layout: temporal_ext (15) + position (19) + age (1) + weight/height (2).
PLAYER_FEATURE_LEN = 15 + len(_POSITIONS) + 1 + 2  # = 37
# Late-fused per-play layout: context (11) + situation (4) + downyards (3) = 18, so the
# everything arm carries literally every non-leaky play-level feature (the gamestate trio
# down/yardsToGo/distanceToGoal included).
DOWNYARDS_DIM = 3
GAME_STATE_DIM = CONTEXT_DIM + SITUATION_DIM + DOWNYARDS_DIM  # 11 + 4 + 3 = 18

ARMS = ("off", "everything")
RESULTS_DIR = Path("results/full_power")
# Shared baseline (off = plain hybrid_ts, 8 feat) is trained once and lives here; this script's
# aggregator reads its paired deltas against it. Run off from only one of the three experiments.
BASELINE_DIR = Path("results/baseline_off")
MODELS_DIR = Path("models")


# --------------------------------------------------------------------------------------
# Static per-player lookup for the everything arm: position one-hot + age + weight/height
# --------------------------------------------------------------------------------------
def build_static_lookup(ext_base) -> dict[tuple, np.ndarray]:
    """Map (gameId, playId, mirrored) -> float32 (22, 22): [position(19), age(1), weight_Z, height_Z].

    Read in the same nflId-sorted order the feature arrays use, so rows align to players. age and
    weight/height come from the player/game tables and the prepped partition respectively."""
    pos_map, birth_map, game_date, mean_birth = _player_static_maps()
    fp = ext_base.feature_df_partition  # has weight_Z/height_Z columns (full prepped frame)
    lut: dict[tuple, np.ndarray] = {}
    for (g, p, m), frames in ext_base.play_frames.items():
        sub = fp.loc[(g, p, m, frames[0])]
        nflids = list(sub.index)
        assert len(nflids) == 22, f"expected 22 players, got {len(nflids)} for {(g, p, m)}"
        pos = np.zeros((22, len(_POSITIONS)), dtype=np.float32)
        for i, nid in enumerate(nflids):
            pos[i, pos_map[nid]] = 1.0
        gd = game_date[g]
        age = np.array([(gd - (birth_map.get(nid) or mean_birth)) / 365.25 for nid in nflids], dtype=np.float32)
        wh = sub[["weight_Z", "height_Z"]].to_numpy(dtype=np.float32)
        assert wh.shape == (22, 2) and not np.isnan(wh).any(), f"bad weight/height for {(g, p, m)}"
        lut[(g, p, m)] = np.concatenate([pos, age.reshape(22, 1), wh], axis=1)  # (22, 22)
    return lut


def build_gamestate_lookup(ext_base, split: str) -> dict[tuple, np.ndarray]:
    """Map (gameId, playId, mirrored) -> float32 (18,): [context(11), situation(4), downyards(3)].

    context/situation are mirror-invariant per-play vectors from plays.csv; downyards
    (down, yardsToGo, distanceToGoal) is read from the prepped parquet via the gamestate
    experiment's own lookup, so the values match that experiment exactly."""
    ctx = build_play_vectors("context")
    sit = build_play_vectors("situation")
    downyards = build_downyards_lookup(split)  # (gameId, playId, mirrored) -> [down, yardsToGo, distanceToGoal]
    lut: dict[tuple, np.ndarray] = {}
    for (g, p, m) in ext_base.play_frames:
        lut[(g, p, m)] = np.concatenate([ctx[(g, p)], sit[(g, p)], downyards[(g, p, m)]]).astype(np.float32)
    return lut


class EverythingDataset(Dataset):
    """Produce x = [player(37), gamestate(18)] = (..., 22, 55) from temporal_ext windows.

    player = temporal_ext(15) ++ static(22); gamestate(18) is broadcast across players/frames and
    appended as the final channels so GameStateHybridTS splits it off after pooling."""

    def __init__(self, ext_base, static_lut, gs_lut):
        self.base = ext_base
        self.static_lut = static_lut
        self.gs_lut = gs_lut
        self.keys = ext_base.keys

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        feat, tgt = self.base[idx]  # (T, 22, 15)
        g, p, m, _ = self.base.keys[idx]
        st = self.static_lut[(g, p, m)]  # (22, 22)
        gs = self.gs_lut[(g, p, m)]  # (18,)
        t, players, _ = feat.shape
        st_b = np.broadcast_to(st, (t, players, st.shape[1]))
        gs_b = np.broadcast_to(gs, (t, players, gs.shape[0]))
        out = np.concatenate([feat, st_b, gs_b], axis=-1)  # (T, 22, 52)
        return out.astype(np.float32), tgt


# --------------------------------------------------------------------------------------
# Data prep + train/eval one seed
# --------------------------------------------------------------------------------------
def prepare_data(arm: str):
    wl = CONFIG["window"]
    if arm == "off":
        train_base = load_datasets("hybrid_ts", split="train", window_length=wl)
        val_base = load_datasets("hybrid_ts", split="val", window_length=wl)
        test_base = load_datasets("hybrid_ts", split="test", window_length=wl)
        return train_base, val_base, test_base, test_base, RAW_FEATURE_COUNT, 0

    train_base = load_extended_temporal("train", wl)
    val_base = load_extended_temporal("val", wl)
    test_base = load_extended_temporal("test", wl)
    train_ds = EverythingDataset(train_base, build_static_lookup(train_base), build_gamestate_lookup(train_base, "train"))
    val_ds = EverythingDataset(val_base, build_static_lookup(val_base), build_gamestate_lookup(val_base, "val"))
    eval_ds = EverythingDataset(test_base, build_static_lookup(test_base), build_gamestate_lookup(test_base, "test"))
    return train_ds, val_ds, eval_ds, test_base, PLAYER_FEATURE_LEN, GAME_STATE_DIM


def train_one_seed(arm, seed, device, num_workers, batch_size, patience, max_epochs, data):
    train_ds, val_ds, eval_ds, test_base, feature_len, gsd = data
    seed_everything(seed, workers=True)
    model_arm = "off" if arm == "off" else "on"

    model = GameStateLitModel(
        model_arm, CONFIG["model_dim"], CONFIG["num_layers"], CONFIG["window"],
        feature_len, gsd, learning_rate=1e-4, dropout=0.3,
    )

    version = f"{arm}_S{seed}"
    logger = TensorBoardLogger(save_dir=MODELS_DIR, name="full_power", version=version, log_graph=False)

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

    # off -> shared baseline dir (trained once, reused by all experiments); on-arms here.
    out_dir = BASELINE_DIR if arm == "off" else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    record = dict(
        model="hybrid_ts",
        arm=arm,
        seed=seed,
        feature_len=feature_len,
        game_state_dim=gsd,
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
    ).write_parquet(best_ckpt.with_suffix(".fp_test_preds.parquet"))

    print(f"\n[DONE] full_power arm={arm} seed={seed} | val_loss={val_loss:.4f} "
          f"test_ADE={test_ade:.4f} yd over {n_eval} frames | feature_len={feature_len} "
          f"game_state_dim={gsd} params={model.num_params:,}")
    return record


def aggregate():
    on_rows = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    off_rows = [json.loads(p.read_text()) for p in sorted(BASELINE_DIR.glob("off_S*.json"))]  # shared baseline
    if not on_rows and not off_rows:
        print("No per-run JSONs found in", RESULTS_DIR, "or", BASELINE_DIR)
        return
    import statistics as st

    if on_rows:  # CSV holds this experiment's own on-arms; off lives in the shared baseline dir
        pl.DataFrame(on_rows).drop("best_ckpt").write_csv("results/full_power_experiment.csv")
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
    seeds = sorted(set(off) & set(by_arm.get("everything", {})))
    if seeds:
        deltas = [by_arm["everything"][s] - off[s] for s in seeds]
        print(f"  paired (everything-off) seeds {seeds}: mean delta = {st.mean(deltas):+.4f}  (negative = helps)")
    print("\nWrote results/full_power_experiment.csv")


if __name__ == "__main__":
    parser = ArgumentParser(description="Experiment 3: full-power (all features) on hybrid_ts")
    parser.add_argument("--arm", choices=ARMS, help="off / everything")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
    else:
        assert args.arm, "provide --arm (or --aggregate)"
        data = prepare_data(args.arm)
        for seed in args.seeds:
            print(f"\n===== full_power arm={args.arm} seed={seed} (device {args.device}) =====")
            train_one_seed(
                args.arm, seed, args.device,
                args.num_workers, args.batch_size, args.patience, args.max_epochs, data,
            )
