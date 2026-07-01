"""
Experiment 2: pre-snap play context (late-fusion) on the best hybrid_ts (M128/L4/W15).

The honest, leakage-free successor to the gamestate experiment. Every feature here is known
BEFORE the snap, so none of it can leak the tackle outcome. Two on-arms, paired vs `off`:

  off                baseline (plain hybrid_ts, 8 raw features)
  context            11 dims: offenseFormation one-hot (7), defendersInTheBox, passProbability,
                             quarter, gameClock (seconds remaining in quarter)
  situation           4 dims: possession_score, defense_score, possession win-probability,
                             expectedPoints (all pre-snap / non-leaky)
  context_downyards  14 dims: context (11) + the already-tested gamestate trio down, yardsToGo,
                             distanceToGoal (3) -- a targeted combination test of related pre-snap
                             structure (formation/box alongside down/distance)

Mechanism: identical late-fusion to the gamestate experiment -- the per-play context vector
is broadcast across players/frames, split off inside the model, and fused after player pooling
(it never goes through the per-player GRU/attention and is never averaged away). This script
carries its OWN copy of the late-fusion machinery (GameStateDataset / GameStateHybridTS /
GameStateLitModel) so it does not import from any other experiment script; only the feature
vectors differ from gamestate.

Nulls are imputed with TRAIN-split means (formation null -> all-zero one-hot), so val/test never
inform the imputation. Config fixed at M128/L4/W15; same recipe as the originals.

Usage (one arm per GPU; seeds sequential in-process):
    uv run python src/run_presnap_context.py --arm off               --device 0
    uv run python src/run_presnap_context.py --arm context           --device 1
    uv run python src/run_presnap_context.py --arm situation         --device 2
    uv run python src/run_presnap_context.py --arm context_downyards --device 3
    uv run python src/run_presnap_context.py --aggregate
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch.callbacks as callbacks
import numpy as np
import polars as pl
import torch
from lightning import LightningModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

# BDB2024_Dataset is imported so it resolves in this script's __main__ namespace when the
# temporal pickle (created with a __main__.BDB2024_Dataset class reference) is unpickled.
from datasets import PREPPED_DATA_DIR, RAW_FEATURE_COUNT, BDB2024_Dataset, load_datasets  # noqa: F401
from models import HybridTS, LitModel

CONFIG = dict(model_dim=128, num_layers=4, window=15)  # the 4.09 hybrid_ts winner

PLAYS_CSV = Path("data/bdb_2024/plays.csv")
GAMES_CSV = Path("data/bdb_2024/games.csv")

FORMATIONS = ["EMPTY", "I_FORM", "JUMBO", "PISTOL", "SHOTGUN", "SINGLEBACK", "WILDCAT"]  # 7, sorted
CONTEXT_DIM = len(FORMATIONS) + 4  # + defendersInTheBox, passProbability, quarter, gameClock_sec
SITUATION_DIM = 4  # possession_score, defense_score, possession_winprob, expectedPoints
DOWNYARDS_DIM = 3  # down, yardsToGo, distanceToGoal (the previously-tested gamestate trio)

# context_downyards is a targeted combination: the new pre-snap structure (context) fused with
# the already-tested gamestate trio, which are mechanistically related (formation/box vs down/distance).
ARMS = ("off", "context", "situation", "context_downyards")
ARM_DIM = {
    "off": 0,
    "context": CONTEXT_DIM,
    "situation": SITUATION_DIM,
    "context_downyards": CONTEXT_DIM + DOWNYARDS_DIM,
}

RESULTS_DIR = Path("results/presnap_context")
# Shared baseline (off = plain hybrid_ts, 8 feat) is trained once and lives here; this script's
# aggregator reads its paired deltas against it. Run off from only one of the three experiments.
BASELINE_DIR = Path("results/baseline_off")
MODELS_DIR = Path("models")

DOWNYARDS_FEATURES = ["down", "yardsToGo", "distanceToGoal"]  # the gamestate trio (in the prepped parquet)


# --------------------------------------------------------------------------------------
# Self-contained late-fusion machinery (the per-play vector is split off inside the model and
# fused after player pooling, so it never goes through the per-player GRU/attention). This is a
# verbatim copy of the gamestate experiment's late-fusion -- the new experiments carry their own
# copy so they do not import from any other experiment script.
# --------------------------------------------------------------------------------------
class GameStateDataset(Dataset):
    """Wrap a loaded BDB2024_Dataset and append a per-play vector (broadcast over players/frames).

    Returns (features, target) with features shaped (..., 22, F+G): the base window unchanged plus
    G constant game-state channels. The base dataset is untouched."""

    def __init__(self, base, lut: dict[tuple, np.ndarray]):
        self.base = base
        self.lut = lut
        self.keys = base.keys

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        feat, tgt = self.base[idx]
        g, p, m, _ = self.base.keys[idx]
        gs = self.lut[(g, p, m)]  # [G]
        if feat.ndim == 3:  # (T, 22, F)
            t, players, _ = feat.shape
            gs_b = np.broadcast_to(gs, (t, players, gs.shape[0]))
        else:  # (22, F)
            players, _ = feat.shape
            gs_b = np.broadcast_to(gs, (players, gs.shape[0]))
        return np.concatenate([feat, gs_b], axis=-1).astype(np.float32), tgt


def _wide_decoder(in_dim: int, model_dim: int, dropout: float) -> nn.Sequential:
    """Same structure as models._build_decoder, but the first Linear accepts in_dim."""
    return nn.Sequential(
        nn.Linear(in_dim, model_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(model_dim, model_dim // 4),
        nn.ReLU(),
        nn.LayerNorm(model_dim // 4),
        nn.Linear(model_dim // 4, 2),
    )


def _gs_branch(game_state_dim: int, gs_hidden: int) -> nn.Sequential:
    # BatchNorm handles the very different scales (down 1-4 vs distanceToGoal 1-99, etc.).
    return nn.Sequential(
        nn.BatchNorm1d(game_state_dim),
        nn.Linear(game_state_dim, gs_hidden),
        nn.ReLU(),
    )


class GameStateHybridTS(HybridTS):
    """hybrid_ts (time->space) with late-fused per-play context.

    Input: [B, T, 22, F+G]. The first F channels go through the parent's BatchNorm/GRU/attention/
    pool unchanged; the last G channels (constant across players & time) are pulled once and joined
    to the pooled embedding before a wider decoder."""

    def __init__(self, feature_len, model_dim, num_layers, dropout, window_length, game_state_dim):
        super().__init__(feature_len, model_dim, num_layers, dropout, window_length)
        self.player_feature_len = feature_len
        self.game_state_dim = game_state_dim
        gs_hidden = max(1, model_dim // 4)
        self.gs_branch = _gs_branch(game_state_dim, gs_hidden)
        self.decoder = _wide_decoder(model_dim + gs_hidden, model_dim, dropout)
        self.hyperparams = {**self.hyperparams, "game_state_dim": game_state_dim, "fusion": "late"}

    def forward(self, x: Tensor) -> Tensor:
        b, t, p, _ = x.size()
        f = self.player_feature_len
        xp = x[..., :f]  # [B,T,P,F]
        gs = x[:, -1, 0, f:]  # [B,G] -- last frame, any player (constant)
        xp = self.feature_norm_layer(xp.reshape(-1, f)).reshape(b, t, p, f)
        xp = xp.permute(0, 2, 1, 3).reshape(b * p, t, f)
        _, h_n = self.gru(xp)
        h = h_n[-1].reshape(b, p, -1)  # [B,P,M]
        h = self.transformer_encoder(h)  # attention across players
        h = torch.squeeze(self.player_pooling_layer(h.permute(0, 2, 1)), -1)  # [B,M]
        h = torch.cat([h, self.gs_branch(gs)], dim=-1)  # late fusion
        return self.decoder(h)


def build_model(arm, model_dim, num_layers, window, feature_len, game_state_dim, dropout):
    """off -> the plain HybridTS (provably the original architecture). on -> the late-fusion variant."""
    if arm == "on":
        return GameStateHybridTS(feature_len, model_dim, num_layers, dropout, window, game_state_dim)
    return HybridTS(feature_len, model_dim, num_layers, dropout, window)


class GameStateLitModel(LitModel):
    """Inherits LitModel's training/validation/predict steps and AdamW optimizer unchanged
    (identical recipe); only model construction differs. hybrid_ts only."""

    def __init__(self, arm, model_dim, num_layers, window, feature_len, game_state_dim,
                 learning_rate: float = 1e-4, dropout: float = 0.3):
        # Skip LitModel.__init__ (its model_type dispatch) but keep LightningModule setup.
        LightningModule.__init__(self)
        self.save_hyperparameters()
        self.model = build_model(arm, model_dim, num_layers, window, feature_len, game_state_dim, dropout)
        self.model_type = f"hybrid_ts_{arm}"  # for logging only; steps don't dispatch on it
        self.learning_rate = learning_rate
        self.loss_fn = nn.SmoothL1Loss()
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def _val_loss_from_ckpt(ckpt_path: Path) -> float:
    import re

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


def build_downyards_lookup(split: str) -> dict[tuple, np.ndarray]:
    """Map (gameId, playId, mirrored) -> float32 [down, yardsToGo, distanceToGoal], read from the
    prepped parquet (same source/values as the gamestate experiment). Asserts play-constant."""
    df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_features.parquet")
    sub = df.select(["gameId", "playId", "mirrored", *DOWNYARDS_FEATURES]).unique()
    per_play = sub.group_by(["gameId", "playId", "mirrored"]).len()
    assert per_play["len"].max() == 1, f"downyards not constant within a play in split={split}"
    lut: dict[tuple, np.ndarray] = {}
    for row in sub.iter_rows(named=True):
        lut[(row["gameId"], row["playId"], row["mirrored"])] = np.array(
            [row[c] for c in DOWNYARDS_FEATURES], dtype=np.float32
        )
    return lut


# --------------------------------------------------------------------------------------
# Per-play context / situation vectors, with TRAIN-mean imputation for nulls
# --------------------------------------------------------------------------------------
def _gameclock_to_seconds(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        mm, ss = s.split(":")
        return float(mm) * 60 + float(ss)
    except (ValueError, AttributeError):
        return None


def _train_play_ids() -> set[tuple]:
    df = pl.read_parquet(PREPPED_DATA_DIR / "train_features.parquet", columns=["gameId", "playId"]).unique()
    return set(map(tuple, df.iter_rows()))


def build_play_vectors(arm: str) -> dict[tuple, np.ndarray]:
    """Map (gameId, playId) -> float32 vector for the given arm.

    Numeric nulls are imputed with the mean over TRAIN plays only. Formation nulls become an
    all-zero one-hot (a learnable 'unknown' encoded as absence)."""
    plays = pl.read_csv(PLAYS_CSV, infer_schema_length=20000, null_values=["NA"])
    games = pl.read_csv(GAMES_CSV)
    home_team = {row["gameId"]: row["homeTeamAbbr"] for row in games.iter_rows(named=True)}

    plays = plays.with_columns(
        gameClock_sec=pl.col("gameClock").map_elements(_gameclock_to_seconds, return_dtype=pl.Float64),
    )

    train_ids = _train_play_ids()
    is_train = [(row["gameId"], row["playId"]) in train_ids for row in plays.iter_rows(named=True)]
    train_plays = plays.filter(pl.Series(is_train))

    def train_mean(col: str) -> float:
        return float(train_plays[col].drop_nulls().mean())

    if arm == "context":
        means = {c: train_mean(c) for c in ("defendersInTheBox", "passProbability", "quarter", "gameClock_sec")}
        out: dict[tuple, np.ndarray] = {}
        for row in plays.iter_rows(named=True):
            onehot = np.zeros(len(FORMATIONS), dtype=np.float32)
            if row["offenseFormation"] in FORMATIONS:
                onehot[FORMATIONS.index(row["offenseFormation"])] = 1.0
            scal = [
                row["defendersInTheBox"] if row["defendersInTheBox"] is not None else means["defendersInTheBox"],
                row["passProbability"] if row["passProbability"] is not None else means["passProbability"],
                row["quarter"] if row["quarter"] is not None else means["quarter"],
                row["gameClock_sec"] if row["gameClock_sec"] is not None else means["gameClock_sec"],
            ]
            out[(row["gameId"], row["playId"])] = np.concatenate([onehot, np.array(scal, dtype=np.float32)])
        return out

    # situation
    ep_mean = train_mean("expectedPoints")
    out = {}
    for row in plays.iter_rows(named=True):
        off_is_home = row["possessionTeam"] == home_team[row["gameId"]]
        poss_score = row["preSnapHomeScore"] if off_is_home else row["preSnapVisitorScore"]
        def_score = row["preSnapVisitorScore"] if off_is_home else row["preSnapHomeScore"]
        poss_wp = (
            row["preSnapHomeTeamWinProbability"] if off_is_home else row["preSnapVisitorTeamWinProbability"]
        )
        ep = row["expectedPoints"] if row["expectedPoints"] is not None else ep_mean
        out[(row["gameId"], row["playId"])] = np.array(
            [poss_score, def_score, poss_wp, ep], dtype=np.float32
        )
    return out


def build_lookup(base, arm: str, split: str) -> dict[tuple, np.ndarray]:
    """Map every (gameId, playId, mirrored) key in `base` to its per-play vector.

    All vectors are mirror-invariant (down/formation/scores don't change under y-mirroring), so
    the same vector serves both mirror copies. For context_downyards, the new context vector is
    concatenated with the gamestate trio read via the gamestate experiment's own lookup (so the
    down/yardsToGo/distanceToGoal values match that experiment exactly)."""
    if arm == "context_downyards":
        ctx = build_play_vectors("context")  # (gameId, playId) -> 11
        downyards = build_downyards_lookup(split)  # (gameId, playId, mirrored) -> [down, yardsToGo, distanceToGoal]
        return {
            (g, p, m): np.concatenate([ctx[(g, p)], downyards[(g, p, m)]]).astype(np.float32)
            for (g, p, m) in base.play_frames
        }
    play_vec = build_play_vectors(arm)  # context or situation, keyed (gameId, playId)
    return {(g, p, m): play_vec[(g, p)] for (g, p, m) in base.play_frames}


# --------------------------------------------------------------------------------------
# Data prep + train/eval one seed (model + recipe reuse GameStateLitModel)
# --------------------------------------------------------------------------------------
def prepare_data(arm: str):
    wl = CONFIG["window"]
    train_base = load_datasets("hybrid_ts", split="train", window_length=wl)
    val_base = load_datasets("hybrid_ts", split="val", window_length=wl)
    test_base = load_datasets("hybrid_ts", split="test", window_length=wl)
    if arm == "off":
        return train_base, val_base, test_base, test_base, 0
    train_ds = GameStateDataset(train_base, build_lookup(train_base, arm, "train"))
    val_ds = GameStateDataset(val_base, build_lookup(val_base, arm, "val"))
    eval_ds = GameStateDataset(test_base, build_lookup(test_base, arm, "test"))
    return train_ds, val_ds, eval_ds, test_base, ARM_DIM[arm]


def train_one_seed(arm, seed, device, num_workers, batch_size, patience, max_epochs, data):
    train_ds, val_ds, eval_ds, test_base, gsd = data
    seed_everything(seed, workers=True)
    model_arm = "off" if arm == "off" else "on"

    model = GameStateLitModel(
        model_arm, CONFIG["model_dim"], CONFIG["num_layers"], CONFIG["window"],
        RAW_FEATURE_COUNT, gsd, learning_rate=1e-4, dropout=0.3,
    )

    version = f"{arm}_S{seed}"
    logger = TensorBoardLogger(save_dir=MODELS_DIR, name="presnap_context", version=version, log_graph=False)

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
    ).write_parquet(best_ckpt.with_suffix(".ctx_test_preds.parquet"))

    print(f"\n[DONE] presnap_context arm={arm} seed={seed} | val_loss={val_loss:.4f} "
          f"test_ADE={test_ade:.4f} yd over {n_eval} frames | game_state_dim={gsd} params={model.num_params:,}")
    return record


def aggregate():
    on_rows = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    off_rows = [json.loads(p.read_text()) for p in sorted(BASELINE_DIR.glob("off_S*.json"))]  # shared baseline
    if not on_rows and not off_rows:
        print("No per-run JSONs found in", RESULTS_DIR, "or", BASELINE_DIR)
        return
    import statistics as st

    if on_rows:  # CSV holds this experiment's own on-arms; off lives in the shared baseline dir
        pl.DataFrame(on_rows).drop("best_ckpt").write_csv("results/presnap_context_experiment.csv")
    rows = on_rows + off_rows  # on-arms + shared off, for the paired-delta summary below
    by_arm = {}
    for arm in ARMS:
        a = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["seed"])
        by_arm[arm] = {r["seed"]: r["test_ade"] for r in a}
        if a:
            ades = [r["test_ade"] for r in a]
            print(f"  {arm:10s} seeds={[r['seed'] for r in a]} mean={st.mean(ades):.4f} "
                  f"std={st.pstdev(ades) if len(ades) > 1 else 0:.4f}")
    off = by_arm.get("off", {})
    for arm in [a for a in ARMS if a != "off"]:
        seeds = sorted(set(off) & set(by_arm.get(arm, {})))
        if seeds:
            deltas = [by_arm[arm][s] - off[s] for s in seeds]
            print(f"  paired ({arm}-off) seeds {seeds}: mean delta = {st.mean(deltas):+.4f}  (negative = helps)")
    print("\nWrote results/presnap_context_experiment.csv")


if __name__ == "__main__":
    parser = ArgumentParser(description="Experiment 2: pre-snap play context (late-fusion) on hybrid_ts")
    parser.add_argument("--arm", choices=ARMS, help="off / context / situation")
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
            print(f"\n===== presnap_context arm={args.arm} seed={seed} (device {args.device}) =====")
            train_one_seed(
                args.arm, seed, args.device,
                args.num_workers, args.batch_size, args.patience, args.max_epochs, data,
            )
