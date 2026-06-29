"""
Phase 1: game-state (play-context) late-fusion experiment -- fully self-contained.

Question: does giving the model play-level context it currently cannot see --
`down`, `yardsToGo`, `distanceToGoal` -- improve tackle-location prediction, on the
FULL data and FULL recipe (not a subsample)?

Design (no corners cut, nothing in the existing pipeline is modified):
  * Data:    reuses the EXISTING temporal dataset pickle (data/datasets/temporal/),
             i.e. the exact full-data windows behind the 4.09 result. hybrid_ts loads
             it at window 15; the single-frame transformer control loads it at window 1
             (raw 8 features, one frame). No re-prep, no rebuild, no subsample.
  * Game state: read straight from the prepped parquets (same source), verified
             constant within a play. Appended to each player's vector as 3 broadcast
             columns -> (..., 22, 8+3). `playResult` (the play OUTCOME) is never used.
  * Models:  subclass HybridTS / SportsTransformer. Reuse their encoder/pool; add a
             tiny game-state branch and a wider decoder; the forward splits off the 3
             columns and LATE-fuses them after player pooling (play-level info never
             goes through the per-player GRU/attention, and is never averaged away).
  * Recipe:  GameStateLitModel subclasses LitModel and overrides ONLY model construction,
             so training_step / validation_step / configure_optimizers (AdamW, SmoothL1)
             are inherited unchanged. The Trainer block mirrors train.py.train_model.

Fairness: BOTH arms run through THIS harness. The "off" arm is the plain parent model
on the unwrapped dataset, so off vs on differ ONLY by the 3 features -- and the off arm
is provably the original architecture (sanity gate below).

Sanity gate: `--model hybrid_ts --arm off` reproduces the plain hybrid_ts baseline.

Usage (one run = one GPU; launch the 2x2x{seeds} matrix in parallel):
    uv run python src/run_gamestate_experiment.py --model hybrid_ts   --arm off --seed 0 --device 0
    uv run python src/run_gamestate_experiment.py --model hybrid_ts   --arm on  --seed 0 --device 1
    uv run python src/run_gamestate_experiment.py --model transformer --arm off --seed 0 --device 2
    uv run python src/run_gamestate_experiment.py --model transformer --arm on  --seed 0 --device 3
    ...repeat for --seed 1..7
    uv run python src/run_gamestate_experiment.py --aggregate         # collate per-run JSONs -> CSV
"""

from __future__ import annotations

import json
import re
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

from datasets import PREPPED_DATA_DIR, RAW_FEATURE_COUNT, BDB2024_Dataset, load_datasets
from models import HybridTS, LitModel, SportsTransformer

# Play-level context the models currently never see. NOTE: `playResult` is the play
# outcome and is DELIBERATELY excluded (it would leak the target).
GAME_STATE_FEATURES = ["down", "yardsToGo", "distanceToGoal"]
GAME_STATE_DIM = len(GAME_STATE_FEATURES)

# Best established config per architecture (fixed -- this is a clean delta study, not a sweep).
CONFIGS = {
    "hybrid_ts": dict(model_dim=128, num_layers=4, window=15),  # the 4.09 winner
    "transformer": dict(model_dim=128, num_layers=2, window=1),  # single-frame raw-8 control
}

RESULTS_DIR = Path("results/gamestate")
MODELS_DIR = Path("models")


# --------------------------------------------------------------------------------------
# Game-state lookup + dataset wrapper (reads the real prepped data; wraps the real dataset)
# --------------------------------------------------------------------------------------
def build_gamestate_lookup(split: str) -> dict[tuple, np.ndarray]:
    """Map (gameId, playId, mirrored) -> float32[down, yardsToGo, distanceToGoal].

    Read from the same prepped parquet the temporal dataset was built from. Asserts the
    values are constant within a play, so the broadcast below is exact.
    """
    df = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_features.parquet")
    sub = df.select(["gameId", "playId", "mirrored", *GAME_STATE_FEATURES]).unique()
    per_play = sub.group_by(["gameId", "playId", "mirrored"]).len()
    assert per_play["len"].max() == 1, f"game-state not constant within a play in split={split}"
    lut: dict[tuple, np.ndarray] = {}
    for row in sub.iter_rows(named=True):
        key = (row["gameId"], row["playId"], row["mirrored"])
        lut[key] = np.array([row[c] for c in GAME_STATE_FEATURES], dtype=np.float32)
    return lut


class GameStateDataset(Dataset):
    """Wrap a loaded BDB2024_Dataset and append the 3 broadcast game-state columns.

    Returns (features, target) with features shaped (..., 22, F+3) -- a (22, F+3) single
    frame or a (T, 22, F+3) window, matching whatever the base dataset serves. The base
    dataset (full data, real windowing) is untouched.
    """

    def __init__(self, base: BDB2024_Dataset, lut: dict[tuple, np.ndarray]):
        self.base = base
        self.lut = lut
        self.keys = base.keys  # proxy so callers can read keys uniformly

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
        out = np.concatenate([feat, gs_b], axis=-1).astype(np.float32)
        return out, tgt


# --------------------------------------------------------------------------------------
# Late-fusion models: reuse the parent encoder/pool, add a game-state branch + wider head
# --------------------------------------------------------------------------------------
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
    # BatchNorm handles the very different scales (down 1-4 vs distanceToGoal 1-99).
    return nn.Sequential(
        nn.BatchNorm1d(game_state_dim),
        nn.Linear(game_state_dim, gs_hidden),
        nn.ReLU(),
    )


class GameStateHybridTS(HybridTS):
    """hybrid_ts (time->space) with late-fused play-level game state.

    Input: [B, T, 22, F+G]. The first F channels go through the parent's BatchNorm/GRU/
    attention/pool unchanged; the last G channels (constant across players & time) are
    pulled once and joined to the pooled embedding before a wider decoder.
    """

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


class GameStateTransformer(SportsTransformer):
    """Single-frame SportsTransformer with late-fused play-level game state.

    Input: [B, 22, F+G]. Mirrors the parent forward, splitting off the G game-state
    channels and joining them to the pooled embedding before a wider decoder.
    """

    def __init__(self, feature_len, model_dim, num_layers, dropout, game_state_dim):
        super().__init__(feature_len, model_dim, num_layers, dropout)
        self.player_feature_len = feature_len
        self.game_state_dim = game_state_dim
        gs_hidden = max(1, model_dim // 4)
        self.gs_branch = _gs_branch(game_state_dim, gs_hidden)
        self.decoder = _wide_decoder(model_dim + gs_hidden, model_dim, dropout)
        self.hyperparams = {**self.hyperparams, "game_state_dim": game_state_dim, "fusion": "late"}

    def forward(self, x: Tensor) -> Tensor:
        f = self.player_feature_len
        xp = x[..., :f]  # [B,P,F]
        gs = x[:, 0, f:]  # [B,G] -- any player (constant)
        xp = self.feature_norm_layer(xp.permute(0, 2, 1)).permute(0, 2, 1)
        xp = self.feature_embedding_layer(xp)
        xp = self.transformer_encoder(xp)
        h = torch.squeeze(self.player_pooling_layer(xp.permute(0, 2, 1)), -1)  # [B,M]
        h = torch.cat([h, self.gs_branch(gs)], dim=-1)  # late fusion
        return self.decoder(h)


def build_model(model_name, arm, model_dim, num_layers, window, feature_len, game_state_dim, dropout):
    """off -> the plain parent architecture (provably original). on -> the late-fusion variant."""
    if model_name == "hybrid_ts":
        if arm == "on":
            return GameStateHybridTS(feature_len, model_dim, num_layers, dropout, window, game_state_dim)
        return HybridTS(feature_len, model_dim, num_layers, dropout, window)
    if arm == "on":
        return GameStateTransformer(feature_len, model_dim, num_layers, dropout, game_state_dim)
    return SportsTransformer(feature_len, model_dim, num_layers, dropout)


# --------------------------------------------------------------------------------------
# Lightning wrapper: reuse LitModel's training recipe verbatim, swap only the model
# --------------------------------------------------------------------------------------
class GameStateLitModel(LitModel):
    """Inherits LitModel's training/validation/test/predict steps and AdamW optimizer
    unchanged (identical recipe); only model construction differs."""

    def __init__(
        self,
        model_name: str,
        arm: str,
        model_dim: int,
        num_layers: int,
        window: int,
        feature_len: int,
        game_state_dim: int,
        learning_rate: float = 1e-4,
        dropout: float = 0.3,
    ):
        # Skip LitModel.__init__ (its model_type dispatch) but keep LightningModule setup.
        LightningModule.__init__(self)
        self.save_hyperparameters()
        self.model = build_model(
            model_name, arm, model_dim, num_layers, window, feature_len, game_state_dim, dropout
        )
        self.model_type = f"{model_name}_{arm}"  # for logging only; steps don't dispatch on it
        self.learning_rate = learning_rate
        self.loss_fn = nn.SmoothL1Loss()
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# Train + evaluate one (model, arm, seed)
# --------------------------------------------------------------------------------------
def _val_loss_from_ckpt(ckpt_path: Path) -> float:
    m = re.search(r"val_loss=([\d.]+)", Path(ckpt_path).name)
    return float(m.group(1).rstrip(".")) if m else float("nan")


def evaluate_test_ade(model, best_ckpt, eval_ds, base, device, num_workers):
    """Test ADE in yards over NON-mirrored frames only (project convention). ADE is
    computed in anchor-relative space, which equals absolute ADE (the anchor cancels)."""
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


def prepare_data(model_name, arm):
    """Load the FULL datasets once (shared across all seeds in this process).

    Returns (cfg, train_ds, val_ds, eval_ds, test_base, game_state_dim). The datasets are
    read-only and seed-independent, so loading them once and reusing across seeds is exact
    (only the per-seed dataloaders/model are rebuilt). Avoids reloading the pickle 5x.
    """
    cfg = CONFIGS[model_name]
    wl = cfg["window"]  # hybrid_ts -> 15, transformer -> 1 (single frame)
    # Full data: the existing temporal pickle behind 4.09. transformer just reads it at W1.
    train_base = load_datasets("hybrid_ts", split="train", window_length=wl)
    val_base = load_datasets("hybrid_ts", split="val", window_length=wl)
    test_base = load_datasets("hybrid_ts", split="test", window_length=wl)

    if arm == "on":
        luts = {s: build_gamestate_lookup(s) for s in ("train", "val", "test")}
        train_ds = GameStateDataset(train_base, luts["train"])
        val_ds = GameStateDataset(val_base, luts["val"])
        eval_ds = GameStateDataset(test_base, luts["test"])
        gsd = GAME_STATE_DIM
    else:
        train_ds, val_ds, eval_ds = train_base, val_base, test_base
        gsd = 0
    return cfg, train_ds, val_ds, eval_ds, test_base, gsd


def train_one_seed(model_name, arm, seed, device, num_workers, batch_size, patience, max_epochs, data):
    """Train + evaluate a single seed using preloaded datasets. seed_everything is called
    here (before model init and dataloader creation) so each seed controls init + shuffle."""
    cfg, train_ds, val_ds, eval_ds, test_base, gsd = data
    seed_everything(seed, workers=True)
    model_dim, num_layers, window = cfg["model_dim"], cfg["num_layers"], cfg["window"]
    feature_len = RAW_FEATURE_COUNT  # 8 raw player features; game state is separate

    model = GameStateLitModel(
        model_name, arm, model_dim, num_layers, window, feature_len, gsd, learning_rate=1e-4, dropout=0.3
    )

    version = f"{arm}_S{seed}"
    logger = TensorBoardLogger(save_dir=MODELS_DIR, name=f"gamestate_{model_name}", version=version, log_graph=False)

    # drop_last: the game-state BatchNorm runs on the raw batch dim, so a size-1 final
    # batch would crash it (the player-level BatchNorms never see batch==1). Dropping the
    # last partial batch is applied to BOTH arms, so the on-off comparison is unaffected.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True, num_workers=num_workers)

    # Trainer block mirrors train.py.train_model (same recipe).
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
    test_ade, n_eval, preds, keys, targets = evaluate_test_ade(
        model, best_ckpt, eval_ds, test_base, device, num_workers
    )

    # Persist: a per-run JSON (parallel-safe -- no shared file) + a predictions parquet.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(
        model=model_name,
        arm=arm,
        seed=seed,
        model_dim=model_dim,
        num_layers=num_layers,
        window=window,
        game_state_features=GAME_STATE_FEATURES if arm == "on" else [],
        params=model.num_params,
        val_loss=val_loss,
        test_ade=round(test_ade, 4),
        n_test_nonmirrored=n_eval,
        best_ckpt=str(best_ckpt),
    )
    out_json = RESULTS_DIR / f"{model_name}_{arm}_S{seed}.json"
    out_json.write_text(json.dumps(record, indent=2))

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
    ).write_parquet(best_ckpt.with_suffix(".gs_test_preds.parquet"))

    print(f"\n[DONE] {model_name} arm={arm} seed={seed} | val_loss={val_loss:.4f} "
          f"test_ADE={test_ade:.4f} yd over {n_eval} frames | params={model.num_params:,}")
    print(f"       -> {out_json}")
    return record


def aggregate():
    """Collate per-run JSONs into results/gamestate_experiment.csv with mean+/-std per arm."""
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    if not rows:
        print("No per-run JSONs found in", RESULTS_DIR)
        return
    df = pl.DataFrame([{k: v for k, v in r.items() if k != "game_state_features"} for r in rows])
    out = Path("results/gamestate_experiment.csv")
    df.write_csv(out)
    summary = (
        df.group_by(["model", "arm"])
        .agg(
            pl.col("test_ade").mean().round(4).alias("test_ade_mean"),
            pl.col("test_ade").std().round(4).alias("test_ade_std"),
            pl.col("val_loss").mean().round(4).alias("val_loss_mean"),
            pl.len().alias("n_seeds"),
        )
        .sort(["model", "arm"])
    )
    print(summary)
    print(f"\nWrote {out} ({len(rows)} runs)")


if __name__ == "__main__":
    parser = ArgumentParser(description="Phase 1: game-state late-fusion experiment")
    parser.add_argument("--model", choices=["hybrid_ts", "transformer"], help="architecture")
    parser.add_argument("--arm", choices=["off", "on"], help="off=baseline, on=+game state")
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
        assert args.model and args.arm, "provide --model and --arm (or --aggregate)"
        # Load the full data ONCE, then sweep seeds in-process (no external bash loop).
        data = prepare_data(args.model, args.arm)
        for seed in args.seeds:
            print(f"\n===== {args.model} arm={args.arm} seed={seed} (device {args.device}) =====")
            train_one_seed(
                args.model,
                args.arm,
                seed,
                args.device,
                args.num_workers,
                args.batch_size,
                args.patience,
                args.max_epochs,
                data,
            )
