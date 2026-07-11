"""
team-identity (off/def) late-fusion experiment

    - Data: reuse existing temporal dataset pickel
    - Teams: possessionTeam (off) and defensiveTeam (def)
    - Model: stgnn_ts with hybrid_knn_hub edges

16 dimension, learned embedding
using latest best model/topology
"""

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
from models import HybridTS, LitModel, SportsTransformer, STGNN_TS
from models import build_adjacency_torch, edge_features_torch

TEAM_EMBED_DIM = 16  # per side (offense / defense)
PLAYS_CSV = Path("data/bdb_2024/plays.csv")
RESULTS_DIR = Path("results/team")
MODELS_DIR = Path("models")


# Model registry: test with different model/configuration presets
MODEL_REGISTRY: dict[str, dict] = {
    "stgnn_ts_knn_hybrid_hub": dict(
        model_type="stgnn_ts",
        model_dim=128,
        num_layers=8,
        window_length=10,
        topology="knn_hybrid_hub",
        edge_features=True,
        knn_k=4,
    ),
    "hybrid_ts": dict(
        model_type="hybrid_ts",
        model_dim=128,
        num_layers=4,
        window_length=15,
    ),
    "transformer": dict(
        model_type="transformer",
        model_dim=128,
        num_layers=2,
        window_length=1,
    ),
}
DEFAULT_MODEL = "stgnn_ts_knn_hybrid_hub"


# Team lookup (read straight from plays.csv; possessionTeam/defensiveTeam are dropped
# from the prepped parquets after the join, so they're not available there) + dataset wrap
def build_team_vocab() -> dict[str, int]:
    """
    Derive the team vocabulary from whatever abbreviations  appear in
    plays.csv's possessionTeam/defensiveTeam columns
    """
    plays = pl.read_csv(PLAYS_CSV, null_values=["NA", "nan", "N/A", "NaN", ""])
    teams = sorted(
        set(plays["possessionTeam"].unique().to_list()) | set(plays["defensiveTeam"].unique().to_list())
    )
    return {team: idx for idx, team in enumerate(teams)}


def build_team_lookup(split: str, team_to_idx: dict[str, int]) -> dict[tuple, np.ndarray]:
    """
    Map (gameId, playId, mirrored) -> int64[offense_idx, defense_idx].
    Reads possessionTeam/defensiveTeam from plays.csv and joins onto the (gameId, playId,
    mirrored) keys in this split's prepped parquet
    """
    plays = pl.read_csv(PLAYS_CSV, null_values=["NA", "nan", "N/A", "NaN", ""])
    feat = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_features.parquet")
    keys = feat.select(["gameId", "playId", "mirrored"]).unique()

    joined = keys.join(
        plays.select(["gameId", "playId", "possessionTeam", "defensiveTeam"]).unique(),
        on=["gameId", "playId"],
        how="left",
    )
    missing = joined.filter(pl.col("possessionTeam").is_null() | pl.col("defensiveTeam").is_null())
    assert missing.height == 0, f"{missing.height} plays in split={split} missing team info"

    lut: dict[tuple, np.ndarray] = {}
    for row in joined.iter_rows(named=True):
        key = (row["gameId"], row["playId"], row["mirrored"])
        lut[key] = np.array(
            [team_to_idx[row["possessionTeam"]], team_to_idx[row["defensiveTeam"]]], dtype=np.int64
        )
    return lut


class TeamDataset(Dataset):
    """
    Wrap a loaded BDB2024_Dataset and attach a [offense_idx, defense_idx] pair per item.
    Returns (features, team_indices, target).
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
        team_idx = self.lut[(g, p, m)]  # int64[2]: [offense_idx, defense_idx]
        return feat, team_idx, tgt


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


def _stgnn_ts_pool_embedding(self: STGNN_TS, x: Tensor) -> Tensor:
    """_pool_embedding shim for STGNN_TS (mirrors models.STGNN_TS.forward() up to the decoder)."""

    B, T, P, F = x.size()
    last = x[:, -1, :, :]
    pos, vel, side, bc = last[..., 0:2], last[..., 2:4], last[..., 6], last[..., 7]

    x = self.feature_norm_layer(x.reshape(-1, F)).reshape(B, T, P, F)
    x = x.permute(0, 2, 1, 3).reshape(B * P, T, F)
    _, h_n = self.gru(x)
    x = h_n[-1].reshape(B, P, -1)

    adj = build_adjacency_torch(side, bc, pos, self.knn_k, self.topology)
    edge_feats = edge_features_torch(pos, vel) if self.use_edge_features else None
    for layer in self.gat_layers:
        x = layer(x, adj, edge_feats)

    return torch.squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)


def _hybrid_ts_pool_embedding(self: HybridTS, x: Tensor) -> Tensor:
    """_pool_embedding shim for HybridTS (mirrors its forward() up to the decoder)."""
    B, T, P, F = x.size()
    x = self.feature_norm_layer(x.reshape(-1, F)).reshape(B, T, P, F)
    x = x.permute(0, 2, 1, 3).reshape(B * P, T, F)
    _, h_n = self.gru(x)
    x = h_n[-1].reshape(B, P, -1)
    x = self.transformer_encoder(x)
    return torch.squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)


def _transformer_pool_embedding(self: SportsTransformer, x: Tensor) -> Tensor:
    """_pool_embedding shim for SportsTransformer (mirrors its forward() up to the decoder)."""
    B, P, F = x.size()
    x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
    x = self.feature_embedding_layer(x)
    x = self.transformer_encoder(x)
    return torch.squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)


# Attach the shims as bound methods (process-local monkeypatch, not a change to models.py
# on disk) so any STGNN_TS/HybridTS/SportsTransformer instance built via LitModel can be
# late-fused the same way.
STGNN_TS._pool_embedding = _stgnn_ts_pool_embedding
HybridTS._pool_embedding = _hybrid_ts_pool_embedding
SportsTransformer._pool_embedding = _transformer_pool_embedding


class TeamAwareModel(nn.Module):
    """Generic late-fusion wrapper: any base model with `_pool_embedding(x) -> [B, M]` gets
    two small team-embedding tables and a decoder widened to accept them.

    forward(x, team_idx) where team_idx: [B, 2] int64 = [offense_idx, defense_idx].
    """

    def __init__(self, base_model: nn.Module, model_dim: int, dropout: float, num_teams: int,
                 team_embed_dim: int = TEAM_EMBED_DIM):
        super().__init__()
        assert hasattr(base_model, "_pool_embedding"), f"{type(base_model).__name__} has no _pool_embedding"
        self.base = base_model
        self.offense_embedding = nn.Embedding(num_teams, team_embed_dim)
        self.defense_embedding = nn.Embedding(num_teams, team_embed_dim)
        self.decoder = _wide_decoder(model_dim + 2 * team_embed_dim, model_dim, dropout)
        self.hyperparams = {
            **getattr(base_model, "hyperparams", {}),
            "num_teams": num_teams,
            "team_embed_dim": team_embed_dim,
            "fusion": "late_team",
        }
        self.window_length = getattr(base_model, "window_length", 1)

    def forward(self, x: Tensor, team_idx: Tensor) -> Tensor:
        h = self.base._pool_embedding(x)  # [B, model_dim]
        off_emb = self.offense_embedding(team_idx[:, 0])
        def_emb = self.defense_embedding(team_idx[:, 1])
        h = torch.cat([h, off_emb, def_emb], dim=-1)
        return self.decoder(h)


def build_model(model_name: str, arm: str, feature_len: int, num_teams: int,
                 batch_size: int = 256, dropout: float = 0.3) -> nn.Module:
    """
    off -> the plain registered architecture, built via LitModel's own dispatch.
    on -> that same base model wrapped with team-identity late fusion.
    """
    spec = MODEL_REGISTRY[model_name]
    lit = LitModel(batch_size=batch_size, feature_len=feature_len, dropout=dropout, **spec)
    base = lit.model
    if arm == "off":
        return base
    return TeamAwareModel(base, model_dim=spec["model_dim"], dropout=dropout, num_teams=num_teams)


# --------------------------------------------------------------------------------------
# Lightning wrapper: reuse LitModel's training recipe, override only the steps that need
# to unpack the extra team-index tensor from the batch.
# --------------------------------------------------------------------------------------
class TeamLitModel(LitModel):
    """Inherits LitModel's AdamW optimizer unchanged; only model construction and the
    train/val/predict steps differ (the "off" arm batches are still (x, y) pairs, so the
    steps branch on self.arm to unpack the right shape)."""

    def __init__(
        self,
        model_name: str,
        arm: str,
        feature_len: int,
        num_teams: int,
        learning_rate: float = 1e-4,
        dropout: float = 0.3,
    ):
        # Skip LitModel.__init__ (its model_type dispatch) but keep LightningModule setup.
        LightningModule.__init__(self)
        self.save_hyperparameters()
        self.arm = arm
        self.model = build_model(model_name, arm, feature_len, num_teams, batch_size=256, dropout=dropout)
        self.model_type = f"{model_name}_{arm}"  # for logging only
        self.learning_rate = learning_rate
        self.loss_fn = nn.SmoothL1Loss()
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _forward_batch(self, batch):
        if self.arm == "on":
            x, team_idx, y = batch
            return self.model(x, team_idx), y
        x, y = batch
        return self.model(x), y

    def training_step(self, batch, batch_idx):
        y_hat, y = self._forward_batch(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self._forward_batch(batch)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat, _y = self._forward_batch(batch)
        return y_hat


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


def prepare_data(model_name: str, arm: str):
    """Load the FULL datasets once (shared across all seeds in this process).

    Returns (window, train_ds, val_ds, eval_ds, test_base, team_to_idx). The datasets are
    read-only and seed-independent, so loading them once and reusing across seeds is exact
    (only the per-seed dataloaders/model are rebuilt). Avoids reloading the pickle 5x and
    rebuilding the team lookup 5x.
    """
    window = MODEL_REGISTRY[model_name]["window_length"]
    # Full data: the existing temporal pickle. hybrid_ts/stgnn_ts serve it at window 10;
    # transformer reads the same pickle at the window (single frame). load_datasets dispatches
    # on whether model_type is in TEMPORAL_MODEL_TYPES, so "hybrid_ts" here just selects the
    # shared "temporal" pickle -- it does not build a hybrid_ts-specific dataset.
    train_base = load_datasets("hybrid_ts", split="train", window_length=window)
    val_base = load_datasets("hybrid_ts", split="val", window_length=window)
    test_base = load_datasets("hybrid_ts", split="test", window_length=window)

    team_to_idx = None
    if arm == "on":
        team_to_idx = build_team_vocab()

        # save vocab for reproducibility
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        vocab_path = RESULTS_DIR / "team_vocab.json"
        if not vocab_path.exists():
            vocab_path.write_text(json.dumps(team_to_idx, indent=2, sort_keys=True))

        luts = {s: build_team_lookup(s, team_to_idx) for s in ("train", "val", "test")}
        train_ds = TeamDataset(train_base, luts["train"])
        val_ds = TeamDataset(val_base, luts["val"])
        eval_ds = TeamDataset(test_base, luts["test"])
    else:
        train_ds, val_ds, eval_ds = train_base, val_base, test_base
    return window, train_ds, val_ds, eval_ds, test_base, team_to_idx


def train_one_seed(model_name, arm, seed, device, num_workers, batch_size, patience, max_epochs, data):
    """Train + evaluate a single seed using preloaded datasets. seed_everything is called
    here (before model init and dataloader creation) so each seed controls init + shuffle."""
    window, train_ds, val_ds, eval_ds, test_base, team_to_idx = data
    seed_everything(seed, workers=True)
    feature_len = RAW_FEATURE_COUNT  # 8 raw player features; team identity is separate
    num_teams = len(team_to_idx) if team_to_idx else 0

    model = TeamLitModel(model_name, arm, feature_len, num_teams, learning_rate=1e-4, dropout=0.3)

    version = f"{arm}_S{seed}"
    logger = TensorBoardLogger(save_dir=MODELS_DIR, name=f"team_{model_name}", version=version, log_graph=False)

    # drop_last: matches phase-1 gamestate convention (a size-1 final batch could crash a
    # batch-dim BatchNorm somewhere downstream); applied to BOTH arms so the comparison is
    # unaffected.
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
        window=window,
        team_embed_dim=TEAM_EMBED_DIM if arm == "on" else 0,
        num_teams=num_teams,
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
    ).write_parquet(best_ckpt.with_suffix(".team_test_preds.parquet"))

    print(f"\n[DONE] {model_name} arm={arm} seed={seed} | val_loss={val_loss:.4f} "
          f"test_ADE={test_ade:.4f} yd over {n_eval} frames | params={model.num_params:,}")
    print(f"       -> {out_json}")
    return record


def aggregate():
    """Collate per-run JSONs into results/team/team_experiment.csv with mean+/-std per arm
    and paired ON-OFF ADE differences."""
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    if not rows:
        print("No per-run JSONs found in", RESULTS_DIR)
        return
    
    df = pl.DataFrame(rows)
    out = RESULTS_DIR / "team_experiment.csv"
    df.write_csv(out)

    summary = (
        df.group_by(["model", "arm"])
        .agg(
            pl.col("test_ade").mean().round(4).alias("test_ade_mean"),
            pl.col("test_ade").std().round(4).alias("test_ade_std"),
            pl.col("val_loss").mean().round(4).alias("val_loss_mean"),
            pl.col("params").mean().cast(pl.Int64).alias("params"),
            pl.len().alias("n_seeds"),
        )
        .sort(["model", "arm"])
    )
    print(summary)

    # Paired comparison: ON - OFF for matching seeds
    for model in sorted(df["model"].drop_nulls().unique()):
        off = (
            df.filter((pl.col("model") == model) & (pl.col("arm") == "off"))
            .select(["seed", "test_ade"])
            .rename({"test_ade": "off_ade"})
        )
        on = (
            df.filter((pl.col("model") == model) & (pl.col("arm") == "on"))
            .select(["seed", "test_ade"])
            .rename({"test_ade": "on_ade"})
        )
        paired = (
            off.join(on, on="seed", how="inner")
            .with_columns((pl.col("on_ade") - pl.col("off_ade")).alias("delta"))
            .sort("seed")
        )
        if paired.height:
            print(f"\n{model} paired (ON - OFF):")
            print(paired.select(["seed", "delta"]))
            print(
                f"Mean paired delta: {paired['delta'].mean():+.4f} "
                "(negative = team identity improves ADE)"
            )
    print(f"\nWrote {out} ({len(rows)} runs)")

if __name__ == "__main__":
    parser = ArgumentParser(description="offense/defense team-identity late-fusion experiment")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default=DEFAULT_MODEL, help="architecture")
    parser.add_argument("--arm", choices=["off", "on"], help="off=baseline, on=+team identity")
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