"""
Late-fusion experiment testing whether play-level context — offensive formation
(offenseFormation) and defenders in the box (defendersInTheBox) — improves player
trajectory prediction (ADE) on top of the STGNN-TS / HybridTS / Transformer base models.

Two arms are trained per model, each across multiple seeds:
  - "off": baseline model, no play context.
  - "on":  baseline model wrapped in PlayContextFusion, which late-fuses a one-hot
           formation embedding and a small MLP over z-scored defendersInTheBox into
           the pooled player embedding before the decoder.

Pipeline:
  1. prepare_data()   - builds datasets/lookups (formation vocab, ditb stats) per arm.
  2. train_one_seed()  - trains one (model, arm, seed) combination, evaluates test ADE,
                          and writes a per-run JSON + predictions parquet to RESULTS_DIR.
  3. aggregate()        - collects all per-run JSONs into a summary CSV, reporting
                          mean/std test ADE per (model, arm) and paired on-vs-off deltas
                          by seed.
"""

from __future__ import annotations
 
import json
import re
from argparse import ArgumentParser
from pathlib import Path
 
import torch.nn.functional as F
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

 
PLAYS_CSV = Path("data/bdb_2024/plays.csv")
RESULTS_DIR = Path("results/play_context")
MODELS_DIR = Path("models")

CATEGORICAL_CONTEXT = ["offenseFormation"]
    # Formations: ['EMPTY', 'I_FORM', 'JUMBO', 'PISTOL', 'SHOTGUN', 'SINGLEBACK', 'WILDCAT']
CONTINUOUS_CONTEXT = ["defendersInTheBox"]
    # 1,2, ... , 11

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
def build_formation_vocab() -> dict[str, int]:
    """Derive formation vocabulary from plays.csv. Unknown/null -> index 0 ('UNK')."""
    plays = pl.read_csv(PLAYS_CSV, null_values=["NA", "nan", "N/A", "NaN", ""])
    formations = sorted(
        f for f in plays["offenseFormation"].unique().to_list() if f is not None
    )
    # index 0 = UNK (handles nulls or unseen formations at inference time)
    vocab = {"UNK": 0}
    vocab.update({f: i + 1 for i, f in enumerate(formations)})
    return vocab
 
 
def compute_ditb_stats(split: str = "train") -> tuple[float, float]:
    """Compute mean and std of defendersInTheBox from the training split for z-scoring."""
    plays = pl.read_csv(PLAYS_CSV, null_values=["NA", "nan", "N/A", "NaN", ""])
    feat = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_features.parquet")
    train_play_ids = feat.select(["gameId", "playId"]).unique()
    ditb = (
        plays.join(train_play_ids, on=["gameId", "playId"], how="inner")["defendersInTheBox"]
        .drop_nulls()
    )
    return float(ditb.mean()), float(ditb.std())
 
 
def build_play_context_lookup(
    split: str,
    formation_vocab: dict[str, int],
    ditb_mean: float,
    ditb_std: float,
) -> dict[tuple, np.ndarray]:
    """Map (gameId, playId, mirrored) -> float32[formation_idx, ditb_z].
 
    formation_idx is an int stored as float32 (cast to int64 in the model for embedding lookup).
    ditb_z is the z-scored defendersInTheBox value. Nulls are filled with 0.0 (mean).
    Mirroring does not affect formation or ditb, so both sides share the same lookup.
    """
    plays = pl.read_csv(PLAYS_CSV, null_values=["NA", "nan", "N/A", "NaN", ""])
    feat = pl.read_parquet(PREPPED_DATA_DIR / f"{split}_features.parquet")
    keys = feat.select(["gameId", "playId", "mirrored"]).unique()
 
    joined = keys.join(
        plays.select(["gameId", "playId", "offenseFormation", "defendersInTheBox"]).unique(),
        on=["gameId", "playId"],
        how="left",
    )
 
    lut: dict[tuple, np.ndarray] = {}
    for row in joined.iter_rows(named=True):
        key = (row["gameId"], row["playId"], row["mirrored"])
        formation_idx = formation_vocab.get(row["offenseFormation"] or "UNK", 0)
        ditb = row["defendersInTheBox"]
        ditb_z = (ditb - ditb_mean) / (ditb_std + 1e-8) if ditb is not None else 0.0
        lut[key] = np.array([float(formation_idx), float(ditb_z)], dtype=np.float32)
    return lut
 
 
class PlayContextDataset(Dataset):
    """Wrap BDB2024_Dataset and attach [formation_idx_as_float, ditb_z] per item.
 
    formation_idx is stored as float32 here so it can travel through the DataLoader
    with the rest of the batch as a single float tensor; it is cast to int64 inside
    PlayContextFusion before the embedding lookup.
    Returns (features, context, target).
    """
 
    def __init__(self, base: BDB2024_Dataset, lut: dict[tuple, np.ndarray]):
        self.base = base
        self.lut = lut
        self.keys = base.keys
 
    def __len__(self) -> int:
        return len(self.base)
 
    def __getitem__(self, idx: int):
        feat, tgt = self.base[idx]
        g, p, m, _ = self.base.keys[idx]
        ctx = self.lut[(g, p, m)]  # float32[2]: [formation_idx_as_float, ditb_z]
        return feat, ctx, tgt
 
 
# --------------------------------------------------------------------------------------
# Late-fusion module: formation embedding + ditb MLP -> concatenated to pooled embedding
# --------------------------------------------------------------------------------------
def _wide_decoder(in_dim: int, model_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, model_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(model_dim, model_dim // 4),
        nn.ReLU(),
        nn.LayerNorm(model_dim // 4),
        nn.Linear(model_dim // 4, 2),
    )
 
 
class PlayContextFusion(nn.Module):
    """Late-fuses formation (embedding) + defendersInTheBox (float branch) to the player
    pool embedding before the decoder. Can be wrapped around any base model that exposes
    `_pool_embedding(x) -> [B, model_dim]`."""
 
    def __init__(self, base_model: nn.Module, model_dim: int, dropout: float, num_formations: int):
        super().__init__()
        assert hasattr(base_model, "_pool_embedding"), f"{type(base_model).__name__} has no _pool_embedding"
        self.base = base_model
        self.window_length = getattr(base_model, "window_length", 1)
 
        self.num_formations = num_formations
 
        # Defenders in the box: BatchNorm handles scale; small MLP maps to hidden dim
        ditb_hidden = max(4, model_dim // 16)
        self.ditb_branch = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Linear(1, ditb_hidden),
            nn.ReLU(),
        )
 
        context_dim = num_formations + ditb_hidden
        self.decoder = _wide_decoder(model_dim + context_dim, model_dim, dropout)
        self.hyperparams = {
            **getattr(base_model, "hyperparams", {}),
            "num_formations": num_formations,
            "formation_encoding": "one_hot",
            "ditb_hidden": ditb_hidden,
            "fusion": "late_play_context",
        }
 
    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        # ctx: [B, 2] float32 -- [formation_idx_as_float, ditb_z]
        formation_idx = ctx[:, 0].long()  # [B] int64 for embedding lookup
        ditb_z = ctx[:, 1:2]             # [B, 1] float for ditb branch
 
        h = self.base._pool_embedding(x)                        # [B, model_dim]
        # one-hot encoding concatenated directly to pooled embedding
        formation_onehot = F.one_hot(formation_idx, num_classes=self.num_formations,).float()
        ditb_emb = self.ditb_branch(ditb_z)
        h = torch.cat([h, formation_onehot, ditb_emb], dim=-1) # [B, ditb_hidden]
        return self.decoder(h)
 
 
# _pool_embedding shims (process-local monkeypatches, same pattern as team experiment)
def _stgnn_ts_pool_embedding(self: STGNN_TS, x: Tensor) -> Tensor:
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
    B, T, P, F = x.size()
    x = self.feature_norm_layer(x.reshape(-1, F)).reshape(B, T, P, F)
    x = x.permute(0, 2, 1, 3).reshape(B * P, T, F)
    _, h_n = self.gru(x)
    x = h_n[-1].reshape(B, P, -1)
    x = self.transformer_encoder(x)
    return torch.squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)
 
 
def _transformer_pool_embedding(self: SportsTransformer, x: Tensor) -> Tensor:
    B, P, F = x.size()
    x = self.feature_norm_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
    x = self.feature_embedding_layer(x)
    x = self.transformer_encoder(x)
    return torch.squeeze(self.player_pooling_layer(x.permute(0, 2, 1)), -1)
 
 
STGNN_TS._pool_embedding = _stgnn_ts_pool_embedding
HybridTS._pool_embedding = _hybrid_ts_pool_embedding
SportsTransformer._pool_embedding = _transformer_pool_embedding
 
 
def build_model(model_name: str, arm: str, feature_len: int, num_formations: int,
                batch_size: int = 256, dropout: float = 0.3) -> nn.Module:
    spec = MODEL_REGISTRY[model_name]
    lit = LitModel(batch_size=batch_size, feature_len=feature_len, dropout=dropout, **spec)
    base = lit.model
    if arm == "off":
        return base
    # arm == "on": wrap base model
    return PlayContextFusion(base, model_dim=spec["model_dim"], dropout=dropout,
                              num_formations=num_formations)
 
 
# Lightning wrapper
class PlayContextLitModel(LitModel):
    def __init__(self, model_name: str, arm: str, feature_len: int, num_formations: int,
                 learning_rate: float = 1e-4, dropout: float = 0.3):
        LightningModule.__init__(self)
        self.save_hyperparameters()
        self.arm = arm
        self.model = build_model(model_name, arm, feature_len, num_formations, dropout=dropout)
        self.model_type = f"{model_name}_{arm}"
        self.learning_rate = learning_rate
        self.loss_fn = nn.SmoothL1Loss()
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
 
    def _forward_batch(self, batch):
        if self.arm == "on":
            x, ctx, y = batch
            return self.model(x, ctx), y
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
 
 
# Train + evaluate
def _val_loss_from_ckpt(ckpt_path: Path) -> float:
    m = re.search(r"val_loss=([\d.]+)", Path(ckpt_path).name)
    return float(m.group(1).rstrip(".")) if m else float("nan")
 
 
def evaluate_test_ade(model, best_ckpt, eval_ds, base, device, num_workers):
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
    window = MODEL_REGISTRY[model_name]["window_length"]
    # always loads via "hybrid_ts" regardless of model_name
    train_base = load_datasets("hybrid_ts", split="train", window_length=window)
    val_base = load_datasets("hybrid_ts", split="val", window_length=window)
    test_base = load_datasets("hybrid_ts", split="test", window_length=window)
 
    formation_vocab = None
    ditb_mean, ditb_std = 0.0, 1.0
    if arm == "on":
        formation_vocab = build_formation_vocab()
        ditb_mean, ditb_std = compute_ditb_stats(split="train")
 
        # Save vocab and stats for reproducibility
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        vocab_path = RESULTS_DIR / "formation_vocab.json"
        if not vocab_path.exists():
            vocab_path.write_text(json.dumps(formation_vocab, indent=2, sort_keys=True))
        stats_path = RESULTS_DIR / "ditb_stats.json"
        if not stats_path.exists():
            stats_path.write_text(json.dumps({"mean": ditb_mean, "std": ditb_std}, indent=2))
 
        luts = {s: build_play_context_lookup(s, formation_vocab, ditb_mean, ditb_std)
                for s in ("train", "val", "test")}
        train_ds = PlayContextDataset(train_base, luts["train"])
        val_ds = PlayContextDataset(val_base, luts["val"])
        eval_ds = PlayContextDataset(test_base, luts["test"])
    else:
        train_ds, val_ds, eval_ds = train_base, val_base, test_base
 
    num_formations = len(formation_vocab) if formation_vocab else 0
    return window, train_ds, val_ds, eval_ds, test_base, num_formations
 
 
def train_one_seed(model_name, arm, seed, device, num_workers, batch_size, patience,
                   max_epochs, data):
    window, train_ds, val_ds, eval_ds, test_base, num_formations = data
    seed_everything(seed, workers=True)
    feature_len = RAW_FEATURE_COUNT
 
    model = PlayContextLitModel(
        model_name, arm, feature_len, num_formations, learning_rate=1e-4, dropout=0.3,
    )
 
    version = f"{arm}_S{seed}"
    logger = TensorBoardLogger(save_dir=MODELS_DIR, name=f"play_ctx_{model_name}",
                                version=version, log_graph=False)
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               pin_memory=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False,
                             pin_memory=True, num_workers=num_workers)
 
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[device],
        sync_batchnorm=True,
        logger=logger,
        enable_model_summary=True,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=patience),
            callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1,
                                       filename="{epoch}-{val_loss:.3f}"),
        ],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
 
    best_ckpt = Path(trainer.checkpoint_callback.best_model_path)
    val_loss = _val_loss_from_ckpt(best_ckpt)
    test_ade, n_eval, preds, keys, targets = evaluate_test_ade(
        model, best_ckpt, eval_ds, test_base, device, num_workers
    )
 
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(
        model=model_name,
        arm=arm,
        seed=seed,
        window=window,
        num_formations=num_formations,
        params=model.num_params,
        val_loss=val_loss,
        test_ade=round(test_ade, 4),
        n_test_nonmirrored=n_eval,
        best_ckpt=str(best_ckpt),
    )
    out_json = RESULTS_DIR / f"{model_name}_{arm}_S{seed}.json"
    out_json.write_text(json.dumps(record, indent=2))
 
    keys_arr = np.array(keys)
    pl.DataFrame({
        "gameId": keys_arr[:, 0],
        "playId": keys_arr[:, 1],
        "mirrored": keys_arr[:, 2].astype(bool),
        "frameId": keys_arr[:, 3],
        "pred_x_rel": preds[:, 0],
        "pred_y_rel": preds[:, 1],
        "tgt_x_rel": targets[:, 0],
        "tgt_y_rel": targets[:, 1],
    }).write_parquet(best_ckpt.with_suffix(".play_ctx_test_preds.parquet"))
 
    print(f"\n[DONE] {model_name} arm={arm} seed={seed} | "
          f"val_loss={val_loss:.4f} test_ADE={test_ade:.4f} yd | params={model.num_params:,}")
    print(f"       -> {out_json}")
    return record
 
 
def aggregate():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    if not rows:
        print("No per-run JSONs found in", RESULTS_DIR)
        return
 
    df = pl.DataFrame(rows)
    out = RESULTS_DIR / "play_context_experiment.csv"
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
            .with_columns(
                # delta = on_ade - off_ade; negative means new features improved model
                (pl.col("on_ade") - pl.col("off_ade")).alias("delta")
            )
            .sort("seed")
        )
        if paired.height:
            print(f"\n{model} paired (ON - OFF):")
            print(paired.select(["seed", "delta"]))
            print(
                f"  mean delta = {paired['delta'].mean():+.4f} "
                "(negative = play context improves ADE)"
            )
    print(f"\nWrote {out} ({len(rows)} runs)") 
 
if __name__ == "__main__":
    parser = ArgumentParser(description="Play context (offenseFormation + defendersInTheBox) late-fusion")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default=DEFAULT_MODEL)
    parser.add_argument("--arm", choices=["off", "on"], help="off=baseline, on=+play context")
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
        data = prepare_data(args.model, args.arm)
        for seed in args.seeds:
            print(f"\n===== {args.model} arm={args.arm} "
                  f"seed={seed} (device {args.device}) =====")
            train_one_seed(
                args.model, args.arm, seed, args.device, args.num_workers,
                args.batch_size, args.patience, args.max_epochs, data,
            )
