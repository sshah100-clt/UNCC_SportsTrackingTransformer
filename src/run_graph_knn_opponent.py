"""
GAT with an opposite-team k-NN topology -- self-contained, full recipe, full data.

Question: if the player-interaction step is an explicit graph where each player attends
only to its k NEAREST OPPONENTS (defender -> nearest offensive threats, and vice versa),
does that help over plain all-pairs attention (the 4.09 hybrid_ts)?

What's new here vs. the existing GAT code (src/models.py / run_graph_temporal_experiment.py):
  * A new topology, `knn_opponent`: k nearest players ON THE OPPOSITE TEAM. The shipped
    `build_adjacency_torch` only has `full`, `bipartite`, `hub`, and `knn` (k nearest of
    ALL players) -- none of which is "k nearest opponents". This is a sparse bipartite graph.
  * FULL recipe. The §9 probe ran a reduced recipe (45 epochs, lr 5e-4, no mirror, 1 seed),
    so its numbers are not comparable to the full-recipe 4.09. This runs the real recipe
    (AdamW lr 1e-4, batch 256, dropout 0.3, SmoothL1, patience 10, max 200 epochs, mirror
    augmentation) on the FULL data, so the result is directly comparable to 4.09.

Design (nothing in the existing pipeline is modified):
  * Data:    the EXISTING temporal pickle (data/datasets/temporal/), hybrid_ts at window 15
             -- the same full data behind 4.09. No re-prep, no subsample.
  * Model:   KnnOpponentSTGNN subclasses SpatioTemporalGNN and overrides ONLY adjacency
             construction for `knn_opponent` (falls back to the shipped builder for the
             other topologies). GRU / GATLayers / edge features / decoder are all reused.
  * Recipe:  GatLitModel subclasses LitModel and overrides ONLY model construction, so
             training_step / configure_optimizers (AdamW, SmoothL1) are inherited unchanged.
  * Config:  M128 / L4 / W15, ordering `ts`, edge features ON -- identical to the 4.09
             winner, so the ONLY difference vs. hybrid_ts is the interaction graph.

Config is FIXED at the hybrid_ts winner (M128/L4/W15); only the topology and k vary.

Usage (one run is fine):
    uv run python src/run_graph_knn_opponent.py --topology knn_opponent --k 5 --device 0
    # optional same-recipe control for a clean topology comparison:
    uv run python src/run_graph_knn_opponent.py --topology full          --device 0
    uv run python src/run_graph_knn_opponent.py --aggregate
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
from torch.utils.data import DataLoader

from datasets import RAW_FEATURE_COUNT, BDB2024_Dataset, load_datasets
from models import LitModel, SpatioTemporalGNN, build_adjacency_torch, edge_features_torch

# Fixed at the hybrid_ts winner so the ONLY change vs. 4.09 is the interaction graph.
MODEL_DIM, NUM_LAYERS, WINDOW, ORDERING = 128, 4, 15, "ts"
TOPOLOGIES = ["knn_opponent", "full", "bipartite", "knn"]

RESULTS_DIR = Path("results/graph_knn")
MODELS_DIR = Path("models")


# --------------------------------------------------------------------------------------
# New topology: k nearest OPPONENTS (sparse bipartite). The shipped builder has no such mode.
# --------------------------------------------------------------------------------------
def build_adjacency_knn_opponent(side: Tensor, bc: Tensor, pos: Tensor, k: int) -> Tensor:
    """(B, N, N) bool adjacency: each player connects to its k nearest OPPONENTS.

    Same conventions as models.build_adjacency_torch's `knn` (symmetrized, self-loops added)
    but neighbour selection is restricted to the opposite team. With k >= 11 (all opponents)
    this equals `bipartite`; with k < 11 it's a sparse bipartite graph. `bc` is unused
    (kept for signature parity with build_adjacency_torch).
    """
    b, n = side.shape
    dist = torch.cdist(pos, pos)  # (B, N, N)
    # Mask out same-team pairs (and self, since side_i*side_i = +1) so topk picks opponents only.
    same_team = (side.unsqueeze(2) * side.unsqueeze(1)) > 0
    dist = dist.masked_fill(same_team, 1e9)
    kk = min(k, n // 2)  # at most the number of opponents (11 for 11v11)
    idx = dist.topk(kk, dim=-1, largest=False).indices  # (B, N, kk) nearest opponents
    adj = torch.zeros(b, n, n, dtype=torch.bool, device=side.device)
    adj.scatter_(2, idx, True)
    adj = adj | adj.transpose(1, 2)  # symmetric: connected if either picked the other
    eye = torch.eye(n, dtype=torch.bool, device=side.device).unsqueeze(0)
    return adj | eye  # self-loops so no node has a fully-masked attention row


class KnnOpponentSTGNN(SpatioTemporalGNN):
    """SpatioTemporalGNN with the extra `knn_opponent` topology. Overrides only adjacency
    construction; the GRU, GATLayers, edge features, pooling and decoder are inherited."""

    def _interact(self, h: Tensor, side: Tensor, bc: Tensor, pos: Tensor, vel: Tensor) -> Tensor:
        if self.topology == "knn_opponent":
            adj = build_adjacency_knn_opponent(side, bc, pos, self.knn_k)
        else:
            adj = build_adjacency_torch(side, bc, self.topology, pos=pos, k=self.knn_k)
        edge_feats = edge_features_torch(pos, vel) if self.use_edge_features else None
        for layer in self.gat_layers:
            h = layer(h, adj, edge_feats)
        return h


# --------------------------------------------------------------------------------------
# Lightning wrapper: reuse LitModel's training recipe verbatim, swap only the model
# --------------------------------------------------------------------------------------
class GatLitModel(LitModel):
    """Inherits LitModel's training/validation/test/predict steps and AdamW optimizer
    unchanged; only model construction differs."""

    def __init__(
        self,
        topology: str,
        knn_k: int,
        feature_len: int,
        model_dim: int = MODEL_DIM,
        num_layers: int = NUM_LAYERS,
        window: int = WINDOW,
        edge_features: bool = True,
        learning_rate: float = 1e-4,
        dropout: float = 0.3,
    ):
        LightningModule.__init__(self)  # skip LitModel.__init__ (its model_type dispatch)
        self.save_hyperparameters()
        self.model = KnnOpponentSTGNN(
            feature_len,
            model_dim=model_dim,
            num_layers=num_layers,
            dropout=dropout,
            window_length=window,
            ordering=ORDERING,
            topology=topology,
            edge_features=edge_features,
            knn_k=knn_k,
        )
        self.model_type = f"stgnn_{topology}"
        self.learning_rate = learning_rate
        self.loss_fn = nn.SmoothL1Loss()
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# Train + evaluate one run
# --------------------------------------------------------------------------------------
def _val_loss_from_ckpt(ckpt_path: Path) -> float:
    m = re.search(r"val_loss=([\d.]+)", Path(ckpt_path).name)
    return float(m.group(1).rstrip(".")) if m else float("nan")


def evaluate_test_ade(model, best_ckpt, test_ds, device, num_workers):
    """Test ADE in yards over NON-mirrored frames only (project convention). ADE is in
    anchor-relative space, which equals absolute ADE (the anchor cancels)."""
    loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=num_workers)
    pred_trainer = Trainer(accelerator="gpu", devices=[device], logger=False, enable_model_summary=False)
    preds = pred_trainer.predict(model, dataloaders=loader, ckpt_path=str(best_ckpt))
    preds = torch.cat(preds, dim=0).cpu().numpy()

    keys = test_ds.keys
    mirrored = np.array([k[2] for k in keys], dtype=bool)
    targets = np.stack([test_ds.tgt_arrays[k] for k in keys]).astype(np.float32)
    assert preds.shape[0] == targets.shape[0]

    mask = ~mirrored
    dist = np.sqrt(((preds[mask] - targets[mask]) ** 2).sum(axis=-1))
    return float(dist.mean()), int(mask.sum()), preds, keys, targets


def run(topology, k, seed, device, num_workers, batch_size, patience, max_epochs, edge_features):
    seed_everything(seed, workers=True)
    feature_len = RAW_FEATURE_COUNT  # 8 raw features; the GAT reads pos/vel/side from them

    # Full data: the existing temporal pickle behind 4.09, served as W15 windows.
    train_ds: BDB2024_Dataset = load_datasets("hybrid_ts", split="train", window_length=WINDOW)
    val_ds: BDB2024_Dataset = load_datasets("hybrid_ts", split="val", window_length=WINDOW)
    test_ds: BDB2024_Dataset = load_datasets("hybrid_ts", split="test", window_length=WINDOW)

    model = GatLitModel(topology, k, feature_len, edge_features=edge_features)

    version = f"{topology}_k{k}_S{seed}"
    logger = TensorBoardLogger(save_dir=MODELS_DIR, name="graph_knn", version=version, log_graph=False)

    # drop_last not needed (GATLayer has no batch-dim BatchNorm), but kept off to match
    # the original temporal recipe exactly; feature_norm runs over batch*time*players.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
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
    test_ade, n_eval, preds, keys, targets = evaluate_test_ade(model, best_ckpt, test_ds, device, num_workers)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(
        topology=topology,
        k=k,
        edge_features=int(edge_features),
        seed=seed,
        model_dim=MODEL_DIM,
        num_layers=NUM_LAYERS,
        window=WINDOW,
        ordering=ORDERING,
        params=model.num_params,
        val_loss=val_loss,
        test_ade=round(test_ade, 4),
        n_test_nonmirrored=n_eval,
        best_ckpt=str(best_ckpt),
    )
    out_json = RESULTS_DIR / f"stgnn_{topology}_k{k}_S{seed}.json"
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
    ).write_parquet(best_ckpt.with_suffix(".knn_test_preds.parquet"))

    print(f"\n[DONE] topology={topology} k={k} seed={seed} | val_loss={val_loss:.4f} "
          f"test_ADE={test_ade:.4f} yd over {n_eval} frames | params={model.num_params:,}")
    print(f"       reference: hybrid_ts (plain attention) = 4.09 at M128/L4/W15")
    print(f"       -> {out_json}")
    return record


def aggregate():
    """Collate per-run JSONs into results/graph_knn_experiment.csv."""
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    if not rows:
        print("No per-run JSONs found in", RESULTS_DIR)
        return
    df = pl.DataFrame(rows).sort(["topology", "k", "seed"])
    out = Path("results/graph_knn_experiment.csv")
    df.write_csv(out)
    print(df.select(["topology", "k", "seed", "val_loss", "test_ade", "params"]))
    print(f"\nWrote {out} ({len(rows)} runs). Reference: hybrid_ts plain attention = 4.09.")


if __name__ == "__main__":
    parser = ArgumentParser(description="GAT with opposite-team k-NN topology (full recipe)")
    parser.add_argument("--topology", choices=TOPOLOGIES, default="knn_opponent")
    parser.add_argument("--k", type=int, default=5, help="neighbours per player (opponents for knn_opponent)")
    parser.add_argument("--no-edge-features", action="store_true", help="disable physical edge features")
    parser.add_argument("--seed", type=int, default=0)
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
        print(f"\n===== stgnn topology={args.topology} k={args.k} seed={args.seed} (device {args.device}) =====")
        run(
            args.topology,
            args.k,
            args.seed,
            args.device,
            args.num_workers,
            args.batch_size,
            args.patience,
            args.max_epochs,
            edge_features=not args.no_edge_features,
        )
