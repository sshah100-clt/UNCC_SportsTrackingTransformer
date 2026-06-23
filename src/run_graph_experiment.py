"""
Small single-frame head-to-head: GNN topologies vs. a Transformer on identical inputs.

Trains several architectures on the SAME single-frame RAW features (8 cols, no windowing,
no engineered diffs), same recipe (AdamW lr=1e-4, SmoothL1, dropout 0.3, early stopping),
and reports test ADE (yards). This is the fast empirical check of whether explicit graph
structure / edge features beat learned all-pairs attention.

  - transformer_raw : SportsTransformer (all-pairs attention) on raw features  -> the bar
  - gat_full_noedge : GAT, complete graph, no edge features  -> should ~ transformer (control)
  - gat_full        : GAT, complete graph + edge features
  - gat_bipartite   : GAT, offense<->defense + edge features  -> the structured contender
  - gat_hub         : GAT, ball-carrier star + edge features

ADE is computed in anchor-relative space (the anchor cancels, so it equals absolute ADE).

Run:  uv run python src/run_graph_experiment.py
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import torch
from torch import nn

from datasets import RAW_FEATURES
from models import GraphModel, SportsTransformer

PREPPED = "data/split_prepped_data"
F = len(RAW_FEATURES)
MODEL_DIM, NUM_LAYERS, DROPOUT = 128, 2, 0.3
BATCH, LR, MAX_EPOCHS, PATIENCE = 256, 5e-4, 60, 8  # higher lr to converge within the small-run budget
# Subsample for speed (the GAT attention falls back to CPU on MPS, so this is CPU-bound).
# Train + a val subset for early stopping; final test ADE uses the FULL test split.
MAX_TRAIN_FRAMES = 15_000
MAX_VAL_FRAMES = 15_000

REGISTRY = {  # name -> (topology, edge_features) ; transformer_raw handled separately
    "gat_full_noedge": ("full", False),
    "gat_full": ("full", True),
    "gat_bipartite": ("bipartite", True),
    "gat_bipartite_noedge": ("bipartite", False),
    "gat_hub": ("hub", True),
}


def build_xy(split: str, max_frames: int | None = None, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """(N, 22, F) raw single-frame features and (N, 2) anchor-relative tackle targets (non-mirrored)."""
    feat = (
        pl.read_parquet(f"{PREPPED}/{split}_features.parquet")
        .filter(~pl.col("mirrored"))
        .sort(["gameId", "playId", "frameId", "side", "is_ball_carrier", "nflId"])
    )
    tgt = pl.read_parquet(f"{PREPPED}/{split}_targets.parquet").filter(~pl.col("mirrored"))
    keys = feat.select(["gameId", "playId", "frameId"]).unique(maintain_order=True)
    x = feat.select(RAW_FEATURES).to_numpy().astype(np.float32).reshape(-1, 22, F)
    assert x.shape[0] == keys.height, f"{x.shape[0]} frames vs {keys.height} keys"
    ydf = keys.join(
        tgt.select(["gameId", "playId", "frameId", "tackle_x_rel", "tackle_y_rel"]),
        on=["gameId", "playId", "frameId"],
        how="left",
    )
    y = ydf.select(["tackle_x_rel", "tackle_y_rel"]).to_numpy().astype(np.float32)
    keep = ~np.isnan(y).any(1)
    x, y = x[keep], y[keep]
    if max_frames and x.shape[0] > max_frames:
        idx = np.random.default_rng(seed).choice(x.shape[0], max_frames, replace=False)
        x, y = x[idx], y[idx]
    return x, y


def make_model(name: str) -> nn.Module:
    if name == "transformer_raw":
        return SportsTransformer(F, model_dim=MODEL_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT)
    topo, ef = REGISTRY[name]
    return GraphModel(F, model_dim=MODEL_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT, topology=topo, edge_features=ef)


@torch.no_grad()
def ade(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device, bs: int = 1024) -> float:
    model.eval()
    dists = []
    for i in range(0, x.shape[0], bs):
        pred = model(x[i : i + bs].to(device))
        dists.append(torch.linalg.vector_norm(pred - y[i : i + bs].to(device), dim=-1).cpu())
    return float(torch.cat(dists).mean())


def train_eval(name, data, device) -> dict:
    # Data stays on CPU; only each batch is moved to the device (keeps GPU memory small).
    (xtr, ytr), (xva, yva), (xte, yte) = data
    torch.manual_seed(42)
    model = make_model(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    n = xtr.shape[0]
    best_va, best_state, bad, best_ep = float("inf"), None, 0, 0
    tic = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx].to(device)), ytr[idx].to(device))
            loss.backward()
            opt.step()
        va = ade(model, xva, yva, device)
        print(f"  [{name}] epoch {epoch:2d}  val_ade={va:.3f}  ({time.time() - tic:.0f}s elapsed)", flush=True)
        if va < best_va - 1e-4:
            best_va, best_state, bad, best_ep = (
                va,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                0,
                epoch,
            )
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    te = ade(model, xte, yte, device)
    n_params = sum(p.numel() for p in model.parameters())
    del model, opt
    if device.type == "mps":
        torch.mps.empty_cache()
    return {
        "model": name,
        "val_ade": round(best_va, 3),
        "test_ade": round(te, 3),
        "params": n_params,
        "epochs": best_ep + 1,
        "secs": round(time.time() - tic, 1),
    }


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")
    print("building data...")
    caps = {"train": MAX_TRAIN_FRAMES, "val": MAX_VAL_FRAMES, "test": None}
    raw = {s: build_xy(s, caps[s]) for s in ("train", "val", "test")}
    for s, (x, _y) in raw.items():
        print(f"  {s}: {x.shape[0]:,} frames")
    # Keep tensors on CPU; train_eval moves each batch to the device on demand.
    data = tuple((torch.from_numpy(x), torch.from_numpy(y)) for x, y in (raw["train"], raw["val"], raw["test"]))

    rows = []
    for name in ["gat_full", "gat_bipartite_noedge"]:
        print(f"\n=== training {name} ===")
        r = train_eval(name, data, device)
        print(r)
        rows.append(r)

    print("\n" + "=" * 72)
    print(f"{'model':18s} {'val ADE':>8s} {'test ADE':>9s} {'params':>9s} {'epochs':>7s} {'secs':>6s}")
    print("-" * 72)
    for r in sorted(rows, key=lambda d: d["test_ade"]):
        print(
            f"{r['model']:18s} {r['val_ade']:8.3f} {r['test_ade']:9.3f} "
            f"{r['params']:9,d} {r['epochs']:7d} {r['secs']:6.1f}"
        )
    print("-" * 72)
    print("reference (full recipe, engineered feats): transformer 4.25 · zoo 5.48 · velocity-only ~6.95")
    print(
        "NOTE: single-frame, raw feats, non-mirrored, train subsampled -> "
        "fast signal, not the final benchmark."
    )


if __name__ == "__main__":
    main()
