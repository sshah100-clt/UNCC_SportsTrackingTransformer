"""
Spatio-temporal graph ablation on top of the best temporal model.

Takes the best temporal config (model_dim 128, num_layers 4, window 15) and asks whether
giving the player-interaction step an explicit graph (topology + edge features) helps, and
which topology / ordering is best. Trains a small set of variants on the SAME windowed raw
features and the same recipe, then reports test ADE (yards).

Variants:
  hybrid_ts                 plain attention interaction (the 4.09 baseline / control)
  stgnn_ts_full_noedge      GAT interaction, complete graph, NO edge features  -> ~ hybrid_ts (correctness)
  stgnn_ts_full_edge        GAT interaction, complete graph, edge features      -> edge-feature effect
  stgnn_ts_bipartite_edge   GAT interaction, bipartite, edge features           -> topology ablation
  stgnn_ts_knn_edge         GAT interaction, k-NN, edge features                -> data-driven sparse topology
  stgnn_st_full_edge        space->time ordering, complete, edge features       -> ts vs st ablation

This is a fast probe: the train set is subsampled and epochs capped, so it fits ~1-2h on one
GPU. Numbers are internally fair across variants but not absolute-comparable to the full-recipe
4.09. ADE is computed in anchor-relative space (the anchor cancels, equals absolute ADE).

Run:  uv run python src/run_graph_temporal_experiment.py
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import torch
from torch import nn

from datasets import RAW_FEATURES
from models import HybridST, HybridTS, SpatioTemporalGNN

PREPPED = "data/split_prepped_data"
F = len(RAW_FEATURES)
MODEL_DIM, NUM_LAYERS, WINDOW = 128, 4, 15  # the best temporal config (hybrid_ts M128/L4/W15)
BATCH, LR, MAX_EPOCHS, PATIENCE = 256, 5e-4, 45, 8
MAX_TRAIN_FRAMES = 40_000  # subsample train for speed; eval uses the full split
KNN_K = 8

# name -> (builder kind, kwargs for SpatioTemporalGNN)
VARIANTS = {
    "hybrid_ts": ("hybrid_ts", {}),
    "stgnn_ts_full_noedge": ("stgnn", dict(ordering="ts", topology="full", edge_features=False)),
    "stgnn_ts_full_edge": ("stgnn", dict(ordering="ts", topology="full", edge_features=True)),
    "stgnn_ts_bipartite_edge": ("stgnn", dict(ordering="ts", topology="bipartite", edge_features=True)),
    "stgnn_ts_knn_edge": ("stgnn", dict(ordering="ts", topology="knn", edge_features=True)),
    "stgnn_st_full_edge": ("stgnn", dict(ordering="st", topology="full", edge_features=True)),
}
RUN_ORDER = list(VARIANTS)


def build_windowed(split: str, max_frames: int | None = None, seed: int = 42):
    """Return (X, y, window_idx, train_rows).

    X: (N, 22, F) per-frame raw features (non-mirrored), ordered by play then frame.
    y: (N, 2) anchor-relative tackle targets.
    window_idx: (N, WINDOW) int row indices -- the WINDOW frames ending at each frame,
                edge-padded at the start of a play. A window for frame i is X[window_idx[i]].
    train_rows: indices to actually train on (subsampled if max_frames given).
    """
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

    # per-play start row for edge-padded windowing (frames are contiguous after the sort)
    gp = keys.select(["gameId", "playId"]).to_numpy()
    n = x.shape[0]
    is_start = np.concatenate([[True], np.any(gp[1:] != gp[:-1], axis=1)])
    play_start = np.maximum.accumulate(np.where(is_start, np.arange(n), 0))
    local = np.arange(n) - play_start
    offs = np.arange(WINDOW - 1, -1, -1)  # WINDOW-1 .. 0 (current frame last)
    window_idx = play_start[:, None] + np.clip(local[:, None] - offs[None, :], 0, None)

    valid = ~np.isnan(y).any(1)
    rows = np.where(valid)[0]
    if max_frames and rows.shape[0] > max_frames:
        rows = np.random.default_rng(seed).choice(rows, max_frames, replace=False)
    return x, y, window_idx, rows


def make_model(name: str) -> nn.Module:
    kind, kw = VARIANTS[name]
    if kind == "hybrid_ts":
        return HybridTS(F, model_dim=MODEL_DIM, num_layers=NUM_LAYERS, window_length=WINDOW)
    if kind == "hybrid_st":
        return HybridST(F, model_dim=MODEL_DIM, num_layers=NUM_LAYERS, window_length=WINDOW)
    return SpatioTemporalGNN(F, model_dim=MODEL_DIM, num_layers=NUM_LAYERS, window_length=WINDOW, knn_k=KNN_K, **kw)


@torch.no_grad()
def ade(model, X, window_idx, y, rows, device, bs=1024):
    model.eval()
    dists = []
    for i in range(0, rows.shape[0], bs):
        r = rows[i : i + bs]
        xb = X[window_idx[r]].to(device)  # (b, WINDOW, 22, F)
        pred = model(xb)
        dists.append(torch.linalg.vector_norm(pred - y[r].to(device), dim=-1).cpu())
    return float(torch.cat(dists).mean())


def train_eval(name, data, device) -> dict:
    X, y, widx, tr = data["train"]
    Xv, yv, widxv, va = data["val"]
    Xt, yt, widxt, te = data["test"]
    torch.manual_seed(42)
    model = make_model(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    n = tr.shape[0]
    best, best_state, bad, best_ep = float("inf"), None, 0, 0
    tic = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = tr[torch.randperm(n).numpy()]
        for i in range(0, n, BATCH):
            r = perm[i : i + BATCH]
            xb = X[widx[r]].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), y[r].to(device))
            loss.backward()
            opt.step()
        v = ade(model, Xv, widxv, yv, va, device)
        print(f"  [{name}] epoch {epoch:2d}  val_ade={v:.3f}  ({time.time() - tic:.0f}s)", flush=True)
        if v < best - 1e-4:
            best, best_state, bad, best_ep = (
                v,
                {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()},
                0,
                epoch,
            )
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    t = ade(model, Xt, widxt, yt, te, device)
    n_params = sum(p.numel() for p in model.parameters())
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "model": name,
        "val_ade": round(best, 3),
        "test_ade": round(t, 3),
        "params": n_params,
        "epochs": best_ep + 1,
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}  | M{MODEL_DIM}/L{NUM_LAYERS}/W{WINDOW}, train<={MAX_TRAIN_FRAMES}, epochs<={MAX_EPOCHS}")
    data = {}
    for split in ("train", "val", "test"):
        x, y, widx, rows = build_windowed(split, MAX_TRAIN_FRAMES if split == "train" else None)
        data[split] = (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(widx), rows)
        print(f"  {split}: {x.shape[0]:,} frames ({rows.shape[0]:,} used)", flush=True)

    rows = []
    for name in RUN_ORDER:
        print(f"\n=== training {name} ===", flush=True)
        r = train_eval(name, data, device)
        print(r, flush=True)
        rows.append(r)

    print("\n" + "=" * 70)
    print(f"{'model':26s} {'val ADE':>8s} {'test ADE':>9s} {'params':>10s} {'epochs':>7s}")
    print("-" * 70)
    for r in sorted(rows, key=lambda d: d["test_ade"]):
        print(f"{r['model']:26s} {r['val_ade']:8.3f} {r['test_ade']:9.3f} {r['params']:10,d} {r['epochs']:7d}")
    print("-" * 70)
    print("reference (full recipe): hybrid_ts 4.09 (M128/L4/W15). This probe is reduced data, not comparable.")
    print("reads: stgnn_ts_full_noedge ~ hybrid_ts (correctness); full_edge vs noedge = edge effect;")
    print("       full/bipartite/knn = topology; ts vs st = ordering.")


if __name__ == "__main__":
    main()
