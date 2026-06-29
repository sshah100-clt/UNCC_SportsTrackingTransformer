"""
Phase 1b: player physical attributes (weight, height) as per-player features.

Question: does adding each player's static size prior -- `weight_Z`, `height_Z` -- improve
tackle-location prediction, on the FULL data and FULL recipe?

Unlike game-state (per-PLAY, late-fused), weight/height are PER-PLAYER (they vary across the
22, constant over frames). So they are appended straight to each player's feature vector
(8 -> 10) and flow through the GRU/attention normally -- NO late fusion, no model subclass.
The "on" arm is simply HybridTS(feature_len=10) / SportsTransformer(feature_len=10).

Design (nothing in the existing pipeline is modified):
  * Data:   the EXISTING temporal pickle (data/datasets/temporal/), hybrid_ts at window 15
            (transformer control at window 1). weight_Z/height_Z are already in the parquets
            and (critically) in the dataset's own `feature_df_partition`.
  * ALIGNMENT (the one real risk): the 2 columns are read from the base dataset's
            `feature_df_partition` with the SAME `.loc[key]` the features come from, so the
            22 weight/height rows are in the exact same nflId order as the 22 feature rows.
            Guaranteed aligned, not assumed.
  * Recipe: uses LitModel directly (no late fusion needed), so AdamW / SmoothL1 / steps are
            the project's own. Trainer block mirrors train.py.train_model.
  * Config: M128/L4/W15 (hybrid_ts), fixed -- only the 2 features change vs. the 4.09 baseline.

The "off" arm here is byte-identical to phase-1 `hybrid_ts off` (plain model, 8 features, same
recipe/seeds). Run both arms here for a self-contained pair, OR run "on" only and compare to
the phase-1 off JSONs.

Usage:
    uv run python src/run_physical_features.py --model hybrid_ts --arm on   --device 0   # seeds 0-4
    uv run python src/run_physical_features.py --model hybrid_ts --arm off  --device 1   # (or reuse phase-1 off)
    uv run python src/run_physical_features.py --aggregate
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
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from datasets import RAW_FEATURE_COUNT, BDB2024_Dataset, load_datasets
from models import LitModel

PHYSICAL_FEATURES = ["weight_Z", "height_Z"]  # per-player static priors, already in the parquets

CONFIGS = {
    "hybrid_ts": dict(model_dim=128, num_layers=4, window=15),  # the 4.09 winner
    "transformer": dict(model_dim=128, num_layers=2, window=1),  # single-frame control
}

RESULTS_DIR = Path("results/physical")
MODELS_DIR = Path("models")


# --------------------------------------------------------------------------------------
# Per-player weight/height lookup, read from the base dataset's OWN partition (guaranteed
# aligned to the feature arrays' player order).
# --------------------------------------------------------------------------------------
def build_physical_lookup(base: BDB2024_Dataset) -> dict[tuple, np.ndarray]:
    """Map (gameId, playId, mirrored) -> float32 (22, 2) of [weight_Z, height_Z].

    Read from base.feature_df_partition via the SAME `.loc[(g,p,m,frame)]` indexing the
    feature arrays use, so the 22 rows are in identical nflId order. weight/height are
    constant across frames, so any frame of the play gives the right (and aligned) values.
    """
    fp = base.feature_df_partition  # pandas, MultiIndex (gameId,playId,mirrored,frameId,nflId)
    lut: dict[tuple, np.ndarray] = {}
    for (g, p, m), frames in base.play_frames.items():
        sub = fp.loc[(g, p, m, frames[0])]  # 22 rows, same order as the feature array
        wh = sub[PHYSICAL_FEATURES].to_numpy(dtype=np.float32)
        assert wh.shape == (22, len(PHYSICAL_FEATURES)), f"bad shape {wh.shape} for play {(g, p, m)}"
        assert not np.isnan(wh).any(), f"NaN weight/height for play {(g, p, m)}"
        lut[(g, p, m)] = wh
    return lut


class PhysicalDataset(Dataset):
    """Wrap a loaded BDB2024_Dataset and append per-player [weight_Z, height_Z] -> (..., 22, 10).

    Handles both windowed (T, 22, F) and single-frame (22, F) base items. weight/height are
    play-constant, so the same (22, 2) is broadcast across all T window frames.
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
        wh = self.lut[(g, p, m)]  # (22, 2) aligned to player order
        if feat.ndim == 3:  # (T, 22, F)
            t, players, _ = feat.shape
            wh_b = np.broadcast_to(wh, (t, players, wh.shape[1]))
        else:  # (22, F)
            wh_b = wh
        out = np.concatenate([feat, wh_b], axis=-1).astype(np.float32)
        return out, tgt


# --------------------------------------------------------------------------------------
# Data prep (once per process) + train/eval one seed
# --------------------------------------------------------------------------------------
def prepare_data(model_name: str, arm: str):
    """Load the FULL datasets once. on -> wrap with weight/height (feature_len 10); off ->
    plain base (feature_len 8). Returns (cfg, train_ds, val_ds, eval_ds, test_base, feature_len)."""
    cfg = CONFIGS[model_name]
    wl = cfg["window"]  # hybrid_ts -> 15, transformer -> 1
    train_base = load_datasets("hybrid_ts", split="train", window_length=wl)
    val_base = load_datasets("hybrid_ts", split="val", window_length=wl)
    test_base = load_datasets("hybrid_ts", split="test", window_length=wl)

    if arm == "on":
        train_ds = PhysicalDataset(train_base, build_physical_lookup(train_base))
        val_ds = PhysicalDataset(val_base, build_physical_lookup(val_base))
        eval_ds = PhysicalDataset(test_base, build_physical_lookup(test_base))
        feature_len = RAW_FEATURE_COUNT + len(PHYSICAL_FEATURES)  # 10
    else:
        train_ds, val_ds, eval_ds = train_base, val_base, test_base
        feature_len = RAW_FEATURE_COUNT  # 8

    return cfg, train_ds, val_ds, eval_ds, test_base, feature_len


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


def train_one_seed(model_name, arm, seed, device, num_workers, batch_size, patience, max_epochs, data):
    cfg, train_ds, val_ds, eval_ds, test_base, feature_len = data
    seed_everything(seed, workers=True)

    # LitModel directly: no late fusion, so the plain HybridTS/SportsTransformer at the right
    # feature_len IS the model. Inherits the full project recipe (AdamW, SmoothL1, steps).
    model = LitModel(
        model_type=model_name,
        batch_size=batch_size,
        model_dim=cfg["model_dim"],
        num_layers=cfg["num_layers"],
        feature_len=feature_len,
        learning_rate=1e-4,
        dropout=0.3,
        window_length=cfg["window"],
    )

    version = f"{arm}_S{seed}"
    logger = TensorBoardLogger(save_dir=MODELS_DIR, name=f"physical_{model_name}", version=version, log_graph=False)

    # drop_last=True matches phase-1 `hybrid_ts off` (so the off arm here reproduces it, and
    # off/on stay paired on samples-per-epoch). Harmless for on (no batch-dim BatchNorm).
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(
        model=model_name,
        arm=arm,
        seed=seed,
        feature_len=feature_len,
        physical_features=PHYSICAL_FEATURES if arm == "on" else [],
        model_dim=cfg["model_dim"],
        num_layers=cfg["num_layers"],
        window=cfg["window"],
        params=model.num_params,
        val_loss=val_loss,
        test_ade=round(test_ade, 4),
        n_test_nonmirrored=n_eval,
        best_ckpt=str(best_ckpt),
    )
    (RESULTS_DIR / f"{model_name}_{arm}_S{seed}.json").write_text(json.dumps(record, indent=2))

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
    ).write_parquet(best_ckpt.with_suffix(".phys_test_preds.parquet"))

    print(f"\n[DONE] {model_name} arm={arm} seed={seed} | val_loss={val_loss:.4f} "
          f"test_ADE={test_ade:.4f} yd over {n_eval} frames | feature_len={feature_len} params={model.num_params:,}")
    return record


def aggregate():
    """Collate per-run JSONs into results/physical_experiment.csv with mean +/- std and paired delta."""
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    if not rows:
        print("No per-run JSONs found in", RESULTS_DIR)
        return
    import statistics as st

    pl.DataFrame([{k: v for k, v in r.items() if k != "physical_features"} for r in rows]).write_csv(
        "results/physical_experiment.csv"
    )
    for model in sorted({r["model"] for r in rows}):
        print(f"\n--- {model} ---")
        arms = {}
        for arm in ["off", "on"]:
            a = sorted([r for r in rows if r["model"] == model and r["arm"] == arm], key=lambda r: r["seed"])
            arms[arm] = {r["seed"]: r["test_ade"] for r in a}
            if a:
                ades = [r["test_ade"] for r in a]
                print(f"  {arm:3s} seeds={[r['seed'] for r in a]} mean={st.mean(ades):.4f} "
                      f"std={st.pstdev(ades) if len(ades) > 1 else 0:.4f}")
        seeds = sorted(set(arms.get("off", {})) & set(arms.get("on", {})))
        if seeds:
            deltas = [arms["on"][s] - arms["off"][s] for s in seeds]
            print(f"  paired (on-off) {seeds}: {[round(d, 4) for d in deltas]}")
            print(f"  mean delta = {st.mean(deltas):+.4f}  (negative = weight/height helps)")
    print("\nWrote results/physical_experiment.csv")


if __name__ == "__main__":
    parser = ArgumentParser(description="Phase 1b: player weight/height as per-player features")
    parser.add_argument("--model", choices=["hybrid_ts", "transformer"], help="architecture")
    parser.add_argument("--arm", choices=["off", "on"], help="off=baseline (8 feat), on=+weight/height (10 feat)")
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
        data = prepare_data(args.model, args.arm)  # load full data + (for on) build aligned lookup ONCE
        for seed in args.seeds:
            print(f"\n===== physical {args.model} arm={args.arm} seed={seed} (device {args.device}) =====")
            train_one_seed(
                args.model, args.arm, seed, args.device,
                args.num_workers, args.batch_size, args.patience, args.max_epochs, data,
            )
