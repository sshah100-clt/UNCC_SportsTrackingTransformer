"""
Tackler-ID Training & Inference — NFL Big Data Bowl 2024

Standalone script that trains STGNN_TS in "tackler" mode
    (predict who makes the tackle, with a probability per defender)
and produces a results dataframe with the full probability distribution per play/frame.

Does not modify train.py -- checkpoints are written to a separate directory
    (models/stgnn_ts_tackler/) so they don't collide with location-prediction runs.
Results (predictions) are written to tackler_results/, separate from checkpoints.
"""

from pathlib import Path

import lightning.pytorch.callbacks as callbacks
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from datasets import RAW_FEATURE_COUNT, BDB2024_Dataset, load_datasets
from models import LitModel
from train import get_epoch_val_loss_from_ckpt

MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)

TACKLER_RESULTS_DIR = Path("tackler_results")
TACKLER_RESULTS_DIR.mkdir(exist_ok=True)


def train_tackler_model(
    model_type: str = "stgnn_ts",
    model_dim: int = 128,
    num_layers: int = 8,
    window_length: int = 10,
    topology: str | None = None,
    edge_features: bool = True,
    knn_k: int = 4,
    learning_rate: float = 1e-4,
    dropout: float = 0.3,
    batch_size: int = 256,
    device: int = 0,
    max_epochs: int = 200,
    patience: int = 10,
    skip_existing: bool = False,
) -> LitModel:
    """Train in tackler-ID mode. Mirrors train.train_model's structure."""
    version_parts = [f"M{model_dim}", f"L{num_layers}", f"W{window_length}"]
    if topology is not None:
        version_parts.append(topology)
        version_parts.append(f"K{knn_k}")
    version_parts.append(f"LR{learning_rate:.0e}")
    version = "_".join(version_parts)

    resolved_topology = topology or "full"

    logger = TensorBoardLogger(
        save_dir=MODELS_PATH,
        name=f"{model_type}_tackler",
        log_graph=False,
        default_hp_metric=False,
        version=version,
    )

    ckpt_dir = Path(logger.log_dir) / "checkpoints"
    existing_ckpt = None
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            best_ckpt = min(ckpts, key=lambda x: get_epoch_val_loss_from_ckpt(x)[1])
            existing_ckpt = str(best_ckpt)
            print(f"Resuming training from best checkpoint: {existing_ckpt}")

    if existing_ckpt is not None:
        lit_model = LitModel.load_from_checkpoint(existing_ckpt)
    else:
        lit_model = LitModel(
            model_type,
            batch_size=batch_size,
            model_dim=model_dim,
            num_layers=num_layers,
            feature_len=RAW_FEATURE_COUNT,
            learning_rate=learning_rate,
            dropout=dropout,
            window_length=window_length,
            topology=resolved_topology,
            edge_features=edge_features,
            knn_k=knn_k,
            task="tackler",
        )

    if skip_existing and existing_ckpt is not None:
        print(f"Skipping training as checkpoint exists: {existing_ckpt}")
        return lit_model

    train_ds: BDB2024_Dataset = load_datasets(
        model_type, split="train", window_length=window_length, target_type="tackler"
    )
    val_ds: BDB2024_Dataset = load_datasets(
        model_type, split="val", window_length=window_length, target_type="tackler"
    )

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=30)
    val_dataloader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True, num_workers=30)

    devices = [device] if device >= 0 else [0, 1]

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        logger=logger,
        devices=devices,
        sync_batchnorm=True,
        enable_model_summary=True,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=patience),
            callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, filename="{epoch}-{val_loss:.3f}"),
            callbacks.ModelSummary(max_depth=2),
        ],
    )

    print(lit_model.get_hyperparams())
    logger.log_hyperparams(lit_model.get_hyperparams())
    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=existing_ckpt)

    best_ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    preds_df = predict_tackler_model_as_df(lit_model, best_ckpt_path, devices[:1])

    results_out_path = TACKLER_RESULTS_DIR / f"{model_type}_{version}_{best_ckpt_path.stem}.results.parquet"
    preds_df.write_parquet(results_out_path, compression="zstd", compression_level=22)
    print(f"Wrote results to {results_out_path}")

    return lit_model


def _nfl_id_order_by_key(dataset: BDB2024_Dataset) -> list[list[int]]:
    """Per-key list of 22 nflIds, in the same order as feature_array rows
    -- needed to map logit columns back to player identity."""
    idx_df = dataset.feature_df_partition.index.to_frame(index=False)
    grouped = idx_df.groupby(["gameId", "playId", "mirrored", "frameId"], sort=False)["nflId"].apply(list)
    key_index = pd.MultiIndex.from_tuples(dataset.keys, names=["gameId", "playId", "mirrored", "frameId"])
    grouped = grouped.reindex(key_index)  # guarantee exact alignment with dataset.keys order
    return grouped.tolist()


def predict_tackler_model_as_df(model: LitModel = None, ckpt_path: Path = None, devices=1) -> pl.DataFrame:
    """
    Run inference and return a dataframe with the full per-defender probability
    distribution for every play/frame, plus the argmax prediction and correctness.

    Output columns:
        gameId, playId, mirrored, frameId, dataset_split
        tacklerNflId              -- actual tackler
        pred_tackler_nflId        -- argmax predicted tackler
        pred_tackler_prob         -- probability assigned to the argmax pick
        correct                   -- pred_tackler_nflId == tacklerNflId
        nfl_ids, probs            -- full length-22 arrays (list columns), aligned
                                      1:1, for building your own top-k / viz downstream
    """
    assert model is not None or ckpt_path is not None
    if model is None:
        model = LitModel.load_from_checkpoint(ckpt_path)

    model_type = model.hparams["model_type"]
    window_length = int(model.hparams.get("window_length", 1))

    dataloaders = {
        split: DataLoader(
            load_datasets(
                model_type,
                split=split,
                window_length=window_length,
                target_type="tackler",
            ),
            batch_size=1024,
            shuffle=False,
            num_workers=10,
        )
        for split in ["train", "val", "test"]
    }
    pred_dfs = []

    for split, dataloader in dataloaders.items():
        pred_trainer = Trainer(devices=devices, logger=False, enable_model_summary=False)
        logits = pred_trainer.predict(model, dataloaders=dataloader, ckpt_path=ckpt_path)
        logits = torch.concat(logits, dim=0)  # [N, 22], offense rows == -inf
        probs = F.softmax(logits, dim=-1).cpu().numpy()  # [N, 22]

        dataset: BDB2024_Dataset = dataloader.dataset
        ds_keys = np.array(dataset.keys)
        nfl_id_orders = _nfl_id_order_by_key(dataset)  # list of 22-length lists, aligned to ds_keys

        assert probs.shape[0] == ds_keys.shape[0] == len(nfl_id_orders)

        tgt_df = pl.from_pandas(dataset.tgt_df_partition, include_index=True)

        argmax_idx = probs.argmax(axis=-1)
        pred_nfl_id = np.array(
            [nfl_id_orders[i][argmax_idx[i]] for i in range(len(argmax_idx))], dtype=np.int64
        )
        pred_prob = probs[np.arange(len(probs)), argmax_idx].astype(np.float32).round(4)

        pred_df = (
            tgt_df.join(
                pl.DataFrame(
                    {
                        "gameId": ds_keys[:, 0],
                        "playId": ds_keys[:, 1],
                        "mirrored": ds_keys[:, 2],
                        "frameId": ds_keys[:, 3],
                        "dataset_split": split,
                        "pred_tackler_nflId": pred_nfl_id,
                        "pred_tackler_prob": pred_prob,
                        "nfl_ids": [np.array(ids, dtype=np.int64) for ids in nfl_id_orders],
                        "probs": [p.astype(np.float32).round(4) for p in probs],
                    },
                    schema_overrides={"mirrored": bool},
                ),
                on=["gameId", "playId", "mirrored", "frameId"],
                how="inner",
            )
            .with_columns(correct=(pl.col("pred_tackler_nflId") == pl.col("tacklerNflId")))
            .with_columns(**{k: pl.lit(v) for k, v in model.hparams.items()})
        )
        assert pred_df.shape[0] == len(dataset)
        pred_dfs.append(pred_df)

    return pl.concat(pred_dfs, how="vertical")


def main():
    # pure_gru
    train_tackler_model(
        model_type="pure_gru",
        num_layers=8,
        edge_features=False,
        max_epochs=200,
        device=0
    )
    # Preferred toplogy
    """
    train_tackler_model(
        model_type="stgnn_ts",
        model_dim=128,
        num_layers=8,
        window_length=10,
        topology="knn_hybrid_hub",
        edge_features=True,
        knn_k=4,
        device=0,
    )
    """


if __name__ == "__main__":
    main()