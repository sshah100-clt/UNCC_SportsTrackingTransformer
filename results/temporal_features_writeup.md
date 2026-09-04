# Adding Temporal Features to Sports Tracking Transformer

## Summary

We extended both the Transformer and Zoo architectures with temporal features (frame-over-frame derivatives capturing acceleration and orientation change) to test whether dynamic information improves tackle location prediction. Both architectures improved, with the Transformer benefiting more: test ADE dropped from 4.61 to 4.25 yards (7.8%), while Zoo dropped from 5.78 to 5.48 yards (5.2%). The Transformer's advantage over Zoo widened from 20.2% to 22.4%. Temporal features also shifted the performance floor: Transformer configs from M128 upward now land between 4.25 and 4.35 yards, down from the original 4.59 to 4.69 range.

## What Changed

### New Features

The original models used 6 features per player: relative position (x_rel, y_rel), velocity (vx, vy), team side, and ball carrier indicator. Orientation (ox, oy) was computed during preprocessing but not used by either model.

We added 6 new features per player, for a total of 12:

| Feature | Formula | What It Captures |
|---------|---------|-----------------|
| ox | cos(orientation) | Body orientation x-component (existing, newly exposed) |
| oy | sin(orientation) | Body orientation y-component (existing, newly exposed) |
| ax | vx[t] - vx[t-1] | X-acceleration |
| ay | vy[t] - vy[t-1] | Y-acceleration |
| delta_ox | ox[t] - ox[t-1] | Orientation change rate (x) |
| delta_oy | oy[t] - oy[t-1] | Orientation change rate (y) |

These are computed as backward differences, so no future information leaks into the current frame. The first frame of each play uses 0.0 (no change yet).

### Fair Comparison

Both architectures received the same new physical quantities, encoded in their native format:

- **Transformer (12 features/player):** Raw feature vectors. The model discovers how to use acceleration and orientation through self-attention.
- **Zoo (28 features/interaction cell, up from 10):** 18 new interaction features following Zoo's established 3-tier pattern (defender raw, defender minus ball_carrier, offense minus defense) for each new quantity (acceleration, orientation, orientation change). Each quantity adds 6 pairwise interaction features.

This ensures we measure the impact of temporal information, not a feature engineering advantage.

## Results

### Overall Performance

| Split | Zoo (Original) | Zoo (Temporal) | Transformer (Original) | Transformer (Temporal) |
|-------|---------------|----------------|----------------------|----------------------|
| Train | 5.01 | 4.72 | 4.03 | 3.97 |
| Val | 5.81 | 5.58 | 4.69 | 4.33 |
| **Test** | **5.78** | **5.48** | **4.61** | **4.25** |

Transformer vs Zoo gap on test: **20.2% to 22.4%** (widened).

Note: "Original" numbers are from the best models selected in the original paper (by validation loss): M512/L2 Transformer and M128/L2 Zoo. "Temporal" numbers are from the best models in our run, also selected by validation loss: M128/L2 Transformer and M128/L2 Zoo.

### Best Model Configurations

The original paper selected models by validation loss. The best Transformer was M512/L2 (val_loss 2.549, test ADE 4.61). The best Zoo was M128/L2 (val_loss 3.222, test ADE 5.78).

With temporal features, both configurations improved:

| Config | Zoo (Original) | Zoo (Temporal) | Transformer (Original) | Transformer (Temporal) |
|--------|---------------|----------------|----------------------|----------------------|
| M128, L2 | 5.78 | 5.48 | 4.60 | 4.25 |
| M512, L2 | 5.77 | 5.54 | 4.61 | 4.26 |

The best Zoo stays at M128/L2 (lowest val_loss in both runs). The best Transformer shifted from M512/L2 to M128/L2 by validation loss (new val_losses: M128/L2 = 2.334, M512/L2 = 2.358). On the test set, the difference between configurations is small (4.25 vs 4.26), and the real story is that the performance floor dropped uniformly: in the original results, every Transformer from M128 upward landed between 4.59 and 4.69 yards. With temporal features, that range compressed to 4.25 to 4.35. The temporal signal is additive rather than capacity-dependent.

### Performance by Game Event

| Event | Zoo (Orig) | Zoo (Temp) | Transformer (Orig) | Transformer (Temp) |
|-------|-----------|-----------|-------------------|-------------------|
| Ball Snap | 8.72 | 8.51 | 8.77 | 8.49 |
| Handoff | 6.69 | 6.34 | 6.59 | 6.38 |
| Run | 7.92 | 7.63 | 6.83 | 6.49 |
| Pass Arrived | 5.13 | 4.86 | 4.72 | 4.34 |
| Pass Caught | 4.62 | 4.33 | 4.22 | 3.75 |
| First Contact | 4.06 | 3.77 | 2.88 | 2.60 |
| Out of Bounds | 5.81 | 4.77 | 1.72 | 1.41 |
| Tackle | 4.07 | 3.74 | 0.98 | 0.97 |

Both architectures improved at every game event. The Transformer at tackle barely moved (0.98 to 0.97) since it was already near-perfect there. The largest gains came at mid-to-late play events where acceleration patterns are most informative: first contact (Zoo -7.1%, Transformer -9.7%), pass caught (Zoo -6.3%, Transformer -11.1%).

### Performance by Frames Before Tackle

| Frames Before Tackle | Zoo (Orig) | Zoo (Temp) | Transformer (Orig) | Transformer (Temp) |
|---------------------|-----------|-----------|-------------------|-------------------|
| 30+ | 9.57 | 9.13 | 9.57 | 9.13 |
| 25-30 | 5.12 | 4.93 | 5.21 | 4.65 |
| 20-25 | 4.34 | 4.21 | 4.19 | 3.61 |
| 15-20 | 3.78 | 3.70 | 3.13 | 2.57 |
| 10-15 | 3.61 | 3.48 | 2.23 | 1.81 |
| 5-10 | 3.88 | 3.61 | 1.52 | 1.38 |
| 0-5 | 4.27 | 3.84 | 1.22 | 1.16 |
| After tackle | 4.61 | 4.23 | 1.16 | 1.16 |

Key observations:

1. **Far from tackle (30+ frames):** Both models improved identically (9.57 to 9.13). At this distance there is minimal convergence signal. The improvement likely comes from orientation features providing slightly better reading of player intent.

2. **Mid-play (10-25 frames):** This is where temporal features have the most impact. The Transformer improved substantially: 2.23 to 1.81 yards at 10-15 frames (18.8% better), 3.13 to 2.57 at 15-20 frames (17.9% better). Zoo improved modestly in the same range (3.61 to 3.48 at 10-15 frames, 3.6% better). The Transformer's self-attention can leverage acceleration patterns across all 22 players simultaneously; Zoo's pairwise-independent processing limits this.

3. **Near tackle (0-5 frames, after tackle):** Both models were already strong here. Zoo improved more proportionally (4.27 to 3.84 near tackle), narrowing the relative gap. The Transformer was already at roughly 1.2 yards, close to the physical limit of prediction accuracy.

4. **Zoo's late-play degradation persists but is reduced.** In the original results, Zoo peaked at 10-15 frames (3.61) then degraded to 4.61 after tackle. With temporal features, the degradation is softer: 3.48 to 4.23. Temporal information partially compensates for Zoo's inability to capture complex multi-player convergence patterns, but the fundamental architectural limitation remains.

### Model Scaling

All 24 model configurations (12 per architecture). The original run trained 23 models (Transformer M128/L1 was not trained). Our temporal-feature run trained all 24. Params and FLOPs shown are from the temporal-feature models; they differ slightly from the original because the input dimension changed (Transformer: 6 to 12, Zoo: 10 to 28 features), affecting the first embedding layer. Original ADE numbers come from `old_results/model_comparison.json`.

**Transformer:**

| Dim | Layers | Params | FLOPs | Test ADE (Original) | Test ADE (Temporal) |
|-----|--------|--------|-------|--------------------|--------------------|
| 32 | 1 | 15K | 0.6M | 5.21 | 5.29 |
| 32 | 2 | 27K | 1.1M | 5.18 | 4.81 |
| 32 | 4 | 53K | 2.2M | 4.95 | 4.80 |
| 32 | 8 | 103K | 4.4M | 4.98 | 4.80 |
| 128 | 1 | 221K | 8.8M | * | 4.35 |
| 128 | 2 | 419K | 17.5M | 4.60 | **4.25** |
| 128 | 4 | 816K | 34.8M | 4.64 | 4.26 |
| 128 | 8 | 1.6M | 69.6M | 4.66 | 4.29 |
| 512 | 1 | 3.5M | 139.5M | 4.69 | 4.31 |
| 512 | 2 | 6.6M | 278.1M | 4.61 | 4.26 |
| 512 | 4 | 12.9M | 555.1M | 4.59 | 4.29 |
| 512 | 8 | 25.6M | 1109.2M | 4.65 | 4.27 |

\* Transformer M128/L1 was not trained in the original run.

**Zoo:**

| Dim | Layers | Params | FLOPs | Test ADE (Original) | Test ADE (Temporal) |
|-----|--------|--------|-------|--------------------|--------------------|
| 32 | 1 | 4K | 0.5M | 6.76 | 7.05 |
| 32 | 2 | 6K | 0.7M | 6.25 | 6.39 |
| 32 | 4 | 12K | 1.3M | 6.26 | 5.92 |
| 32 | 8 | 25K | 2.3M | 6.75 | 6.81 |
| 128 | 1 | 42K | 4.9M | 6.00 | 5.83 |
| 128 | 2 | 75K | 8.9M | **5.78** | **5.48** |
| 128 | 4 | 175K | 17.0M | 5.83 | 5.55 |
| 128 | 8 | 375K | 33.1M | 5.88 | 5.61 |
| 512 | 1 | 608K | 67.3M | 5.71 | 5.50 |
| 512 | 2 | 1.1M | 130.9M | 5.77 | 5.54 |
| 512 | 4 | 2.7M | 259.1M | 5.87 | 5.55 |
| 512 | 8 | 5.9M | 515.5M | 5.89 | 5.66 |

The scaling patterns are consistent with the original paper. Zoo still peaks early and degrades with more capacity. The Transformer still scales well, but now the performance floor is lower: all configs from M128 upward land between 4.25 and 4.35, compared to the original 4.59 to 4.69 range.

Note: four small M32 configs performed slightly worse with temporal features: Zoo M32/L1 (6.76 to 7.05), Zoo M32/L2 (6.25 to 6.39), Zoo M32/L8 (6.75 to 6.81), and Transformer M32/L1 (5.21 to 5.29). These models likely lack the capacity to effectively use the expanded feature sets (28 interaction features for Zoo, 12 raw features for Transformer). All M128+ configs improved in both architectures.

## Methodology Notes

- **No data leakage.** Temporal features are backward differences only (current frame minus previous frame). The first frame of each play uses 0.0.
- **Mirror augmentation.** Y-component derivatives (ay, delta_oy) are correctly negated during y-axis mirroring. X-components (ax, delta_ox) are unaffected.
- **Pipeline order.** Temporal features are computed after direction standardization (consistent coordinate system) and before mirror augmentation (mirroring handles sign flips automatically).
- **Train/test integrity.** Split remains at play level (70/15/15, seed=42). Mirrored copies of the same play always stay in the same split.
- **Evaluation.** Only non-mirrored predictions are evaluated, avoiding double-counting.
- **Same hyperparameters.** All training settings unchanged: AdamW, lr=1e-4, batch_size=256, dropout=0.3, SmoothL1Loss, early stopping patience=10, max 200 epochs.
