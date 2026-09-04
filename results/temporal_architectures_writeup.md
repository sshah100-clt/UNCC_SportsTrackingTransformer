# True Temporal Modeling for NFL Tackle Prediction: GRU and Hybrid Sequence Architectures

## Motivation

The published models in this project predict tackle location from a single frame of player tracking data. Each input is one 0.1-second snapshot of all 22 players, shape `(22, F)`. The model never sees how the play is developing. A follow-up experiment partially addressed this by adding hand-engineered backward-difference features (acceleration `ax, ay` and orientation change `delta_ox, delta_oy`). That work is complete and documented in `temporal_features_writeup.md`. It improved test ADE (Transformer 4.61 to 4.25 yards), but the model is still a single-frame model with one derivative added to it.

This experiment asks a deeper question. Does giving the model a true time window of recent frames, and letting a recurrent architecture learn the dynamics on its own, beat the single-frame approach? Instead of hand-computing one derivative, we feed the model the raw trajectory of the last T frames and let the architecture extract whatever temporal structure matters.

The research question has two parts. First, does true temporal modeling (recurrence over a frame window) improve tackle-location prediction? Second, if it does, which ingredient is responsible: the temporal information itself, the recurrence, the cross-player interaction modeling, or the order in which time and space are processed?

## Core idea

Each training example becomes a window of T consecutive frames, shape `(T, 22, F)`, ending at the current frame, instead of a single snapshot `(22, F)`. The target (tackle location) and the train, validation, and test split are unchanged. Only the input is enriched with recent history. Four new architectures consume this window, and each one is designed to isolate a single factor in the comparison.

## The four architectures

All four consume `(B, T, 22, F)` and predict `(x, y)`. They share the same decoder, feature normalization (BatchNorm), and attention settings as the original `SportsTransformer`.

| Model | Per-player encoder | Cross-player interaction | Role in the study |
|---|---|---|---|
| `windowed_transformer` | flatten T frames into one token | self-attention | Non-recurrent temporal control |
| `pure_gru` | GRU over time | none (mean-pool) | Recurrence without interaction |
| `hybrid_ts` (time to space) | GRU over time | attention over players | Recurrence with interaction |
| `hybrid_st` (space to time) | attention per frame, then GRU over time | attention (per frame) | Same as TS, opposite ordering |

`windowed_transformer` flattens each player's T frames into a single `(T*F)` token and runs the standard transformer attention. It sees all of the temporal information but processes it without recurrence. It is the control that separates having the frames from modeling them recurrently.

`pure_gru` runs a per-player GRU over time, then mean-pools the 22 players. It has recurrence but no mechanism to model how players interact.

`hybrid_ts` (time to space) encodes each player's trajectory with a GRU, then uses attention to model interaction across the 22 trajectory embeddings. This is the natural extension of the original transformer. Each player token now carries its history instead of a single snapshot.

`hybrid_st` (space to time) reverses the order. Attention models interaction at each frame, keeping per-player resolution with no pooling, and then a GRU integrates each player's contextualized sequence over time. It uses the same components as `hybrid_ts` and differs only in ordering. It is the most expensive of the four because the attention runs once per frame.

## Key design decisions and rationale

### 1. Raw features only, no engineered derivatives

The temporal models use 8 raw features per player: `x_rel, y_rel, vx, vy, ox, oy, side, is_ball_carrier`. The engineered backward-diffs (`ax, ay, delta_ox, delta_oy`) are dropped on purpose.

A diff feature like `ax = vx[t] - vx[t-1]` is a finite-difference approximation of what a recurrent network already learns from a velocity sequence, since the GRU sees both `vx[t]` and `vx[t-1]` across timesteps. Providing the hand-computed derivative is redundant. It is also noisier, because finite differences amplify tracking noise, while the measured `vx` and `vy` from the sensor are already smoothed. Keeping the diffs would also weaken the central claim. If the goal is to show that the architecture learns the dynamics, the dynamics should not be handed to it directly. The rule we follow is to keep what the sensor measures (position, velocity, orientation) plus the identity flags, and to drop what we computed by differencing.

### 2. Windowing is on-the-fly, not pre-stored

Per-frame arrays are precomputed once, as before. The window `(T, 22, F)` is assembled at access time in `__getitem__` from a per-play frame index. The start of a play is edge-padded by repeating the earliest frame, which avoids fake-zero artifacts and needs no masking. The `window_length` is set after loading, so one cached dataset serves every value of T. Storage stays flat with no T-fold increase. The keys and targets are identical to the single-frame datasets, and there is no leakage, because windows only ever pull earlier frames from the same play.

### 3. Fairness: same recipe, change only the architecture

Every temporal model uses the same training recipe as the published models: AdamW, learning rate 1e-4, batch size 256, dropout 0.3, SmoothL1Loss, early stopping with patience 10, a maximum of 200 epochs, and model selection by validation loss. They also use the same BatchNorm feature normalization, the same `num_heads` and `dim_feedforward` formulas, and the same decoder head. The only things that differ are the architecture and the raw windowed input. Any difference in results is therefore due to the modeling, not to a changed training procedure or a feature-engineering advantage. One caveat is inherent to GRUs: PyTorch applies GRU dropout only between stacked layers, so the single-layer GRU encoders in the hybrids carry no dropout inside the GRU. Dropout still applies in their attention, decoder, and embedding.

### 4. Shared dataset across the four models

All four temporal types load the same `data/datasets/temporal/` cache (8 raw features, windowed). They draw on the same source data, the same 70/15/15 play-level split (seed 42), and the same keys and targets as the zoo and transformer datasets. Only the cached column selection differs. This guarantees that the four architectures are compared on the same data, which is what makes the ablations valid.

## Experimental design: the ablation axes

The architectures are chosen so that each pairwise comparison isolates one factor.

| Comparison | Isolates | Question answered |
|---|---|---|
| single-frame baseline vs windowed models | temporal information | Does seeing recent frames help at all? |
| `windowed_transformer` vs `hybrid_ts` | recurrence vs flatten | Is recurrent processing worth more than flattening the window into attention? |
| `pure_gru` vs `hybrid_ts` | interaction modeling | Does cross-player attention add value on top of recurrence? |
| `hybrid_ts` vs `hybrid_st` | temporal and spatial ordering | Does processing time then space beat space then time? |
| sweep over T | history length | How much history helps before it plateaus or hurts? |

The published single-frame Transformer and Zoo results serve as external reference points. The main comparisons happen among the four new models, which share the same features and data.

## Scope and grid

Each architecture is trained over the full original grid, `model_dim {32, 128, 512}` by `num_layers {1, 2, 4, 8}`, which is 12 configs, multiplied by the window sweep T {5, 10, 15, 20}.

- 12 configs times 4 windows is 48 configs per architecture.
- 48 times 4 architectures is 192 total training runs.

This keeps the new models directly comparable to the published scaling analysis while adding the history-length dimension. T=1 is omitted, because a one-frame sequence makes the GRUs degenerate. The existing single-frame models serve as the no-temporal anchor.

## Metric and evaluation

The metric is ADE (Average Displacement Error) in yards, the mean Euclidean distance between predicted and true tackle `(x, y)`. Models are selected by validation loss rather than test ADE, which avoids test-set leakage, and then their test ADE is reported. Evaluation uses only non-mirrored predictions, since mirror augmentation doubles the training data but each play should be counted once. Results are broken down by data split, by game event (snap, handoff, tackle, and others), and by frames before tackle, along with a scaling table that lists every config's parameters, FLOPs, test ADE, and window length.

## Implementation summary

Five files changed, with no new dependencies.

- `datasets.py`: adds `RAW_FEATURES`, the shared `temporal` dataset, the per-play frame index, and the on-the-fly windowing in `__getitem__`. A `getattr` fallback lets datasets pickled before windowing still load.
- `models.py`: adds the four model classes and a shared `_build_decoder`, plus `LitModel` dispatch and a `window_length` hyperparameter.
- `train.py`: adds the `window_lengths` grid axis, `W{T}` in the checkpoint directory name, and threads `window_length` through training and prediction.
- `generate_results_summary.py`: generalizes the report from the zoo-vs-transformer pair to all model types. The config regex parses the optional `_W{T}`, and `model_comparison.json` records `window_length`.
- `dvc.yaml`: adds four training stages to the pipeline. The existing stages are unchanged.

## Practical notes

Temporal training is bound by data loading. Assembling T frames per sample on CPU workers dominates the runtime, so the GPUs run at low utilization and each config is slower than the single-frame originals, around 60 to 80 minutes even for the smallest model. The full 192-config sweep takes roughly 2.5 to 3.5 days across four GPUs, with one architecture per GPU, and `hybrid_st` is the slowest to finish.

To produce results after direct training, run `pick_best_models.py` and then `generate_results_summary.py` as plain scripts. They handle whatever model directories exist. The key output is `results/model_comparison.json`, the per-config scaling and window-length table that drives the analysis.

## Expected outcomes

Each possible result tells us something specific. If the hybrids beat both the single-frame baseline and `windowed_transformer`, then true temporal modeling with recurrence is justified. If `windowed_transformer` matches the hybrids, then the temporal information mattered rather than the recurrence, and attention can use a flattened window just as well. The gap between `pure_gru` and the hybrids measures the value of interaction modeling. The gap between `hybrid_ts` and `hybrid_st` shows whether ordering matters. The sweep over T shows how much history is actually useful for predicting where a tackle lands.
