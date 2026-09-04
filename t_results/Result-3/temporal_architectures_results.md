# NFL Tackle Prediction: From Single-Frame to True Temporal Modeling

This document describes three lines of work on the same task and dataset, explains how each model works (its input and output), and compares their performance, including how the different configurations scale. The three lines are:

1. **Original model** (SumerSports): a single-frame Transformer on raw features.
2. **Feature-engineered model**: the same Transformer fed hand-computed temporal-difference features.
3. **Temporal architectures (this work)**: four sequence models that consume a true window of frames on raw features only.

All numbers below are taken from the saved result files: `old_results/results.csv` (original), `results/results.csv` (feature-engineered), and `t_results/results.csv` plus `t_results/model_comparison.json` (temporal).

---

## 1. The task and the metric

The task is to predict where a ball carrier will be tackled, as an `(x, y)` location in yards relative to an anchor point (the ball carrier's position at the first frame of the play). The same target applies to every frame of a play.

The model makes a prediction at **every frame** of the play, not once. Tracking data is sampled at 10 frames per second, so a play of a few seconds is roughly 40 to 50 frames, and the test set is about 80,000 frames across 1,871 plays.

The metric is **Average Displacement Error (ADE)**, the mean straight-line distance in yards between the predicted and true tackle location, averaged over all frames:

ADE = (1/N) * sum over frames of sqrt((x_pred - x_true)^2 + (y_pred - y_true)^2)

Lower is better. Because it averages over all frames, the headline test ADE mixes easy late-play frames (the carrier is nearly at the tackle spot) with hard early-play frames (the play has not developed). Models are selected by validation loss, then their test ADE is reported. Only non-mirrored predictions are evaluated. The train/validation/test split is at the play level (70/15/15, seed 42) and is identical across all models.

---

## 2. How each model works (input and output)

### 2.1 Original Transformer (SumerSports)

- **Input:** a single frame, shape `(22, 6)`. The 6 raw features per player are `x_rel, y_rel, vx, vy, side, is_ball_carrier` (relative position, velocity, team side, and a ball-carrier flag). Orientation was computed in preprocessing but not used.
- **How it works:** each of the 22 players is one token. Features are batch-normalized, embedded to a model dimension, then passed through a stack of self-attention layers (a standard Transformer encoder) that lets every player's representation attend to all others. The result is average-pooled across players and decoded to two numbers.
- **Output:** the tackle location `(x, y)`.
- **Key property:** permutation-equivariant over players (no fixed ordering), with minimal feature engineering.
- **Limitation:** each prediction sees a single instant. The model predicts across the play's timeline but does not see how the play is developing (acceleration, cuts, trends).

### 2.2 Feature-Engineered Transformer

- **Input:** a single frame, shape `(22, 12)`. The original 6 features plus 6 added ones: orientation `ox, oy` (newly exposed), acceleration `ax, ay`, and orientation change `delta_ox, delta_oy`. The acceleration and orientation-change features are backward differences (current frame minus previous frame), so one step of history is encoded into the single frame.
- **How it works:** the architecture is unchanged from the original Transformer; only the per-player input is wider.
- **Output:** the tackle location `(x, y)`.
- **Idea:** inject motion information by hand so a single-frame model can sense change without seeing past frames.

### 2.3 The four temporal architectures (this work)

All four share the same input and output and use **raw features only**.

- **Input:** a window of T frames, shape `(T, 22, 8)`. The 8 raw features per player are `x_rel, y_rel, vx, vy, ox, oy, side, is_ball_carrier`. The engineered difference features (`ax, ay, delta_ox, delta_oy`) are deliberately dropped, because a recurrent model can learn motion from the frame sequence itself and finite differences add noise. The window ends at the current frame; play starts are edge-padded by repeating the earliest frame.
- **Output:** the tackle location `(x, y)`.

The four models differ in how each player's window is encoded and whether players interact:

- **windowed_transformer (non-recurrent control).** Each player's T frames are flattened into one `(T x 8)` token, then the standard Transformer attention runs over the 22 players. It has all the temporal information but processes it without recurrence. This isolates "temporal information through attention" from recurrence.

- **pure_gru (recurrence, no interaction).** A GRU runs over each player's T-frame trajectory and produces one embedding per player; the 22 embeddings are mean-pooled and decoded. There is no mechanism for players to interact.

- **hybrid_ts (time then space).** A GRU encodes each player's trajectory into one embedding, then self-attention models interaction across the 22 trajectory embeddings, then pool and decode. This is the encode-then-interact pattern.

- **hybrid_st (space then time).** Self-attention models interaction across the 22 players at each frame (shared weights, no pooling), then a GRU integrates each player's sequence of frame-contextualized embeddings over time, then pool and decode. Same components as hybrid_ts, opposite ordering. It is the most expensive because attention runs once per frame.

All four predict at every frame, exactly like the originals; the only change is that each prediction now sees a window of recent frames instead of one snapshot.

---

## 3. Architecture details

All models here are **regression** models, not generative ones. A single forward pass maps the input directly to two continuous numbers, the predicted tackle `(x, y)`. There is no sampling, no autoregression, and no token or sequence generation. The "output" is just those two coordinates produced by the final layer, and the model is trained to minimize the distance between the predicted and the true tackle location.

The word "Transformer" here refers to the **encoder-only self-attention network** from "Attention is All You Need" (multi-head self-attention over a set of tokens), built and trained from scratch on tracking data. It is **not** a pretrained language model (not GPT or BERT), there is no tokenizer and no text, and there is no generative decoder. In this project the 22 players are the tokens, and the encoder is used as a permutation-equivariant set encoder: it lets every player's representation attend to all the others, with no positional encoding (player order is arbitrary, so the model must not depend on it).

### 3.1 Shared building blocks

These components are reused across the models below.

- **Feature normalization:** `BatchNorm1d` over the per-player feature vector.
- **Feature embedding (where used):** `Linear(feature_len -> model_dim)`, then `ReLU`, `LayerNorm`, `Dropout`.
- **Transformer encoder:** a stack of `num_layers` standard `TransformerEncoderLayer` blocks. Each block is multi-head self-attention (heads = `min(16, max(2, 2*round(model_dim/64)))`, which is 2, 4, and 16 heads at width 32, 128, and 512) followed by a position-wise feedforward network of width `4*model_dim`, with residual connections, post-layer-normalization, and dropout. It runs over the 22 player tokens with no positional encoding.
- **GRU (where used):** PyTorch `nn.GRU` with hidden size `model_dim`. It reads a sequence frame by frame and its final hidden state summarizes the whole sequence. Dropout applies only between stacked GRU layers, so a single-layer GRU has no dropout inside it.
- **Player pooling:** average across the 22 players, giving one vector per play and making the output permutation-invariant.
- **Decoder (regression head, identical in every model):** `Linear(model_dim -> model_dim)`, `ReLU`, `Dropout`, `Linear(model_dim -> model_dim/4)`, `ReLU`, `LayerNorm`, `Linear(model_dim/4 -> 2)`. The final linear layer produces the two output coordinates. The loss is SmoothL1 (Huber) between predicted and true `(x, y)`.

### 3.2 Original and Feature-engineered Transformer (the base model)

- **Input:** one frame, `(22, F)`, with `F = 6` (original) or `F = 12` (feature-engineered).
- **Pipeline:** BatchNorm, embed each player to `model_dim`, Transformer encoder (attention across the 22 players), mean-pool players, decoder, output `(x, y)`.
- The original and feature-engineered versions are the **same architecture**; they differ only in the number of input features.

### 3.3 windowed_transformer

- **Input:** a window `(T, 22, 8)`.
- **Encoder:** each player's T frames are concatenated into a single `T*8` vector, so the scene becomes `(22, T*8)`, and the base Transformer runs with `feature_len = T*8`. The window is absorbed by the linear embedding; there is no recurrence.
- **Then:** attention across the 22 players, mean-pool, decoder, output `(x, y)`.

### 3.4 pure_gru

- **Input:** a window `(T, 22, 8)`.
- **Encoder:** a GRU of depth `num_layers` reads each player's T-frame trajectory; its final hidden state is that player's embedding (22 embeddings).
- **Then:** mean-pool the 22 embeddings (no attention), decoder, output `(x, y)`.

### 3.5 hybrid_ts (time then space)

- **Input:** a window `(T, 22, 8)`.
- **Encoder:** a single-layer GRU reads each player's trajectory into one embedding (22 embeddings).
- **Interaction:** the Transformer encoder (`num_layers`) runs attention across those 22 trajectory embeddings.
- **Then:** mean-pool, decoder, output `(x, y)`. This is the encode-then-interact ordering.

### 3.6 hybrid_st (space then time)

- **Input:** a window `(T, 22, 8)`.
- **Per-frame interaction:** a linear embedding maps each frame to `model_dim`, then the Transformer encoder (`num_layers`, shared weights) runs attention across the 22 players within each frame independently, with no pooling, producing per-frame contextualized embeddings.
- **Temporal integration:** a single-layer GRU then reads each player's sequence of those per-frame embeddings over time.
- **Then:** mean-pool, decoder, output `(x, y)`. Attention runs once per frame, which makes this the most expensive of the four.

---

## 4. Training recipe (held constant for fairness)

Every model above uses the same recipe so that differences come from the architecture and input, not the training procedure: AdamW, learning rate 1e-4, batch size 256, dropout 0.3, SmoothL1Loss, early stopping with patience 10, maximum 200 epochs, model selection by validation loss, and the same BatchNorm feature normalization and decoder head.

Each architecture is trained over a grid of model width (model_dim in {32, 128, 512}) and depth (num_layers in {1, 2, 4, 8}), which is 12 configurations. The four temporal architectures multiply this by a window sweep T in {5, 10, 15, 20}, giving 48 configurations each and 192 temporal runs in total.

---

## 5. Headline results

Best configuration per architecture, selected by validation loss, with test ADE in yards:

| Line of work | Model | Best config | Features | Input | Test ADE |
|---|---|---|---|---|---|
| Original | Transformer | M512 / L2 | 6 raw, single frame | `(22, 6)` | 4.61 |
| Feature-engineered | Transformer | M128 / L2 | 12 (raw + diffs), single frame | `(22, 12)` | 4.25 |
| Temporal | windowed_transformer | M128 / L2 / W5 | 8 raw, window | `(T, 22, 8)` | 4.24 |
| Temporal | pure_gru | M128 / L4 / W5 | 8 raw, window | `(T, 22, 8)` | 4.42 |
| Temporal | **hybrid_ts** | **M128 / L4 / W15** | 8 raw, window | `(T, 22, 8)` | **4.09** |
| Temporal | hybrid_st | M128 / L2 / W5 | 8 raw, window | `(T, 22, 8)` | 4.58 |

Reference baseline (Zoo architecture, single frame): 5.78 original, 5.48 feature-engineered. All four temporal models beat it.

The main findings:

- **The best result of all three lines is `hybrid_ts` at 4.09 yards.** It beats the original Transformer (4.61) by 0.52 yards (11.3 percent) and the feature-engineered Transformer (4.25) by 0.16 yards (3.8 percent).
- **`hybrid_ts` does this with raw features only (8) and no hand-engineering**, whereas the feature-engineered model needed 12 features with hand-computed derivatives. The architecture learns the motion that was previously computed by hand, and ends up more accurate.
- **`windowed_transformer` (4.24) essentially ties the feature-engineered Transformer (4.25).** Feeding raw window frames to attention recovers what the hand-engineered diffs provided, with no feature engineering. Recurrence (`hybrid_ts`) then improves further.

The clean progression on the selected best model is **4.61 to 4.25 to 4.09**.

---

## 6. Configuration and scaling analysis

### 6.1 Model width (model_dim): M128 is the sweet spot

Best test ADE at each width (lowest test ADE among all layer and window settings for that width):

| Model | M32 | M128 | M512 |
|---|---|---|---|
| windowed_transformer | 4.59 | 4.24 | 4.31 |
| pure_gru | 4.77 | 4.41 | 4.43 |
| hybrid_ts | 4.28 | **4.09** | 4.19 |
| hybrid_st | 4.90 | 4.58 | 4.57 |

For reference, the Transformer lineage shows the same pattern:

| Model | M32 (best) | M128 (best) | M512 (best) |
|---|---|---|---|
| Original Transformer | 4.95 | 4.60 | 4.59 |
| Feature-engineered Transformer | 4.80 | 4.25 | 4.26 |

Across every architecture, the jump from M32 to M128 is large, and M512 is roughly tied with or slightly worse than M128. Capacity beyond M128 does not help. This reproduces the scaling behavior reported in the original paper and is a useful internal consistency check.

### 6.2 Depth (num_layers)

Depth helps up to a point. For the winning `hybrid_ts`, the best width-128 results come at 4 layers. For `windowed_transformer`, 2 layers is best. Beyond that, deeper layers do not improve and sometimes slightly regress, consistent with the width finding that this task does not reward very large models.

### 6.3 Window length (T): modest effect, plateaus around 1.0 to 1.5 seconds

For the best `hybrid_ts` family (M128 / L4), test ADE by window length is nearly flat:

| T (frames) | seconds | hybrid_ts M128/L4 test ADE |
|---|---|---|
| 5 | 0.5 | 4.13 |
| 10 | 1.0 | 4.12 |
| 15 | 1.5 | 4.09 |
| 20 | 2.0 | 4.15 |

More history helps a little, with the optimum around 10 to 15 frames, and longer windows do not keep improving. This matches the intuition that the tackle location is mostly decided by recent convergence, so a second or so of history is enough.

---

## 7. Ablations within the four temporal models

Each pairwise comparison isolates one factor. All numbers are the best test ADE per architecture from the table in section 5.

| Comparison | Numbers | Effect | Conclusion |
|---|---|---|---|
| Recurrence vs flatten | windowed_transformer 4.24 to hybrid_ts 4.09 | -0.15 (-3.5%) | Encoding the trajectory with a GRU beats flattening it into attention. Recurrence helps. |
| Interaction on vs off | pure_gru 4.42 to hybrid_ts 4.09 | -0.33 (-7.5%) | Adding cross-player attention on top of the GRU is the single biggest gain. |
| Ordering (time-space vs space-time) | hybrid_st 4.58 to hybrid_ts 4.09 | -0.49 (-10.7%) | Time then space is much better than space then time. |
| Window length | see section 6.3 | small | About 1.0 to 1.5 seconds of history is the sweet spot. |

The ordering result is the most striking. `hybrid_st` is the worst of the four, worse even than `pure_gru`, which has no interaction at all. So running attention per frame and then a GRU is actively harmful compared with just a GRU. Encoding each trajectory first and then modeling interaction (`hybrid_ts`) is clearly the right ordering for this task.

---

## 8. Breakdown analysis

The breakdowns slice the same per-frame predictions in two ways. The per-event slice averages error only on frames tagged with a named game event (one frame per play for most events). The frames-before-tackle slice groups all frames by how far they are from the tackle. Both roll up to the overall test ADE.

### 8.1 By game event (test set)

Transformer lineage and the best temporal model:

| Event | Original Transformer | FE Transformer | hybrid_ts |
|---|---|---|---|
| Ball snap | 8.77 | 8.49 | 8.45 |
| Handoff | 6.59 | 6.38 | 6.09 |
| Run | 6.83 | 6.49 | 6.59 |
| Pass arrived | 4.72 | 4.34 | 4.30 |
| Pass caught | 4.22 | 3.75 | 3.66 |
| First contact | 2.88 | 2.60 | 2.46 |
| Out of bounds | 1.72 | 1.41 | 1.27 |
| Tackle | 0.98 | 0.97 | 0.79 |

`hybrid_ts` is best at nearly every event, and its advantage is largest at the late, decisive moments: tackle (0.79 vs 0.97, about 18 percent better than the feature-engineered model), out of bounds (1.27 vs 1.41), and first contact (2.46 vs 2.60). These are the frames where player convergence matters most, which is exactly where temporal modeling should help. The one event where the feature-engineered model is marginally better is `run` (6.49 vs 6.59).

Events are not all mid-play moments. They span the whole timeline, from the snap (start, farthest from the tackle, highest error) to the tackle (end, the carrier is at the spot, lowest error). Sorting events by ADE traces the play timeline.

### 8.2 By frames before tackle (test set)

| Frames before tackle | Original Transformer | FE Transformer | hybrid_ts |
|---|---|---|---|
| 30+ (early) | 9.57 | 9.13 | 8.85 |
| 25 to 30 | 5.21 | 4.65 | 4.38 |
| 20 to 25 | 4.19 | 3.61 | 3.47 |
| 15 to 20 | 3.13 | 2.57 | 2.54 |
| 10 to 15 | 2.23 | 1.81 | 1.78 |
| 5 to 10 | 1.52 | 1.38 | 1.34 |
| 0 to 5 | 1.22 | 1.16 | 1.03 |
| After tackle | 1.16 | 1.16 | 0.91 |

Error falls smoothly from about 9 yards far from the tackle to about 1 yard at the tackle, for every model. The overall test ADE is the frame-weighted average of these buckets, pulled up by the large early-play group (the 30+ bucket has by far the most frames). `hybrid_ts` is best in every bucket, including the hard early frames, where it improves the 30+ group from 9.13 (feature-engineered) to 8.85.

---

## 9. Extension: explicit graph structure (a spatio-temporal GAT probe)

The temporal ablations in section 7 say the single biggest lever is cross-player interaction (the attention step), not recurrence or ordering. This raised a natural question: the interaction step in `hybrid_ts` is a self-attention layer, which is one particular graph operation, so would giving it an *explicit* graph (a choice of which players connect, plus hand-supplied edge quantities like distance and closing speed) do better than letting plain attention infer all of that from raw positions? This section answers that with a small probe built directly on the winning `hybrid_ts` configuration.

### 9.1 GNN and GAT in one paragraph, and the key equivalence

A graph neural network (GNN) treats the 22 players as nodes connected by edges and updates each player by passing messages along its edges: every player gathers information from its neighbors and updates its own representation. A graph attention network (GAT) is the version where a player aggregates its neighbors with a *learned, content-dependent weighted average*, the weight on neighbor `j` being an attention coefficient computed from the pair. This is the same operation a Transformer performs. In fact a Transformer's self-attention is exactly a GAT on the **complete graph with no edge features**: every player attends to every other, and each attention weight is computed only from the two players' own feature vectors. So `hybrid_ts`'s player-interaction step is already a GAT. It is not a different model family.

A general GAT adds two levers that plain self-attention does not have:

1. **Topology.** An adjacency mask zeroes out non-edges before the attention softmax, so a player only aggregates from chosen neighbors. This lets the graph be complete (everyone), bipartite (offense attends only to defense and vice versa), a ball-carrier hub (everyone attends to the carrier), or k-nearest-neighbor (each player attends to its k closest). The Transformer is fixed to the complete graph.
2. **Edge features.** A per-pair vector is fed directly into the attention score, for example score `= query_i . key_j + g(edge_ij)`. The plain Transformer has no such input and must infer pairwise quantities from the two players' raw positions and velocities. A GAT hands them over explicitly.

The precise statement, then, is: **Transformer self-attention = GAT on a complete graph with no edge features.** The only two things a graph buys over the `hybrid_ts` we already have are the topology choice and the explicit edge features. This probe isolates exactly those.

### 9.2 The architectures

These models take `hybrid_ts` (and `hybrid_st`) and replace only the player-interaction attention with a GAT layer. The GRU encoder, the ordering, the player pooling, and the decoder head from section 3.1 are unchanged.

- **GAT layer.** Masked multi-head attention (the adjacency mask implements the topology), plus an edge-feature term added to the attention scores, followed by the same feedforward and residual structure as a Transformer encoder layer. With the complete topology and edge features turned off, this layer is the standard self-attention of `hybrid_ts`, which is the correctness control below.
- **Edge features (6 per pair).** Computed from the players' raw positions and velocities: relative position (2), distance (1), relative velocity (2), and closing speed (1, positive when the two players are converging). For the time-then-space model these are read from the current (last) frame of the window; for the space-then-time model they are recomputed at every frame.
- **stgnn_ts (time then space).** A GRU encodes each player's trajectory into one embedding, then the GAT layer models interaction across the 22 embeddings, then pool and decode. Same shape as `hybrid_ts`, with the GAT in place of plain attention.
- **stgnn_st (space then time).** The GAT layer models interaction across the 22 players at each frame, then a GRU integrates each player's per-frame embeddings over time, then pool and decode. Same shape as `hybrid_st`.
- **Topologies tested:** complete, bipartite, and k-nearest-neighbor (k = 8, by inter-player distance, symmetrized, with self-loops).

### 9.3 Probe setup (a reduced, internally fair recipe)

This is a fast single-config probe, not a full sweep. It fixes the winning temporal configuration (model_dim 128, num_layers 4, window 15) and trains six variants on the full training set, but with a shorter recipe so all six finish quickly on one GPU: learning rate 5e-4, at most 45 epochs with early-stopping patience 8, a single seed, and **no mirror augmentation** (it trains on the non-mirrored frames only). Because of these choices the absolute numbers land about 0.3 yard above the full-recipe `hybrid_ts` of 4.09. Every variant uses the identical reduced recipe, so the comparisons between rows are fair even though the absolute level is not comparable to section 5.

### 9.4 Results

Best configuration per variant, test ADE in yards, sorted best first:

| Variant | Ordering | Topology | Edge features | Val ADE | Test ADE | Params | Best epoch |
|---|---|---|---|---|---|---|---|
| stgnn_ts_bipartite_edge | ts | bipartite | on | 4.468 | **4.329** | 866,978 | 9 |
| stgnn_ts_full_edge | ts | complete | on | 4.478 | 4.377 | 866,978 | 10 |
| stgnn_ts_knn_edge | ts | k-NN (k=8) | on | 4.446 | 4.388 | 866,978 | 20 |
| stgnn_ts_full_noedge | ts | complete | off | 4.498 | 4.407 | 866,866 | 16 |
| hybrid_ts (control) | ts | complete | none (plain attention) | 4.480 | 4.414 | 866,866 | 15 |
| stgnn_st_full_edge | st | complete | on | 5.069 | 4.875 | 914,466 | 7 |

Each comparison isolates one factor:

| Comparison | Numbers (test ADE) | Effect | Conclusion |
|---|---|---|---|
| Correctness control | full_noedge 4.407 vs hybrid_ts 4.414 | +0.007 | The complete-graph GAT with edge features off matches `hybrid_ts`. The GAT correctly reduces to self-attention, so the implementation is sound and the comparison is valid. |
| Edge features | full_edge 4.377 vs full_noedge 4.407 | -0.030 | Adding distance and closing speed improves complete-graph attention by 0.03 yard, which is within single-seed noise. |
| Topology | bipartite 4.329, complete 4.377, k-NN 4.388 (all with edges) | 0.06 spread | The three topologies are within 0.06 yard. Bipartite is marginally best here, the opposite of the single-frame probe (where complete beat bipartite), which indicates these gaps are noise rather than a real ranking. |
| Ordering | st 4.875 vs ts 4.377 | +0.50 | Space then time is far worse than time then space, replicating the section 7 ordering result (`hybrid_st` vs `hybrid_ts`). |

### 9.5 What this means

The probe gives one large, reproducible result and one clean negative result.

- **Ordering replicates.** Space then time (`stgnn_st`) is about 0.5 yard worse than time then space, exactly the pattern section 7 found for `hybrid_st` vs `hybrid_ts`. This is the trustworthy signal, and it holds whether the interaction step is plain attention or a GAT.
- **Explicit graph structure does not help at the temporal scale.** All five time-then-space variants fall within 0.085 yard of each other. Edge features move the complete-graph model by only 0.03 yard, and the topology ranking flips relative to the single-frame probe, both of which mark these differences as single-seed noise. The interpretation is not that graphs are weaker than `hybrid_ts`: `hybrid_ts` is itself a GAT, and the control confirms it. Rather, once a GRU encodes each trajectory and attention models the interaction, the model already captures the pairwise information that explicit edge features would supply, so handing it distance and closing speed adds nothing detectable.

This matches the single-frame probe, where edge features gave only about 0.14 yard, and shows that small advantage shrinking to nothing once true temporal modeling is in place. Establishing any sub-0.1 yard edge-feature or topology effect would require the full recipe (mirror augmentation, the long learning-rate schedule, a capacity grid) run over multiple seeds with variance bars; at single seed and a reduced recipe the differences here are not separable from noise.

---

## 10. Methodological notes and caveats

The implementations were audited for correctness (tensor shapes, player and time alignment through the reshapes, GRU and attention usage, permutation equivariance, attention-head divisibility). They are correct and the four models instantiate the four intended conditions. Four honest caveats affect how cleanly the comparisons can be read:

1. **Feature set: the temporal models use 8 features, the original used 6.** The temporal feature set is the original six (`x_rel, y_rel, vx, vy, side, is_ball_carrier`) plus the two orientation features (`ox, oy`). Orientation is a raw sensor-measured quantity (the player's facing direction), not a hand-computed derivative, so it was kept under the rule of using raw measured features while dropping the engineered difference features. The feature-engineered model also uses orientation, so the **temporal vs feature-engineered comparison (4.09 vs 4.25) is matched on orientation** and isolates learned windowing versus hand-computed diffs. The **temporal vs original comparison (4.61 to 4.09), however, bundles two changes**, the time window and the added orientation feature, so it is not a clean windowing-only number. The internal ablations (recurrence, interaction, ordering) are unaffected, because all four temporal models use the identical 8 features. A perfectly clean windowing-only test against the original would retrain the temporal models on the matched 6 features.

2. **Capacity matching for the interaction comparison.** `pure_gru` uses a GRU of depth `num_layers`, while the hybrids use a single-layer GRU encoder plus `num_layers` of attention. So `pure_gru` vs `hybrid_ts` differs by both the added attention and the GRU depth. The direction (interaction helps) holds, but it is not a perfectly clean single-variable swap.

3. **Structural asymmetry between the two hybrids.** `hybrid_ts` lets its GRU map raw features to the model dimension (the GRU also does the embedding), while `hybrid_st` needs a separate linear embedding before attention (attention requires the model dimension) and its GRU maps model dimension to model dimension. This is unavoidable given the ordering, but it means the time-then-space vs space-then-time comparison carries a small extra embedding layer on the space-then-time side. The ordering gap (4.09 vs 4.58) is large enough that this is very unlikely to be the cause.

4. **Normalization and hardware.** BatchNorm normalizes features across batch, time, and player positions together, a consistent choice across all four temporal models. Also, due to a GPU memory limit, the largest `hybrid_st` configurations were trained on a different machine than the other models; with synced BatchNorm and the same seed the effect is small, but it is a cross-hardware note worth recording.

None of these caveats change the headline conclusions, which have comfortable margins.

---

## 11. Summary

- **Single frame, then hand-engineered diffs, then true temporal modeling is a clean improvement:** 4.61 to 4.25 to 4.09 on the selected best model.
- **Less feature engineering, better result.** The best temporal model uses 8 raw features and beats the 12-feature engineered model. The architecture subsumes the manual derivatives.
- **Recurrence helps (about 3.5 percent), interaction helps more (about 7.5 percent),** and together they give the best model, `hybrid_ts`.
- **Ordering matters a lot.** Time then space beats space then time by about 10.7 percent, and space then time is so poor it is worse than no interaction at all. This is the most novel finding.
- **The temporal advantage is largest at the decisive late-play moments** (tackle, first contact, out of bounds), which is the strongest evidence that the model reads developing dynamics rather than a single snapshot.
- **M128 is the capacity sweet spot and history plateaus around 1.0 to 1.5 seconds,** both consistent with the original paper.
- **Explicit graph structure adds nothing over `hybrid_ts` at the temporal scale (section 9).** `hybrid_ts` is already a graph attention network (a complete-graph GAT), and giving it explicit edge features or a different topology moves test ADE by less than 0.1 yard, within single-seed noise, while the time-then-space over space-then-time ordering result replicates strongly. The interaction the model needs is already learned by the GRU and attention.

---

## Appendix A: Full hybrid_ts configuration sweep (48 runs)

Every hybrid_ts configuration from the 192-run temporal sweep (model_dim in {32, 128, 512} x num_layers in {1, 2, 4, 8} x window T in {5, 10, 15, 20} = 48 runs). All use the recipe in section 4 (AdamW, lr 1e-4, batch 256, dropout 0.3, SmoothL1Loss, patience 10, max 200 epochs, 8 raw features). Models are selected by validation loss; test ADE is reported but not used for selection. Window T is in frames (10 frames = 1 second). FLOPs are per single inference. Numbers are from t_results/model_comparison.json.

The selected best configuration is **M128 / L4 / W15** (lowest validation loss, 2.256), which is also the lowest test ADE (4.09); it is shown in bold.

| model_dim | num_layers | window T (frames) | params | inference FLOPs | val loss | test ADE (yd) |
|---|---|---|---|---|---|---|
| 32 | 1 | 5 | 18,106 | 1,422,288 | 2.597 | 4.67 |
| 32 | 1 | 10 | 18,106 | 2,293,488 | 2.630 | 4.74 |
| 32 | 1 | 15 | 18,106 | 3,164,688 | 2.711 | 4.86 |
| 32 | 1 | 20 | 18,106 | 4,035,888 | 2.544 | 4.60 |
| 32 | 2 | 5 | 30,810 | 1,970,000 | 2.545 | 4.61 |
| 32 | 2 | 10 | 30,810 | 2,841,200 | 2.544 | 4.59 |
| 32 | 2 | 15 | 30,810 | 3,712,400 | 2.478 | 4.51 |
| 32 | 2 | 20 | 30,810 | 4,583,600 | 2.468 | 4.49 |
| 32 | 4 | 5 | 56,218 | 3,065,424 | 2.431 | 4.42 |
| 32 | 4 | 10 | 56,218 | 3,936,624 | 2.360 | 4.32 |
| 32 | 4 | 15 | 56,218 | 4,807,824 | 2.419 | 4.39 |
| 32 | 4 | 20 | 56,218 | 5,679,024 | 2.523 | 4.59 |
| 32 | 8 | 5 | 107,034 | 5,256,272 | 2.390 | 4.36 |
| 32 | 8 | 10 | 107,034 | 6,127,472 | 2.383 | 4.34 |
| 32 | 8 | 15 | 107,034 | 6,998,672 | 2.348 | 4.29 |
| 32 | 8 | 20 | 107,034 | 7,869,872 | 2.365 | 4.28 |
| 128 | 1 | 5 | 272,050 | 20,312,736 | 2.347 | 4.29 |
| 128 | 1 | 10 | 272,050 | 31,902,336 | 2.350 | 4.29 |
| 128 | 1 | 15 | 272,050 | 43,491,936 | 2.347 | 4.33 |
| 128 | 1 | 20 | 272,050 | 55,081,536 | 2.362 | 4.33 |
| 128 | 2 | 5 | 470,322 | 28,991,648 | 2.312 | 4.26 |
| 128 | 2 | 10 | 470,322 | 40,581,248 | 2.310 | 4.22 |
| 128 | 2 | 15 | 470,322 | 52,170,848 | 2.303 | 4.24 |
| 128 | 2 | 20 | 470,322 | 63,760,448 | 2.287 | 4.23 |
| 128 | 4 | 5 | 866,866 | 46,349,472 | 2.268 | 4.13 |
| 128 | 4 | 10 | 866,866 | 57,939,072 | 2.270 | 4.12 |
| **128** | **4** | **15** | **866,866** | **69,528,672** | **2.256** | **4.09** |
| 128 | 4 | 20 | 866,866 | 81,118,272 | 2.268 | 4.15 |
| 128 | 8 | 5 | 1,659,954 | 81,065,120 | 2.271 | 4.15 |
| 128 | 8 | 10 | 1,659,954 | 92,654,720 | 2.284 | 4.12 |
| 128 | 8 | 15 | 1,659,954 | 104,244,320 | 2.290 | 4.12 |
| 128 | 8 | 20 | 1,659,954 | 115,833,920 | 2.304 | 4.19 |
| 512 | 1 | 5 | 4,283,026 | 315,307,488 | 2.399 | 4.38 |
| 512 | 1 | 10 | 4,283,026 | 491,421,888 | 2.364 | 4.31 |
| 512 | 1 | 15 | 4,283,026 | 667,536,288 | 2.349 | 4.31 |
| 512 | 1 | 20 | 4,283,026 | 843,650,688 | 2.364 | 4.31 |
| 512 | 2 | 5 | 7,435,410 | 453,832,160 | 2.362 | 4.25 |
| 512 | 2 | 10 | 7,435,410 | 629,946,560 | 2.350 | 4.30 |
| 512 | 2 | 15 | 7,435,410 | 806,060,960 | 2.346 | 4.28 |
| 512 | 2 | 20 | 7,435,410 | 982,175,360 | 2.340 | 4.23 |
| 512 | 4 | 5 | 13,740,178 | 730,881,504 | 2.334 | 4.24 |
| 512 | 4 | 10 | 13,740,178 | 906,995,904 | 2.334 | 4.19 |
| 512 | 4 | 15 | 13,740,178 | 1,083,110,304 | 2.333 | 4.22 |
| 512 | 4 | 20 | 13,740,178 | 1,259,224,704 | 2.317 | 4.19 |
| 512 | 8 | 5 | 26,349,714 | 1,284,980,192 | 2.335 | 4.24 |
| 512 | 8 | 10 | 26,349,714 | 1,461,094,592 | 2.331 | 4.20 |
| 512 | 8 | 15 | 26,349,714 | 1,637,208,992 | 2.352 | 4.24 |
| 512 | 8 | 20 | 26,349,714 | 1,813,323,392 | 2.337 | 4.20 |

A few patterns visible in the full sweep, consistent with section 6:

- **Width:** the jump from M32 to M128 is the large one (best test ADE 4.28 at M32, 4.09 at M128). M512 does not improve on M128 (best 4.19) despite roughly 16 to 30 times the parameters and up to 26 times the FLOPs, so capacity beyond M128 is wasted on this task.
- **Depth:** within M128, 4 layers is the sweet spot; 1 layer underfits (best 4.29) and 8 layers does not beat 4 (best 4.12).
- **Window:** within M128 / L4, test ADE is nearly flat across T (4.13, 4.12, 4.09, 4.15 for T = 5, 10, 15, 20), confirming history plateaus around 1.0 to 1.5 seconds.
- **Selection by validation loss is well behaved:** the lowest val loss (2.256) coincides with the lowest test ADE (4.09), so the selected model is not an artifact of the selection metric.
