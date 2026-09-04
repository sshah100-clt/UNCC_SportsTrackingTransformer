# Three Ablations on the Temporal Tackle-Prediction Model

This document reports three experiments built on the best temporal model, `hybrid_ts` (test ADE
about 4.09): adding game-state context, adding player physical attributes (weight and height), and
sweeping a k-nearest-opponent graph topology. Each asks whether a specific addition improves
tackle-location prediction. All three share one base architecture, one training recipe, and a 5-seed
protocol. Every number below is mean test ADE in yards across the 5 seeds, lower is better.

Summary: none of the additions improves on the temporal baseline. The two changes large enough to
stand out from the seed-to-seed spread both make ADE worse (over-sparsifying the graph, and adding
weight/height). Everything else lands inside the spread of the baseline.

## 1. Shared setup

### 1.1 Task and metric
Predict where the ball carrier is tackled, as an `(x, y)` location in yards relative to an anchor
(the carrier's position at the first frame of the play). The metric is ADE (Average Displacement
Error): the mean Euclidean distance between predicted and true `(x, y)`, in yards. Because the target
is anchor-relative, ADE in relative space equals absolute ADE. Data is the NFL Big Data Bowl 2024
tracking set, split 70/15/15 at the play level (seed 42). Models are selected by validation loss and
reported by test ADE on non-mirrored frames only (about 80,062 frames). Mirror augmentation doubles
the training data (about 740k samples).

### 1.2 The base architecture: `hybrid_ts` (time then space)
All three experiments modify this one model. It consumes a window of T = 15 frames of all 22 players,
shape `(batch, 15, 22, F)`, in two stages:

1. Per-player GRU (time): a single-layer GRU runs over each player's 15-frame trajectory and outputs
   one embedding per player, that player's recent history compressed to a vector.
2. Self-attention across players (space): a Transformer encoder lets the 22 player embeddings attend
   to each other, modeling interactions such as which defenders are converging on the carrier.

The 22 attended embeddings are mean-pooled (permutation-invariant) and passed to a small decoder
(`Linear, ReLU, Dropout, Linear, ReLU, LayerNorm, Linear`) that outputs the two coordinates. It is a
regression model: one forward pass maps input to `(x, y)`, with no sampling or autoregression.

Input features (8 raw per player): `x_rel, y_rel, vx, vy, ox, oy, side, is_ball_carrier` (relative
position, velocity, orientation, team side, ball-carrier flag). No hand-engineered derivatives; the
GRU learns motion from the sequence.

Fixed config (the established winner): `model_dim = 128`, `num_layers = 4` (attention depth),
`window = 15`. Holding this fixed means any ADE change is due to the experiment's addition, not to
capacity or history length, both of which were optimized in prior work.

### 1.3 Training recipe (held constant everywhere)
AdamW, learning rate 1e-4, batch size 256, dropout 0.3, SmoothL1 (Huber) loss, BatchNorm feature
normalization, early stopping with patience 10, max 200 epochs, selection by validation loss, mirror
augmentation on. Identical across every run below; only the architecture or feature addition changes.

### 1.4 Multi-seed protocol
Every configuration is trained on 5 seeds (0 to 4). Within an experiment the compared arms share
seeds, so each comparison is paired: the same initialization and data-shuffle order, with only the
addition different. We report mean and standard deviation across seeds and the paired per-seed delta
(the value for one arm minus the other, computed seed by seed). The seed-to-seed spread is about 0.015
to 0.02 yd, which sets the scale for reading the deltas: a change of this size is the same as the
noise from picking a seed. A single-frame `SportsTransformer` control (M128/L2, no window) is included
in the feature experiments to check whether an effect holds for a weaker model that sees only one
frame.

## 2. Game-state context

### What changed
Added three play-level features the model never saw before: `down`, `yardsToGo`, and `distanceToGoal`
(yards from the line of scrimmage to the target end zone). All three come from `plays.csv` and are
constant for all 22 players across the whole play. `playResult` (the play's outcome) was left out
because it would leak the target.

### How it was added (late fusion)
Game-state is play-level, not per-player. Feeding it through the per-player GRU and attention would
repeat the same three numbers 22 by 15 times and then average them away in the pool. Instead it is
late-fused: a small branch (`BatchNorm, Linear, ReLU`) encodes the three values, and its output is
concatenated to the pooled player embedding just before the decoder. The per-player encoder is
unchanged; only the decoder input widens.

### Architecture and config
`hybrid_ts` (M128/L4/W15) and the single-frame `SportsTransformer` control (M128/L2), each off (8
features) versus on (8 plus 3 game-state, late-fused), 5 seeds.

### Results

| Model | off | on | on minus off |
|---|---|---|---|
| hybrid_ts | 4.120 | 4.126 | +0.006 (no consistent direction across seeds) |
| transformer (control) | 4.603 | 4.589 | -0.014 (lower in 4 of 5 seeds) |

Game-state leaves `hybrid_ts` where it was: the mean moves by +0.006 and the per-seed deltas split
in both directions. On the single-frame transformer it lowers ADE by 0.014, in 4 of 5 seeds. A model
that already sees a window of motion has little to gain from situational context. The single-frame
model, which sees one snapshot, gains a small amount. Both changes are inside the seed spread.

## 3. Player physical attributes (weight and height)

### What changed
Added two per-player features: `weight_Z` and `height_Z`, the z-scored weight and height of each
player from `players.csv`. Unlike game-state these vary across the 22 players (and are constant over
time for a given player), so they are a real per-player signal, a static physical prior. A heavier
carrier may break tackles; a lineman moves differently from a back.

### How it was added
Because they are per-player they are appended to each player's feature vector (8 to 10) and pass
through the GRU and attention normally, with no late fusion. They were read from the dataset's own
per-frame table so each player receives its own weight and height (the alignment was verified). The
model is `hybrid_ts` with `feature_len = 10`.

### Architecture and config
`hybrid_ts` (M128/L4/W15) and the `SportsTransformer` control (M128/L2), each off (8 features) versus
on (10 features), 5 seeds.

### Results

| Model | off | on | on minus off |
|---|---|---|---|
| hybrid_ts | 4.108 | 4.172 | +0.064 (higher, i.e. worse, in all 5 seeds) |
| transformer (control) | 4.608 | 4.607 | -0.001 (no consistent direction) |

Weight and height make `hybrid_ts` worse. The on arm is higher in all five seeds, by +0.035 to +0.094
yd, mean +0.064, well outside the seed spread. The transformer is unchanged (mean -0.001, deltas
split). A likely reason is redundancy: the model already sees the motion that a player's size
produces (its velocity and acceleration), so the two static features add parameters and noise without
new information, and generalization drops. This is the clearest negative of the three experiments.

## 4. k-nearest-opponent graph (GAT) topology sweep

### What changed
Replaced `hybrid_ts`'s plain player self-attention with a graph attention network (GAT) whose
interaction graph is restricted: each player connects only to its k nearest opponents (a sparse
bipartite graph), plus explicit physical edge features. The question was whether focusing each player
on its nearest few opponents, and handing the model distance and closing speed directly, beats plain
all-pairs attention. We swept k in {2, 3, 5, 8} and compared against the full (complete-graph) GAT.

### What a GAT is, and why this is a generalization
A GAT updates each player by a learned weighted average of its neighbors, which is what a
Transformer's self-attention does. The difference is two extra levers: a choice of topology (an
adjacency mask zeroes out non-edges before the attention softmax, here keeping only the k nearest
opponents), and edge features (a per-pair vector added to the attention score). Plain self-attention
is the special case of a complete graph with no edge features, so `hybrid_ts`'s interaction step is
already a GAT. This experiment only varies the topology and turns edge features on.

Edge features (6): relative position (2), distance (1), relative velocity (2), and closing speed (1,
positive when converging), computed from raw positions and velocities at the current frame.

Architecture: `SpatioTemporalGNN` with time-then-space ordering: per-player GRU, then a GAT
interaction in place of plain attention, then mean-pool and decode. Same fixed config M128/L4/W15,
866,978 parameters, identical to the plain model up to the GAT.

### Results (mean test ADE over 5 seeds)

| Topology | mean ADE | std | minus full |
|---|---|---|---|
| knn_opponent k=2 | 4.154 | 0.019 | +0.046 (higher, i.e. worse, in all 5 seeds) |
| knn_opponent k=3 | 4.119 | 0.009 | +0.011 (higher in all 5 seeds) |
| knn_opponent k=5 | 4.096 | 0.016 | -0.012 |
| knn_opponent k=8 | 4.089 | 0.023 | -0.019 (lower in 4 of 5 seeds) |
| full (complete graph) | 4.108 | 0.014 | reference |

ADE falls as k grows: 4.154, 4.119, 4.096, 4.089 for k = 2, 3, 5, 8. Denser graphs do better, the
opposite of the hypothesis. The two sparsest, k=2 and k=3, are higher than the full graph in all five
seeds, by +0.046 and +0.011. The densest tried, k=8 (8 of 11 opponents), edges below full by 0.019,
lower in 4 of 5 seeds, a gap inside the seed spread. Restricting each player to a few nearest
opponents does not help; broader connectivity is better. Edge features were on for every GAT run, so
this compares topology, not the edge-feature effect, which a prior probe found within the seed spread.

## 5. Across the three experiments

| Change | mean delta | direction across seeds |
|---|---|---|
| Game-state on hybrid_ts | +0.006 | mixed |
| Game-state on transformer | -0.014 | lower (better) in 4 of 5 |
| Weight/height on hybrid_ts | +0.064 | higher (worse) in all 5 |
| Weight/height on transformer | -0.001 | mixed |
| kNN k=2 vs full | +0.046 | higher (worse) in all 5 |
| kNN k=8 vs full | -0.019 | lower (better) in 4 of 5 |

The `hybrid_ts` baseline and the neutral changes cluster between about 4.09 and 4.13 yd. The two
changes large enough to clear the seed spread both raise ADE above this: cutting the graph to a few
neighbors (k=2 at 4.154) and adding weight and height (4.172). The changes that lower ADE (game-state
on the transformer, the k=8 graph) move by no more than the seed spread, where a single seed could
swing the sign. The reading is that `hybrid_ts`, a per-player GRU over about 1.5 seconds of history
followed by attention across all 22 players on raw features, is the ceiling for this task and data.
Added play or player context, and explicit graph structure, do not beat it. The single-frame
transformer control sits separately near 4.60, as expected for a model that sees one frame, and is
used only to test whether a feature helps a weaker model.

Running five seeds rather than one matters here because the seed spread (about 0.015 to 0.02 yd) is as
large as most of the changes being tested. A single run could show a 0.01 yd gain that is only the
choice of seed. Across five seeds the picture is steady: nothing improves on the baseline, and two
additions hurt it. The one remaining idea with a concrete reason to help would be a more
tackle-relevant edge quantity such as time-to-intercept rather than raw distance; everything tried so
far points to the temporal baseline being the ceiling.

## Appendix: configurations

| | base model | input | config | added |
|---|---|---|---|---|
| Game-state | hybrid_ts, transformer | 8 raw plus 3 play-level (late-fused) | M128/L4/W15 ; M128/L2 | down, yardsToGo, distanceToGoal |
| Physical | hybrid_ts, transformer | 8 raw plus 2 per-player | M128/L4/W15 ; M128/L2 | weight_Z, height_Z |
| kNN-opponent | SpatioTemporalGNN (GAT) | 8 raw plus 6 edge features | M128/L4/W15, time-then-space | k-nearest-opponent topology, k in {2, 3, 5, 8} |

All runs: AdamW lr 1e-4, batch 256, dropout 0.3, SmoothL1, patience 10, max 200 epochs, 5 seeds,
selection by validation loss, non-mirrored test ADE. Per-run results in `gamestate/`, `physical/`,
and `graph_knn/` (one JSON per run); aggregated CSVs via each experiment's `--aggregate`.
