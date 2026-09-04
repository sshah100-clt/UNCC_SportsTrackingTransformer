# Player Interaction Graph Explorer

Live app: https://swapn049--graph-explorer-serve.modal.run/

(To run it locally instead: `uv run streamlit run src/graph_app.py`.)

This document has two parts. Part 1 is the visualization app. Part 2 is a small modeling experiment
that the app is meant to support. They are separate things: the app is a viewer, and the model lives
only in the experiment.

---

## Part 1: The app is a visualization tool

The app is a visualization, not a model. It draws NFL tracking plays as player graphs so you can see
the interactions on the field, frame by frame. It does not train or run any neural network. It reads
the tracking data, builds the graph (which players connect, and the distance and closing speed between
them), and renders it.

### 1.1 What it shows

For a curated set of plays, one frame at a time (play or scrub with the controls on the chart):

- Nodes are players, blue for offense and red for defense. The gold star is the designated ball carrier.
- The brown football shows where the ball actually is. The star is hollow before the handoff or catch,
  when the carrier is designated but does not yet have the ball, and solid once he is carrying. This
  reads possession from the data instead of assuming it.
- The pink ring marks the player who actually made the tackle, taken from tackles.csv.
- Edges show the chosen graph structure. Edge color shows closing speed, red for converging and blue
  for separating.
- Inspecting a player charts that player's distance and closing speed to the ball carrier across the
  whole play, for example the tackler's pursuit against a defender who got beaten.

### 1.2 How to use it

Open the live app (link above) in a browser, or run it locally. The first visit can take 10 to 20
seconds to wake up, since the hosted app sleeps when idle; after that it is instant. The controls are:

- Select play (sidebar): a curated set of clean, genuine-tackle plays, ordered short to long.
- Topology (sidebar): Complete, Bipartite (offense vs defense), or Ball-carrier hub. This sets which
  edges are drawn.
- Edge-feature coloring (sidebar): turns closing-speed coloring on the edges on or off.
- Play, Pause, and the frame slider (on the chart): play or scrub the play. It runs in the browser, so
  it is smooth and never reloads. It opens on the tackle frame; scrub left to the snap to watch the
  play develop.
- Inspect a player (below the field): pick any player to chart their distance and closing speed to the
  ball carrier across the whole play.

A short tour: open Jaylen Waddle, scrub to frame 1, and press Play. The brown football flies to the
hollow star and fills it solid at the catch, then the defense converges (edges turn red) onto the
pink-ringed tackler at the X. Switch Topology to compare Complete, Bipartite, and Hub. Then inspect the
tackler (number 58) and a defender who got beaten to compare their convergence curves.

### 1.3 Why it exists

The app builds intuition for which player interactions matter, lets you check that the graph
construction is correct before any model is built on it, and works as a debugging tool for a model's
predictions later. It does not tell you whether a graph model works. That is a modeling question,
answered by the experiment in Part 2.

---

## Part 2: The modeling experiment (this is where a GNN is used)

Part 1 is a viewer. The experiment here is the actual model. The app and the model share the same graph
definitions (the app draws them with src/graphs.py, and the model uses a torch version in src/models.py
that is checked to produce identical values), so the graphs you see in the app are the graphs the model
takes as input. The app itself does not run the model.

### 2.1 The model: a GAT, which is a graph neural network (GNN)

The experiment trains a GAT (graph attention network), a type of GNN. A GAT and a Transformer do the
same basic thing: they update each player by taking a weighted sum of information from other players,
where the weight is how much player i should attend to player j. They differ in which players are
allowed to contribute, and in what goes into that weight.

- The SportsTransformer is the all-pairs case. Every player attends to every other (a complete graph),
  and each weight is computed only from the two players' own features. It has no explicit notion of
  edges, so to use something like distance or closing speed it has to infer that from raw positions and
  velocities.
- A GAT adds two things attention does not have: a choice of which players connect (topology), and
  explicit edge features such as distance and closing speed, fed directly into the weight.

So a Transformer is the special case of a GAT on the complete graph with no edge features. The GAT used
here is built that way: standard multi-head attention, plus an adjacency mask that zeroes out
non-edges, plus an edge-feature term added to the attention score. The correctness check below
demonstrates the equivalence.

Reference points on the test set under the full training recipe: Transformer 4.25 yd ADE (the
SportsTransformer), Zoo 5.48, and a velocity-extrapolation baseline around 6.95.

### 2.2 A fast probe

This is a small, cheap experiment to quickly gauge how the GAT compares to the Transformer before
spending GPU-days on a full sweep. It answers whether the direction is worth pursuing, not what the
final numbers are.

The reduced setup, which is why it runs in minutes rather than days: trained from scratch on a
15,000-frame subsample (about 49 times less data than the full training set), single-frame raw features
(no temporal window, so it avoids the data-loading bottleneck the temporal models hit), no mirror
augmentation, one config, one seed, and at most 60 epochs. Every model has matched size (about 419k
parameters) and is trained on the same data and recipe, so comparing one row to another is fair. The
numbers are not comparable to the full-recipe 4.25.

Test-set results:

|  | no edge features | with edge features |
|---|---|---|
| complete graph | 5.218 | 5.078 |
| bipartite (offense vs defense) | 5.532 | 5.109 |

Transformer baseline on the same raw input: 5.225.

What it shows:

- The correctness check passes. The complete-graph GAT with no edge features (5.218) matches the
  Transformer (5.225) at equal size, which confirms the GAT reduces to self-attention and the
  comparison is sound.
- Edge features are the source of the gain. They help on both topologies, 0.14 better on the complete
  graph and 0.42 better on bipartite.
- Sparsifying the graph hurts. Bipartite without edge features (5.532) is the worst, since removing
  same-team edges loses signal. Even bipartite with edge features (5.109) trails the complete graph
  with edge features (5.078).
- The best result is the complete graph with explicit edge features (5.078), about 0.14 yd better than
  the Transformer equivalent at the same size.

Because this is a single seed, the gaps are directional rather than statistically established, though
they do hold on both validation and test. A full-recipe, multi-seed run (see Next steps) is needed to
confirm the size and significance of the effect.

---

## Data notes (these apply to every model, including the original)

- Source: NFL Big Data Bowl 2024. The ball carrier is a per-play label (plays.csv, ballCarrierId),
  constant across all frames, including before the handoff or catch. This is why the app distinguishes
  the designated carrier from the player who currently has the ball.
- The prediction target is the ball carrier's location at the end of the play, defined over five
  events: tackle, out_of_bounds, touchdown, qb_slide, and fumble. About 20 percent of the "tackle"
  labels are therefore not tackles. This is the original task definition and it applies equally to the
  Transformer, Zoo, the temporal models, and the GAT, so cross-model comparisons stay fair. A
  tackle-only breakdown is a clean future ablation.

## Next steps

1. Full-recipe run: full data, mirror augmentation, multiple seeds, and a capacity grid, so the result
   is directly comparable to 4.25 and carries variance bars.
2. Try a k-NN or radius topology, sparse but data-driven, since naive bipartite sparsification appears
   to hurt in the probe.
3. Extend to a spatio-temporal GNN (a per-frame graph plus recurrence) to combine with the temporal
   line of work.
4. Report a tackle-only metric alongside the end-of-play ADE.
