# Additional Feature Experiments on the hybrid_ts Baseline

## Executive narrative

*For a reader who knows the project but not this particular study.*

The temporal study left us in a good place. The winning model, hybrid_ts, runs a per-player GRU
over a 15-frame window and then attends across the 22 players, and it predicts tackle location at
about 4.1 yards of average displacement error using only eight raw per-player features: position
relative to the ball carrier, velocity, facing direction, a team flag, and a ball-carrier flag. It
beats the published transformer and the earlier engineered-difference variant, and it does so with
almost no feature engineering. That is the satisfying part. The uncomfortable part is that the Big
Data Bowl data holds far more than those eight numbers. We know each player's roster position, age,
and body size. We know the pre-snap state of every play: the formation, the defenders in the box,
the score, the down and distance, the win probability. None of it reaches the model. So the
question this study set out to answer is simple to state and hard to resist: is the architecture
genuinely finished learning from raw geometry, or is it leaving accuracy on the table only because
we never handed it the rest of what the data knows?

We answered it with three experiments that add features in the three ways the data allows, each
matched to the kind of feature it is. Per-player signals that change every frame (acceleration,
speed) are fed through the GRU and attention alongside the original eight. Static per-player
identity (position, age, size) is broadcast across the window and fed the same way. Whole-play
context (formation, score, down) is late-fused, meaning it is held out of the sequence model and
joined to the play representation only after the players are pooled, so it cannot be averaged away.
Everything was kept honest in the ways that matter for a feature study. Every arm was compared
against the same eight-feature baseline retrained on the same three random seeds, so the comparison
is paired rather than a contest between two separately tuned numbers. Every added feature is known
before the snap or is a physical property of the player, so nothing can leak the tackle outcome.
And every value that could be missing was imputed from the training split only. The result is ten
feature arms, each run on three seeds, all judged by a per-seed paired difference against that
shared baseline.

The finding runs against the naive expectation. More information did not translate into more
accuracy. Not one arm meaningfully beat the baseline, and several were clearly worse. The single
best arm added per-frame acceleration and improved the baseline by about four hundredths of a yard,
a gain of roughly one percent. It is small, but it is real in the sense that matters here: it beats
the paired baseline on all three seeds, not on average alone. That result also makes mechanical
sense. Acceleration is the second derivative of position, and a recurrent model reading a short
window of frames apparently cannot fully reconstruct it, so handing it over directly recovers a
little accuracy that the geometry alone was missing. Speed and step-distance, which merely restate
the velocity the model already has, added nothing on top of it, which is exactly the redundancy we
would predict.

Everything else was neutral or harmful. Static player identity was the clearest failure. Roster
position cost about a tenth of a yard, adding body size on top made it worse, and both effects held
on every seed. A constant per-player label broadcast across the window gives the model extra
parameters and a fixed signal it can memorize, not a spatial cue it can localize with. Pre-snap
context was a wash. Formation, box count, score, and down describe what kind of play is being run,
not where on the field the runner will be brought down, and the tracked positions of the 22 players
already carry that spatial information. The maximalist arm that combined all of it at once was among
the worst results in the study, which is the cleanest statement of the whole finding: stacking
features does not stack their benefits. The one small win from acceleration is swamped by the
identity and context channels, and the extra capacity hurts generalization rather than helping it.

The take-home for the project is that the minimal-feature design was close to the right call. On
this task the model is nearly saturated on raw relative geometry. The only feature worth adding is
per-frame acceleration, and even that is a careful one-percent gain rather than a breakthrough.
Everything we might have assumed would help, player role, physical build, game situation, is inert
or counterproductive here. Two caveats keep this honest. The effects are small and rest on three
seeds, so the acceleration gain is directionally solid but not yet a significance claim, and the
age feature is diluted by a large fraction of missing birth dates. The natural next step, a run that
keeps only the features that helped and drops the rest, is scoped and ready. But the headline is
already clear, and it is a useful one to be able to state with evidence: for tackle localization,
the black box was not starving for features. It had almost everything it needed in the geometry all
along.

## Purpose

The temporal study left us with a clear winner: `hybrid_ts` (per-player GRU over time, then
attention across players), at model dimension 128, 4 layers, and a 15-frame window. It reaches
that result using only 8 raw per-player features (position, velocity, orientation, side, and a
ball-carrier flag). The dataset ships with far more than 8 usable fields, so the question here is
simple to state: if we hand the same architecture more of what the data already knows about each
player and each play, does it predict the tackle location any better, or is the raw geometry
already enough?

These three experiments answer that. They hold the architecture and the training recipe fixed and
change only the input features. Each experiment pairs one or more "on" arms (extra features added)
against a single shared "off" baseline (the plain 8-feature model), and every arm is run on the
same three random seeds so the comparison is paired rather than a comparison of two separately
tuned numbers.

## What stays fixed

Everything except the input features is held constant so that any change in test error is
attributable to the features and nothing else.

- Architecture: `hybrid_ts`, model dimension 128, 4 layers, 15-frame window. This is the exact
  configuration that won the temporal study.
- Training recipe: AdamW, learning rate 1e-4, batch size 256, dropout 0.3, SmoothL1 loss, early
  stopping with patience 10, maximum 200 epochs. The checkpoint with the best validation loss is
  the one evaluated. This recipe is identical to the temporal models, so the comparison is fair.
- Evaluation: test ADE (Average Displacement Error) in yards, defined as the mean Euclidean
  distance between the predicted and true tackle location. It is computed over the 80,062
  non-mirrored test frames only (the mirrored augmentation copies are never scored).
- Seeds: 0, 1, and 2. Every arm is trained three times, once per seed. Reported numbers are the
  mean over the three seeds, and every "does it help" claim is a per-seed paired difference
  against the off baseline trained on the same seed.

## The baseline, and an important note on 4.09 versus 4.148

The temporal writeup quotes `hybrid_ts` at test ADE 4.09. That number was the single best
configuration selected out of a large sweep. It is the right number to advertise as the best model
we have, but it is not the right number to compare these arms against, because it was cherry-picked
as the minimum of many runs.

The honest baseline for a paired feature study is the same 8-feature model retrained on the same
three seeds used by every arm here. That baseline (the `off` arm, trained once and shared across
all three experiments) comes out at:

| seed | off test ADE |
|------|--------------|
| 0    | 4.1285       |
| 1    | 4.1798       |
| 2    | 4.1353       |
| mean | 4.148        |

So the reference point in this writeup is 4.148, not 4.09. The gap between the two is exactly what
you expect from selecting a minimum over a sweep versus averaging three fresh seeds. Judging the
arms against 4.09 would unfairly penalize all of them; judging against the same-seed 4.148 is the
correct paired test. The off baseline has 866,866 trainable parameters.

## How features enter the model

The dataset offers three kinds of extra information, and each kind can only be fused into the model
in a way that respects what it is. The experiments use three matching mechanisms.

1. Per-frame per-player features. These change every frame for every player (for example
   acceleration or speed). They are handed to the model as extra channels on the per-player vector,
   so they flow through the GRU and the attention exactly like the original 8 features. They are
   read by slicing channels out of an extended temporal dataset (`temporal_ext`, 15 channels) whose
   first 8 channels are byte-for-byte identical to the standard temporal dataset. That identity was
   verified directly: slicing `temporal_ext[..., :8]` reproduces the baseline windows with a maximum
   absolute difference of 0. This guarantees the per-frame arms sit on the exact same footing as the
   4.148 baseline.

2. Static per-player features. These are constant for a player within a play (for example roster
   position, age, or body size). They are looked up as a (22, k) table and broadcast across the 15
   window frames before being concatenated to the per-player vector, so they too pass through the
   GRU and attention. The lookup reads players in the same nflId-sorted order the feature arrays use,
   so the appended rows line up with the correct player.

3. Per-play context features. These are single values for the whole play (for example the offensive
   formation or the score). Averaging them through the per-player pooling would destroy them, so they
   are late-fused instead: the per-play vector is split off inside the model, passed through its own
   small branch (batch norm, linear, ReLU), and concatenated to the pooled player embedding just
   before the decoder. It never enters the GRU or the attention and is never pooled away. This is the
   same late-fusion mechanism used in the earlier gamestate experiment, copied into these scripts so
   they do not depend on it.

All numeric nulls in the per-play features are imputed with the mean over the training plays only,
so validation and test never inform the imputation. A missing offensive formation becomes an
all-zero one-hot, which the model can read as "unknown."

Leakage was avoided by construction. Every feature used here is either a physical property of the
player, a kinematic quantity available at prediction time, or a pre-snap play descriptor. No
post-snap fields, no play result, and no event labels are used.

## Feature dictionary: what each channel is and what it means

All coordinates share one frame. The pipeline first standardizes play direction so the offense
always moves in the positive x direction, then expresses every player position relative to a
per-play anchor (the ball carrier's location at the start of the play). So x_rel is downfield
offset from the ball carrier and y_rel is lateral offset, both in yards. Speeds and orientations
live in the same rotated frame. This is worth keeping in mind because it is why the raw geometry is
already so informative: the model sees every player's position, motion, and facing relative to the
ball carrier.

### Baseline: the 8 raw per-player features (present in every arm)

These are the inputs the 4.148 baseline already uses. Every arm keeps all 8 and adds to them.

| feature          | definition                                                        | meaning on the field                                              |
|------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|
| x_rel            | player x minus ball-carrier x at play start                       | how far downfield the player is relative to the ball carrier       |
| y_rel            | player y minus ball-carrier y at play start                       | how far to the side the player is relative to the ball carrier     |
| vx               | speed times cosine of motion direction                            | downfield component of velocity (yards per second)                 |
| vy               | speed times sine of motion direction                              | lateral component of velocity (yards per second)                   |
| ox               | cosine of facing angle                                            | downfield component of where the player is facing (unit vector)    |
| oy               | sine of facing angle                                              | lateral component of where the player is facing (unit vector)      |
| side             | +1 if on the offense (possession team), -1 if on defense          | which team the player belongs to                                   |
| is_ball_carrier  | 1 for the player carrying the ball, else 0                        | flags the runner the tackle is being predicted for                 |

### Experiment 1 additions (per-player, fed through the GRU and attention)

| arm         | feature   | definition                                                             | meaning on the field                                                        |
|-------------|-----------|------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| dynamics    | ax        | frame-to-frame change in vx (backward difference), 0 on the first frame | how hard the player is accelerating or braking downfield                     |
| dynamics    | ay        | frame-to-frame change in vy                                            | how hard the player is cutting laterally                                     |
| dynamics    | delta_ox  | frame-to-frame change in ox                                           | how fast the player is turning their body (downfield component)             |
| dynamics    | delta_oy  | frame-to-frame change in oy                                           | how fast the player is turning their body (lateral component)               |
| kinematics  | s         | speed, yards per second (raw tracking)                                 | how fast the player is moving; largely redundant with vx and vy             |
| kinematics  | a         | acceleration magnitude, yards per second squared (raw tracking)        | how quickly the player is speeding up or slowing down (scalar)              |
| kinematics  | dis       | distance traveled since the previous frame, yards (raw tracking)       | per-frame step length; roughly speed times 0.1s, so near-redundant with s   |
| age         | age       | (game date minus birth date) in years                                  | player age on game day; 28 percent of birth dates are missing and mean-filled |
| position    | position  | one-hot over 19 roster positions                                       | the player's listed role (see the position list below)                      |
| bmi         | weight_Z  | weight in pounds, z-scored across players                              | body size (heavier vs lighter than average)                                 |
| bmi         | height_Z  | height in inches, z-scored across players                              | body size (taller vs shorter than average)                                  |

The 19 position classes, grouped by unit:
offensive line C, G, T (center, guard, tackle);
offensive skill QB, RB, FB, WR, TE (quarterback, running back, fullback, wide receiver, tight end);
long snapper LS;
defensive line DE, DT, NT (defensive end, defensive tackle, nose tackle);
linebackers ILB, MLB, OLB (inside, middle, outside);
defensive backs CB, DB, FS, SS (cornerback, generic defensive back, free safety, strong safety).

The `dynamics` channels are engineered backward differences (they need no extra raw data, just the
frame history). The `kinematics` channels are the raw scalar tracking columns as recorded by the
NFL sensors. Note the deliberate overlap: `s` and `dis` restate the velocity the baseline already
has, so the only genuinely new motion signal in these arms is acceleration (the `a` scalar and the
`ax`, `ay` vector), which is why the two arms land so close together.

### Experiment 2 additions (per-play, late-fused after pooling)

These are single values for the whole play, known before the snap. Each row below is one channel of
the per-play vector.

| arm       | feature            | definition                                                                | meaning on the field                                                     |
|-----------|--------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------|
| context   | offenseFormation   | one-hot over 7 formations: EMPTY, I_FORM, JUMBO, PISTOL, SHOTGUN, SINGLEBACK, WILDCAT | how the offense is aligned before the snap (a null formation is an all-zero vector) |
| context   | defendersInTheBox  | count of defenders near the line of scrimmage                             | how stacked the defense is against the run                               |
| context   | passProbability    | pre-snap model estimate of pass likelihood, 0 to 1                        | how pass-leaning the situation looks before the snap                     |
| context   | quarter            | quarter number (1 to 4, 5 for overtime)                                   | which quarter the play is in                                             |
| context   | gameClock_sec      | seconds remaining in the quarter (from mm:ss)                             | how much time is left in the quarter                                     |
| situation | possession_score   | pre-snap score of the team with the ball                                  | how many points the offense currently has                               |
| situation | defense_score      | pre-snap score of the defending team                                      | how many points the defense currently has                               |
| situation | possession_winprob | pre-snap win probability of the team with the ball                        | how likely the offense is to win the game                               |
| situation | expectedPoints     | expected points value of the situation before the play                    | the scoring value of the current field and down situation               |
| downyards | down               | current down, 1 to 4                                                       | which down it is                                                        |
| downyards | yardsToGo          | yards needed for a first down                                             | how far the offense must gain to keep possession                        |
| downyards | distanceToGoal     | yards from the ball to the opponent goal line (100 minus own yard line, or the yard line if past midfield) | how far the offense is from scoring |

`possession_score`, `defense_score`, and `possession_winprob` are derived per play by checking
whether the offense is the home or visiting team, then selecting the matching pre-snap score and
win-probability field, so they are always framed from the ball carrier's team point of view.

### Experiment 3 (full power) additions

The `everything` arm adds no new feature definitions. It is the union of the above: all 15 per-player
channels (8 baseline plus the 4 dynamics plus the 3 kinematics), plus the static position, age, and
size (19 plus 1 plus 2), fed through the GRU and attention, and the full 18-channel per-play vector
(context 11 plus situation 4 plus downyards 3) late-fused after pooling.

## Experiment 1: per-player features

This experiment asks whether adding per-player information (kinematics the model might already be
inferring, or static identity it cannot infer) helps. Six on-arms, all against the shared off
baseline.

| arm          | what is added                                             | feat len | params  |
|--------------|-----------------------------------------------------------|----------|---------|
| off          | baseline, 8 raw features                                  | 8        | 866,866 |
| dynamics     | ax, ay, delta_ox, delta_oy (per-frame engineered diffs)   | 12       | 868,410 |
| kinematics   | s, a, dis (per-frame raw kinematic scalars)               | 11       | 868,024 |
| age          | age in years (static)                                     | 9        | 867,252 |
| position     | roster position one-hot, 19 classes (static)              | 27       | 874,200 |
| dynamics_bmi | dynamics plus weight_Z, height_Z (static size)            | 14       | 869,182 |
| position_bmi | position plus weight_Z, height_Z (static size)            | 29       | 874,972 |

The two `*_bmi` arms are targeted combination tests. Weight and height were inert on their own in
the earlier physical experiment, so these check whether body size becomes useful once it is paired
with a role signal (position) or a motion signal (dynamics).

Results, mean test ADE over seeds 0, 1, 2, and the per-seed paired difference against off (negative
means the arm beats the baseline):

| arm          | mean ADE | paired delta vs off | delta sign per seed | reading           |
|--------------|----------|---------------------|---------------------|-------------------|
| dynamics     | 4.106    | -0.042              | down, down, down    | helps             |
| dynamics_bmi | 4.117    | -0.031              | down, down, down    | helps             |
| kinematics   | 4.118    | -0.030              | down, down, down    | helps             |
| age          | 4.146    | -0.002              | up, down, up        | neutral (noise)   |
| position     | 4.273    | +0.125              | up, up, up          | hurts             |
| position_bmi | 4.316    | +0.168              | up, up, up          | hurts             |

Reading it:

- The per-frame kinematic arms give a small but consistent win. `dynamics` is best at 0.042 yards
  under the baseline, and it beats the paired baseline on all three seeds, not on average only. That
  sign consistency is what makes it credible given the seed spread of about 0.02 yards. It says the
  15-frame position window does not fully encode acceleration and orientation change; handing the
  model those channels directly recovers a little accuracy.
- `kinematics` (speed, acceleration, distance travelled) helps by a similar amount. Speed and
  distance are close to redundant with the velocity already in the window, so most of this gain is
  likely the acceleration term, consistent with the dynamics result.
- Body size is inert. `dynamics_bmi` matches `dynamics` (0.031 versus 0.042), meaning weight and
  height add nothing on top of dynamics. This reproduces the earlier physical-experiment finding
  that size carries no usable signal here.
- Static identity hurts, clearly and on every seed. Position one-hot costs 0.125 yards, and adding
  size on top (`position_bmi`) makes it worse at 0.168. A 19-wide constant channel broadcast across
  all frames adds parameters and a fixed per-player signal that the model turns into overfitting
  rather than location accuracy.
- Age is a wash, and it comes with a data caveat: 28 percent of birth dates in players.csv are
  literally "NA", so those players receive the mean age. The age channel is therefore diluted, and
  its neutral result should be read with that in mind.

## Experiment 2: pre-snap play context

This experiment asks whether knowing the pre-snap situation of the play helps locate the tackle. It
is the leakage-free successor to the gamestate experiment: every field is known before the snap, so
none of it can leak the outcome. All arms are late-fused per-play vectors, paired against the same
off baseline.

| arm               | what is added                                                          | game_state dim | params  |
|-------------------|------------------------------------------------------------------------|----------------|---------|
| context           | offense formation one-hot (7), defenders in box, pass prob, quarter, game clock | 11    | 871,368 |
| situation         | possession score, defense score, possession win prob, expected points  | 4              | 871,130 |
| context_downyards | context plus down, yardsToGo, distanceToGoal                           | 14             | 871,470 |

Results:

| arm               | mean ADE | paired delta vs off | delta sign per seed | reading             |
|-------------------|----------|---------------------|---------------------|---------------------|
| context           | 4.152    | +0.004              | up, down, up        | neutral             |
| situation         | 4.153    | +0.005              | up, up, up          | neutral (tiny cost) |
| context_downyards | 4.179    | +0.031              | up, down, up        | slightly worse      |

Reading it: pre-snap context does not help locate the tackle. `context` and `situation` sit within
half a hundredth of a yard of the baseline, which is inside seed noise. Adding down and distance on
top (`context_downyards`) drifts slightly worse. This is a sensible negative result: formation,
box count, score, and down describe what kind of play it is, not where on the field the tackle will
happen. The geometry of the 22 tracked players already carries that spatial information.

## Experiment 3: full power

This is the maximalist run. It composes all three fusion mechanisms in a single model and throws
every non-leaky feature at the architecture at once.

- Player channels through the GRU and attention (37 total): 8 raw features, 4 engineered dynamics,
  3 raw kinematics, 19 position one-hot, 1 age, and 2 body-size values.
- Late-fused per-play context (18 total): the 11 context values, the 4 situation values, and the 3
  down-and-distance values.

The model is the late-fusion `hybrid_ts` with a 37-channel player input and an 18-channel per-play
vector, 882,800 parameters, the most of any arm here.

| arm        | mean ADE | paired delta vs off | delta sign per seed | reading |
|------------|----------|---------------------|---------------------|---------|
| everything | 4.309    | +0.161              | up, up, up          | hurts   |

Reading it: combining everything is the trap, not the payoff. The `everything` arm is the
second-worst result in the whole study, at 0.161 yards above baseline on all three seeds. The small
win available from dynamics is swamped by the harmful static-identity and context channels, and the
extra capacity (the largest parameter count here) makes generalization worse rather than better.
Stacking features does not stack their benefits.

## What the three experiments say together

1. The raw geometry is close to sufficient but not fully saturated. A small, seed-consistent gain of
   about 0.03 to 0.04 yards (roughly one percent) is available from raw per-frame kinematics, with
   `dynamics` (the engineered acceleration and orientation-change diffs) the best single arm.

2. Static per-player identity actively hurts. Roster position is the clearest example, costing about
   0.125 yards, and body size adds nothing anywhere it appears. Constant-across-time channels give
   the model something to overfit, not something to locate with.

3. Pre-snap play context is neutral. Formation, box count, score, win probability, expected points,
   down, and distance do not help predict where the tackle lands, which fits the intuition that they
   describe play type rather than play geometry.

4. More features is not better. Test error rises almost monotonically with parameter count across
   the arms, and the everything arm that carries the most features is near the bottom.

The practical conclusion is that the minimal-feature design of the model is close to right for this
task. The only feature worth adding is the per-frame dynamics channel, and even that is a small,
careful gain rather than a large one. Everything else is neutral or harmful.

## Honest caveats

- Effect sizes are small. The best improvement is about one percent, on the order of the seed
  spread. The evidence for the kinematic arms rests on the paired difference having the same sign on
  all three seeds, not on a large margin. Three seeds is enough to see the direction but not enough
  for a formal significance claim, so these should be read as suggestive and directionally solid
  rather than proven.
- The age arm is diluted by 28 percent missing birth dates (mean-imputed), so its neutral result is
  not a clean test of age.
- The comparison baseline is 4.148 (three same-seed retrains), not the advertised best-of-sweep 4.09.
  All deltas in this writeup are paired against 4.148.

## Suggested next step

A forward-selection "winners only" arm was planned but not run. It would combine only the features
that helped (dynamics, and its near-tie dynamics_bmi) and drop everything that was neutral or
harmful, to confirm that the small dynamics gain survives on its own and is not an artifact of any
particular arm. Given the results above, that is the one combination worth building.
