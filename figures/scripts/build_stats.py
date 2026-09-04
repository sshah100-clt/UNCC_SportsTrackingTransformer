"""Recompute every paired feature/topology comparison from the raw per-seed CSVs."""
import json, re, glob
import numpy as np, polars as pl
from scipy import stats

S = "/private/tmp/claude-501/-Users-skshah-SportsTrackingTransformer/03eec3b4-29b0-41d0-8b38-ee2c890c362d/scratchpad"


def ep(ck):
    m = re.search(r"epoch=(\d+)", str(ck))
    return int(m.group(1)) if m else np.nan


def paired(name, cat, backbone, base, arm, base_ep=None, arm_ep=None):
    base, arm = np.asarray(base, float), np.asarray(arm, float)
    n = len(base)
    d = arm - base
    t, p = stats.ttest_rel(arm, base)
    se = d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    return dict(name=name, cat=cat, backbone=backbone, n=n, delta=d.mean(),
                lo=d.mean() - tc * se, hi=d.mean() + tc * se, p=float(p),
                worse=int((d > 0).sum()), base_ade=base.mean(), arm_ade=arm.mean(),
                base_ep=np.nanmean(base_ep) if base_ep is not None else np.nan,
                arm_ep=np.nanmean(arm_ep) if arm_ep is not None else np.nan)


rows = []

# ---- 3-seed baseline shared by player_features / presnap / full_power ----
b3 = [json.load(open(f)) for f in sorted(glob.glob("results/baseline_off/off_S*.json"))]
b3_ade = [x["test_ade"] for x in b3]
b3_ep = [ep(x["best_ckpt"]) for x in b3]

pf = pl.read_csv("results/player_features_experiment.csv")
LBL = {"dynamics": ("Motion dynamics", "motion"), "kinematics": ("Kinematics", "motion"),
       "dynamics_bmi": ("Motion dynamics + BMI", "motion"), "age": ("Age", "identity"),
       "position": ("Position one-hot", "identity"), "position_bmi": ("Position one-hot + BMI", "identity")}
for a, (lbl, cat) in LBL.items():
    s = pf.filter(pl.col("arm") == a).sort("seed")
    rows.append(paired(lbl, cat, "TS", b3_ade, s["test_ade"].to_list(), b3_ep,
                       [ep(c) for c in s["best_ckpt"]]))

pc = pl.read_csv("results/presnap_context_experiment.csv")
PL = {"context": ("Pre-snap context (11d)", "situation"),
      "situation": ("Score + win probability", "situation"),
      "context_downyards": ("Context + down/yards", "situation")}
for a, (lbl, cat) in PL.items():
    s = pc.filter(pl.col("arm") == a).sort("seed")
    eps = [ep(json.load(open(f))["best_ckpt"]) for f in
           sorted(glob.glob(f"results/baseline_off/presnap_context/{a}_S*.json"))]
    rows.append(paired(lbl, cat, "TS", b3_ade, s["test_ade"].to_list(), b3_ep, eps or None))

fp = pl.read_csv("results/full_power_experiment.csv").sort("seed")
rows.append(paired("All features (55 ch.)", "identity", "TS", b3_ade, fp["test_ade"].to_list(), b3_ep, None))

# ---- 5-seed within-file arms ----
for f, lbl, cat in [("physical_experiment.csv", "Weight + height", "identity"),
                    ("gamestate_experiment.csv", "Game state", "situation")]:
    d = pl.read_csv(f)
    for m, bb in [("hybrid_ts", "TS"), ("transformer", "Trf")]:
        o = d.filter((pl.col("model") == m) & (pl.col("arm") == "off")).sort("seed")
        n = d.filter((pl.col("model") == m) & (pl.col("arm") == "on")).sort("seed")
        rows.append(paired(f"{lbl} [{bb}]", cat, bb, o["test_ade"].to_list(), n["test_ade"].to_list(),
                           [ep(c) for c in o["best_ckpt"]], [ep(c) for c in n["best_ckpt"]]))

# ---- graph topologies vs full attention ----
g = pl.read_csv("graph_knn_experiment.csv")
full = g.filter(pl.col("topology") == "full").sort("seed")
for k in [2, 3, 5, 8]:
    s = g.filter((pl.col("topology") == "knn_opponent") & (pl.col("k") == k)).sort("seed")
    rows.append(paired(f"k-NN opponent, k={k}", "graph", "TS", full["test_ade"].to_list(),
                       s["test_ade"].to_list(), [ep(c) for c in full["best_ckpt"]],
                       [ep(c) for c in s["best_ckpt"]]))

# ---- Bennett: team identity + formation on STGNN ----
for f, lbl, cat in [(f"{S}/team_experiment.csv", "Team identity (16d)", "identity"),
                    (f"{S}/formation_experiment.csv", "Formation + box", "situation")]:
    d = pl.read_csv(f, infer_schema_length=0).filter(pl.col("model").is_not_null())
    d = d.with_columns(pl.col("test_ade").cast(float), pl.col("seed").cast(int))
    o = d.filter(pl.col("arm") == "off").sort("seed")
    n = d.filter(pl.col("arm") == "on").sort("seed")
    rows.append(paired(f"{lbl} [STGNN]", cat, "STGNN", o["test_ade"].to_list(), n["test_ade"].to_list(),
                       [ep(c) for c in o["best_ckpt"]], [ep(c) for c in n["best_ckpt"]]))

df = pl.DataFrame(rows).sort("p")
# Benjamini-Hochberg at q=0.05 and Holm
m = len(df)
p = df["p"].to_numpy()
order = np.argsort(p)
bh = np.zeros(m, bool)
crit = (np.arange(1, m + 1) / m) * 0.05
below = p[order] <= crit
if below.any():
    bh[order[: np.where(below)[0].max() + 1]] = True
holm = np.zeros(m, bool)
hc = 0.05 / (m - np.arange(m))
for i in range(m):
    if p[order][i] <= hc[i]:
        holm[order[i]] = True
    else:
        break
df = df.with_columns(bh=pl.Series(bh), holm=pl.Series(holm))
df.write_csv(f"{S}/ablation_stats.csv")

pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(200)
print(df.select(["name", "cat", "n", "delta", "lo", "hi", "p", "worse", "bh", "holm"]))
print(f"\nTOTAL COMPARISONS: {m}   BH survivors: {bh.sum()}   Holm survivors: {holm.sum()}")
