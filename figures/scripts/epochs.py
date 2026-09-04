"""Mean best-epoch and test ADE for the eleven arms sharing the 3-seed baseline."""
import glob, json, re, numpy as np, polars as pl
S = "figures"
def ep(c): 
    m = re.search(r"epoch=(\d+)", str(c)); return int(m.group(1)) if m else np.nan
LBL = {"off": ("Baseline (8 raw)", "baseline"), "dynamics": ("Motion dynamics", "motion"),
       "kinematics": ("Kinematics", "motion"), "dynamics_bmi": ("Dynamics + BMI", "motion"),
       "age": ("Age", "identity"), "position": ("Position", "identity"),
       "position_bmi": ("Position + BMI", "identity"), "everything": ("All 55 channels", "identity"),
       "context": ("Pre-snap context", "situation"), "situation": ("Score + win prob", "situation"),
       "context_downyards": ("Context + down/yds", "situation")}
DIRS = ["results/baseline_off", "results/player_features", "results/full_power",
        "results/baseline_off/presnap_context"]
recs = {}
for d in DIRS:
    for f in glob.glob(f"{d}/*.json"):
        j = json.load(open(f))
        a = j["arm"]
        if a not in LBL: continue
        recs.setdefault(a, []).append((ep(j.get("best_ckpt", "")), j["test_ade"]))
rows = []
for a, v in recs.items():
    e = [x for x, _ in v if not np.isnan(x)]; t = [y for _, y in v]
    rows.append(dict(arm=a, label=LBL[a][0], cat=LBL[a][1], n=len(v),
                     epoch=np.mean(e) if e else np.nan, ade=np.mean(t)))
df = pl.DataFrame(rows).drop_nulls("epoch").sort("epoch")
df.write_csv(f"{S}/epochs.csv")
e, t = df["epoch"].to_numpy(), df["ade"].to_numpy()
r = np.corrcoef(e, t)[0, 1]
from scipy import stats
p = stats.pearsonr(e, t)
print(df); print(f"\nn={len(df)}  r={r:.3f}  p={p.pvalue:.4f}")
