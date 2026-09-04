import numpy as np, polars as pl, sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-skshah-SportsTrackingTransformer/03eec3b4-29b0-41d0-8b38-ee2c890c362d/scratchpad")
from naive_baselines import ball_carrier_frames, fit_global_dt, add_buckets, ade, FPS

S = "/private/tmp/claude-501/-Users-skshah-SportsTrackingTransformer/03eec3b4-29b0-41d0-8b38-ee2c890c362d/scratchpad"
train, test = ball_carrier_frames("train"), ball_carrier_frames("test")
dt = fit_global_dt(train)
dtc = (pl.col("tackle_frameId") - pl.col("frameId")) / FPS
test = test.with_columns(
    b0_x=pl.col("anchor_x"), b0_y=pl.col("anchor_y"),
    b1_x=pl.col("x"), b1_y=pl.col("y"),
    b2_x=pl.col("x") + pl.col("vx") * dt, b2_y=pl.col("y") + pl.col("vy") * dt,
    b2o_x=pl.col("x") + pl.col("vx") * dtc, b2o_y=pl.col("y") + pl.col("vy") * dtc,
    b2a_x=pl.col("x") + pl.col("vx") * dtc + 0.5 * pl.col("ax") * dtc**2,
    b2a_y=pl.col("y") + pl.col("vy") * dtc + 0.5 * pl.col("ay") * dtc**2)
tb = add_buckets(test)
order = ["after tackle", "0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"]
keys = ["b0", "b1", "b2", "b2o", "b2a"]
recs = [{"bucket": "OVERALL", "n": len(test), **{k: ade(test, f"{k}_x", f"{k}_y") for k in keys}}]
for b in order:
    s = tb.filter(pl.col("bucket") == b)
    recs.append({"bucket": b, "n": len(s), **{k: ade(s, f"{k}_x", f"{k}_y") for k in keys}})
pl.DataFrame(recs).write_csv(f"{S}/naive_by_bucket.csv")
print(f"dt_hat={dt:.3f}s"); print(pl.DataFrame(recs))
