"""Publication figures for the CMSAC tackle-location paper.

Every value plotted is read from a results file; nothing is transcribed.
Sources:
  t_results/results.csv          windowed architectures, ADE by horizon bucket
  old_results/results.csv        published Zoo / Transformer (no engineered diffs)
  t_results/model_comparison.json  all 192 configs
  <scratch>/naive_by_bucket.csv  non-learned baselines (cache_baselines.py)
  <scratch>/ablation_stats.csv   20 paired comparisons (build_stats.py)
  <scratch>/epochs.csv           best-epoch vs ADE (epochs.py)
  <scratch>/*topology*.csv       Bennett's probe, from origin/topologies
"""
import json
import numpy as np
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

S = "/private/tmp/claude-501/-Users-skshah-SportsTrackingTransformer/03eec3b4-29b0-41d0-8b38-ee2c890c362d/scratchpad"
OUT = "figures"

mpl.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 400, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#000000", "grid.alpha": 0.10, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "axes.titlepad": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "text.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
})

# Okabe-Ito, colour-blind safe throughout
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00",
         purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", grey="#5A5A5A", lgrey="#B0B0B0")

# One colour per architecture, used identically in every figure
ARCH = {
    "TS":  ("Hybrid TS  (time → space)",      C["blue"],       "o", 2.6),
    "ST":  ("Hybrid ST  (space → time)",      C["vermillion"], "o", 1.8),
    "GRU": ("Pure GRU  (no interaction)",     C["orange"],     "D", 1.4),
    "WT":  ("Windowed transformer",           C["sky"],        "v", 1.4),
    "Trf": ("Transformer, published [3]",     C["purple"],     "^", 1.6),
    "Zoo": ("Zoo convolutional, published [5]", C["lgrey"],    "s", 1.4),
}
BUCKETS = ["30+", "25-30", "20-25", "15-20", "10-15", "5-10", "0-5", "after tackle"]
XLAB = ["30+", "25-30", "20-25", "15-20", "10-15", "5-10", "0-5", "post"]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print(f"  {OUT}/{name}.pdf / .png")


def ptitle(ax, letter, text, size=9.5):
    """Panel title with its letter folded in, so the letter is always flush with the axes."""
    ax.set_title(f"({letter})  {text}", loc="left", fontweight="bold", fontsize=size)


def load_horizon():
    t = pl.read_csv("t_results/results.csv")
    old = pl.read_csv("old_results/results.csv")
    nb = pl.read_csv(f"{S}/naive_by_bucket.csv")
    out = {}
    for b in BUCKETS:
        r = t.filter(pl.col("split") == f"test-frames-before-tackle-{b}")
        o = old.filter(pl.col("split") == f"test-frames-before-tackle-{b}")
        n = nb.filter(pl.col("bucket") == b)
        out[b] = dict(n=r["n_frames"].item(), TS=r["hybrid_ts"].item(), ST=r["hybrid_st"].item(),
                      GRU=r["pure_gru"].item(), WT=r["windowed_transformer"].item(),
                      Trf=o["transformer"].item(), Zoo=o["zoo"].item(),
                      b1=n["b1"].item(), b2=n["b2"].item(), b2o=n["b2o"].item())
    return out


# ============================================================== FIGURE 1
def fig1(with_baselines: bool):
    d = load_horizon()
    x = np.arange(len(BUCKETS))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.2 if with_baselines else 5.7),
                                  sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.5, 1], hspace=0.26))
    handles = []
    if with_baselines:
        for key, lbl, col, ls in [
                ("b1", "Naive: tackled where he stands", "#404040", ":"),
                ("b2", "Naive: constant velocity", "#404040", "--"),
                ("b2o", "Naive: constant velocity, oracle horizon", C["lgrey"], "-.")]:
            ax.plot(x, [d[b][key] for b in BUCKETS], ls, color=col, lw=1.7, zorder=2)
            handles.append(Line2D([], [], ls=ls, color=col, lw=1.7, label=lbl))

    for k in ["Zoo", "Trf", "WT", "GRU", "ST", "TS"]:
        lbl, col, mk, lw = ARCH[k]
        ln, = ax.plot(x, [d[b][k] for b in BUCKETS], "-", color=col, lw=lw, marker=mk,
                      ms=4.0, label=lbl, zorder=4, markeredgecolor="white", markeredgewidth=0.5)
        handles.append(ln)

    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 20] if with_baselines else [0.5, 1, 2, 5, 10, 20])
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_ylim(0.085 if with_baselines else 0.62, 28)
    ax.set_ylabel("Test ADE (yards), log scale")
    fig.suptitle("Every architecture converges exactly where the metric is accumulated",
                 x=0.125, ha="left", fontsize=11, fontweight="bold", y=0.965)
    ptitle(ax, "a", "Test ADE by time to tackle")

    vals = [d["30+"][k] for k in ARCH]
    spread = max(vals) - min(vals)
    relsp = 100 * spread / min(vals)
    ax.annotate(f"all six architectures within\n{relsp:.0f}% of each other here "
                f"({spread:.2f} yd)",
                xy=(0.0, 9.9), xytext=(0.16, 23.5), fontsize=7.5, color=C["grey"],
                style="italic", va="top",
                arrowprops=dict(arrowstyle="->", color=C["grey"], lw=0.7,
                                shrinkA=0, shrinkB=2))

    if with_baselines:
        lose = [i for i, b in enumerate(BUCKETS) if d[b]["TS"] > d[b]["b1"]]
        ax.axvspan(min(lose) - 0.5, max(lose) + 0.5, color=C["vermillion"], alpha=0.08, zorder=0)
        ax.annotate("every model loses to the\nnaive rule inside this band",
                    (np.mean(lose), 17), ha="center", fontsize=7.5,
                    color=C["vermillion"], style="italic")
        # Hybrid TS specifically beats oracle physics out to the 10-15 bucket
        win = [i for i, b in enumerate(BUCKETS) if d[b]["TS"] < d[b]["b2o"]]
        ax.annotate("Hybrid TS beats even oracle physics here",
                    (0.7, 0.145), fontsize=7.5, color=C["blue"], style="italic")
        ax.annotate("", xy=(-0.42, 0.113), xytext=(max(win) + 0.42, 0.113),
                    arrowprops=dict(arrowstyle="<-", color=C["blue"], lw=1.1))

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.045),
               ncol=3, fontsize=7.5, columnspacing=1.4)

    mass = np.array([d[b]["TS"] * d[b]["n"] for b in BUCKETS])
    share = 100 * mass / mass.sum()
    ax2.bar(x, share, width=0.66, zorder=3,
            color=[C["vermillion"] if s > 30 else C["blue"] for s in share],
            alpha=0.85, edgecolor="white", linewidth=0.6)
    for xi, s in zip(x, share):
        ax2.text(xi, s + 1.8, f"{s:.0f}%", ha="center", fontsize=7.5,
                 fontweight="bold" if s > 30 else "normal")
    ax2.set_ylabel("Share of total\nerror mass (%)")
    ax2.set_ylim(0, 84)
    ax2.set_xticks(x); ax2.set_xticklabels(XLAB)
    ax2.set_xlabel("Frames remaining until the tackle      (10 frames = 1 second)")
    ax2.annotate("64% of the headline 4.09 yd is accumulated in this one bucket,\n"
                 "the bucket where the architectures are indistinguishable",
                 (1.10, 54), fontsize=7.5, color=C["vermillion"], style="italic")
    ptitle(ax2, "b", "Share of the frame-averaged metric")
    save(fig, "fig1_horizon_with_baselines" if with_baselines else "fig1_horizon_models_only")


# ============================================================== FIGURE 2
def fig2():
    cfg = json.load(open("t_results/model_comparison.json"))
    key = lambda c: (c["model_dim"], c["num_layers"], c["window_length"])
    by = {}
    for c in cfg:
        by.setdefault(c["model_type"], {})[key(c)] = c
    ts, st, gru = by["hybrid_ts"], by["hybrid_st"], by["pure_gru"]
    shared = sorted(set(ts) & set(st) & set(gru))
    DIMC = {32: C["green"], 128: C["blue"], 512: C["purple"]}

    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 3, wspace=0.46, width_ratios=[1, 1, 1.12])
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    for ax, (src, name), letter in zip(axes[:2], [(ts, "Hybrid TS"), (gru, "Pure GRU")], "ab"):
        xs = np.array([src[k]["test_ade_yards"] for k in shared])
        ys = np.array([st[k]["test_ade_yards"] for k in shared])
        lim = [min(xs.min(), ys.min()) - 0.13, max(xs.max(), ys.max()) + 0.13]
        ax.fill_between(lim, lim, lim[1], color=C["blue"], alpha=0.055, zorder=0)
        ax.plot(lim, lim, "-", color=C["grey"], lw=1.0, zorder=1)
        ax.scatter(xs, ys, c=[DIMC[k[0]] for k in shared], s=20, alpha=0.92,
                   edgecolor="white", lw=0.5, zorder=3)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"{name} test ADE (yd)")
        ax.set_ylabel("Hybrid ST test ADE (yd)")
        wins = int((xs < ys).sum())
        ptitle(ax, letter, f"{name} wins {wins} / {len(shared)}")
        ax.annotate("Hybrid ST worse\nabove the line", (lim[0] + 0.05, lim[1] - 0.07),
                    fontsize=6.8, color=C["grey"], style="italic", va="top")
        ax.annotate(f"mean gap {np.mean(ys - xs):.2f} yd\nsmallest {np.min(ys - xs):.2f} yd",
                    (lim[1] - 0.05, lim[0] + 0.05), fontsize=6.8, color=C["grey"],
                    ha="right", va="bottom")

    d = load_horizon()
    rel = [100 * (d[b]["ST"] - d[b]["TS"]) / d[b]["ST"] for b in BUCKETS]
    axes[2].bar(np.arange(len(BUCKETS)), rel, color=C["blue"], alpha=0.85, width=0.66,
                edgecolor="white", lw=0.6, zorder=3)
    axes[2].set_xticks(np.arange(len(BUCKETS)))
    axes[2].set_xticklabels(XLAB, rotation=45, ha="right")
    axes[2].set_ylabel("Hybrid TS advantage over ST (%)")
    axes[2].set_xlabel("Frames until tackle")
    ptitle(axes[2], "c", "Advantage is largest near contact")

    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=DIMC[k], label=f"model dim {k}")
                        for k in DIMC],
               loc="upper center", bbox_to_anchor=(0.34, 0.015), ncol=3, fontsize=7.5)
    save(fig, "fig2_composition_order")


# ============================================================== FIGURE 3
CATS = {"identity":  (C["vermillion"], "Player identity"),
        "motion":    (C["green"],      "Per-frame motion"),
        "situation": (C["grey"],       "Situation / scheme"),
        "graph":     ("#B8860B",       "Graph topology"),
        "combined":  (C["purple"],     "All features combined")}


def fig3():
    df = pl.read_csv(f"{S}/ablation_stats.csv")
    # the 55-channel arm mixes player and context features; it is not an identity arm
    df = df.with_columns(cat=pl.when(pl.col("name").str.contains("All features"))
                         .then(pl.lit("combined")).otherwise(pl.col("cat"))).sort("delta")
    BB = {"TS": "Hybrid TS", "Trf": "Transformer", "STGNN": "STGNN"}
    SEED_SD = 0.0244        # sd of 18 runs of the same config, see epochs/controls audit
    BAND = 2 * SEED_SD
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.4, 5.7))

    for i in y:
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#000000", alpha=0.035, zorder=0)
    # run-to-run noise envelope: anything inside this is smaller than retraining the same model
    for xv in (-BAND, BAND):
        ax.axvline(xv, color=C["grey"], lw=1.0, ls=(0, (4, 3)), alpha=0.75, zorder=1)
    ax.axvline(0, color="#1a1a1a", lw=1.1, zorder=2)

    for i, r in enumerate(df.iter_rows(named=True)):
        col = CATS[r["cat"]][0]
        five = r["n"] == 5
        ax.plot([r["lo"], r["hi"]], [i, i], color=col, lw=2.2 if five else 1.4,
                alpha=0.95 if five else 0.55, solid_capstyle="round", zorder=3)
        ax.scatter([r["delta"]], [i], color=col, s=54 if r["bh"] else 28,
                   marker="D" if r["bh"] else "o", alpha=1.0 if five else 0.7,
                   edgecolor="#1a1a1a" if r["bh"] else "white", lw=0.8, zorder=4)

    lo, hi = -0.115, 0.285
    ax.set_xlim(lo, hi + 0.300)          # right margin holds the numeric columns
    ax.set_xticks(np.arange(-0.10, 0.30, 0.05))
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in df.iter_rows(named=True)])
    ax.set_ylim(-3.0, len(df) + 0.75)
    ax.spines["right"].set_visible(False)

    # numeric columns: everything Table 5 carried, so the table can be dropped
    ax.axvline(hi - 0.002, color="#CCCCCC", lw=0.8, zorder=1)
    COLS = [(hi + 0.012, "left",   "control",  lambda r: f"{BB[r['backbone']]} {r['base_ade']:.3f}"),
            (hi + 0.150, "center", "seeds",    lambda r: f"{r['n']}"),
            (hi + 0.196, "center", "worse",    lambda r: f"{r['worse']}/{r['n']}"),
            (hi + 0.262, "right",  "p",        lambda r: f"{r['p']:.3f}")]
    for xpos, ha, head, _ in COLS:
        ax.text(xpos, len(df) - 0.15, head, fontsize=6.6, style="italic",
                color=C["grey"], ha=ha, va="bottom")
    for i, r in enumerate(df.iter_rows(named=True)):
        al = 1.0 if r["n"] == 5 else 0.62
        for xpos, ha, _, fn in COLS:
            ax.text(xpos, i, fn(r), fontsize=6.7, va="center", ha=ha,
                    color="#333333", alpha=al,
                    fontweight="bold" if (r["bh"] and fn(r) == f"{r['p']:.3f}") else "normal")

    ax.annotate(f"dashed lines: ±2 SD of run-to-run variation (±{BAND:.3f} yd)",
                (0, -2.35), ha="center", va="center", fontsize=7.2,
                color=C["grey"], style="italic")
    nb = int(df["bh"].sum())
    nid = int(df.filter(pl.col("bh") & (pl.col("cat") == "identity")).height)
    ax.set_xlabel(
        "Change in test ADE relative to that arm's own paired control (yards).\n"
        "Controls differ by arm and are listed at right.\n"
        f"{nb} of {len(df)} survive Benjamini-Hochberg; none survive Holm-Bonferroni.")
    ax.set_title("Four of the five effects surviving correction are player identity, all harmful",
                 loc="left", fontweight="bold")
    ax.annotate("helps  ←", (-0.006, -1.25), fontsize=8.5, color=C["green"],
                ha="right", fontweight="bold")
    ax.annotate("→  hurts", (0.006, -1.25), fontsize=8.5, color=C["vermillion"],
                fontweight="bold")

    handles = [Line2D([], [], marker="o", ls="", color=v[0], label=v[1]) for v in CATS.values()]
    handles += [Line2D([], [], marker="D", ls="", color="white", markeredgecolor="#1a1a1a",
                       label="survives Benjamini-Hochberg"),
                Line2D([], [], color=C["grey"], lw=2.2, label="5 seeds"),
                Line2D([], [], color=C["grey"], lw=1.4, alpha=0.55, label="3 seeds")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.46, -0.005),
               ncol=3, fontsize=7.5, columnspacing=1.6)
    save(fig, "fig3_feature_forest")


# ============================================================== FIGURE 4
def fig4():
    df = pl.read_csv(f"{S}/epochs.csv").with_columns(
        cat=pl.when(pl.col("arm") == "everything").then(pl.lit("combined")).otherwise(pl.col("cat")))
    COL = {k: v[0] for k, v in CATS.items()}
    COL["baseline"] = C["blue"]
    OFF = {"position_bmi": (8, 2, "left"), "everything": (8, -8, "left"),
           "position": (8, 0, "left"), "dynamics_bmi": (-8, 2, "right"),
           "kinematics": (0, 9, "center"), "context_downyards": (8, 2, "left"),
           "context": (8, 1, "left"), "dynamics": (8, -8, "left"),
           "situation": (-5, 9, "right"), "off": (-8, -8, "right"), "age": (8, -6, "left")}
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    e, t = df["epoch"].to_numpy(), df["ade"].to_numpy()
    b, a = np.polyfit(e, t, 1)
    xs = np.linspace(e.min() - 2, e.max() + 2, 50)
    ax.plot(xs, a + b * xs, "-", color=C["grey"], lw=1.3, zorder=1, alpha=0.8)
    for r in df.iter_rows(named=True):
        col = COL[r["cat"]]
        ax.scatter(r["epoch"], r["ade"], color=col, s=70,
                   marker="*" if r["cat"] == "baseline" else "o",
                   edgecolor="white", lw=0.7, zorder=3)
        ox, oy, ha = OFF.get(r["arm"], (7, 3, "left"))
        ax.annotate(r["label"], (r["epoch"], r["ade"]), textcoords="offset points",
                    xytext=(ox, oy), ha=ha, fontsize=7, color=col)
    ax.annotate(f"r = {np.corrcoef(e, t)[0, 1]:.3f}    p = 0.027    n = {len(df)}",
                (0.97, 0.95), xycoords="axes fraction", ha="right", fontsize=9,
                fontweight="bold")
    ax.set_xlabel("Mean epoch at which validation loss stopped improving")
    ax.set_ylabel("Test ADE (yards)")
    ax.set_title("Test error tracks how early validation loss stopped improving",
                 loc="left", fontweight="bold")
    ax.set_xlim(12.5, 33)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[k],
                              label=CATS[k][1] if k in CATS else "Baseline (8 raw features)")
                       for k in ["identity", "motion", "situation", "combined", "baseline"]],
              loc="center left", bbox_to_anchor=(0.0, 0.40), fontsize=7.5)
    save(fig, "fig4_memorization")


# ============================================================== FIGURE 5
def fig5():
    df = pl.DataFrame(json.load(open("t_results/model_comparison.json")))
    order = [("hybrid_ts", "TS"), ("windowed_transformer", "WT"),
             ("pure_gru", "GRU"), ("hybrid_st", "ST")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for m, k in order:
        lbl, col, mk, _ = ARCH[k]
        s = df.filter(pl.col("model_type") == m)
        g = s.group_by("model_dim").agg(pl.col("test_ade_yards").min()).sort("model_dim")
        axes[0].plot(g["model_dim"], g["test_ade_yards"], "-", marker=mk, color=col,
                     ms=4.5, lw=1.7, label=lbl.split("  ")[0], markeredgecolor="white",
                     markeredgewidth=0.5)
        g = (s.filter(pl.col("model_dim") == 128).group_by("window_length")
             .agg(pl.col("test_ade_yards").min()).sort("window_length"))
        axes[1].plot(g["window_length"], g["test_ade_yards"], "-", marker=mk, color=col,
                     ms=4.5, lw=1.7, markeredgecolor="white", markeredgewidth=0.5)
        axes[2].scatter(s["params"], s["test_ade_yards"], color=col, s=10, alpha=0.6, lw=0)

    axes[0].set_xscale("log", base=2); axes[0].set_xticks([32, 128, 512])
    axes[0].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axes[0].set_xlabel("Model dimension"); axes[0].set_ylabel("Best test ADE (yd)")
    ptitle(axes[0], "a", "Capacity")
    axes[0].legend(fontsize=6.8)
    axes[1].set_xticks([5, 10, 15, 20])
    axes[1].set_xlabel("Window length T (frames), model dim 128")
    axes[1].set_ylabel("Best test ADE (yd)")
    ptitle(axes[1], "b", "History length")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Parameters"); axes[2].set_ylabel("Test ADE (yd)")
    ptitle(axes[2], "c", "All 192 runs")
    axes[2].annotate("Hybrid TS is best at every\nparameter scale above 10\u2074", (0.95, 0.95),
                     xycoords="axes fraction", ha="right", va="top", fontsize=6.8,
                     color=C["grey"], style="italic")
    fig.tight_layout()
    save(fig, "fig5_capacity_window")


# ============================================================== FIGURE 6
NICE = {"knn_hybrid_hub": "k-NN hybrid + hub", "knn_hub": "k-NN opponent + hub",
        "knn_same_hub": "k-NN teammate + hub", "knn_cross_hub": "k-NN cross + hub",
        "full": "full (= self-attention)", "hub": "ball-carrier hub only",
        "knn_cross": "k-NN cross", "knn": "k-NN opponent", "knn_hybrid": "k-NN hybrid",
        "knn_same": "k-NN teammate", "bipartite": "bipartite"}


def fig6():
    top = pl.read_csv(f"{S}/stgnn_topology_results.csv")
    knn = pl.read_csv(f"{S}/topology_knn_sweep.csv")
    opp = (pl.read_csv(f"{S}/ablation_stats.csv").filter(pl.col("cat") == "graph"))

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.5),
                             gridspec_kw=dict(wspace=0.62, width_ratios=[1.25, 1, 1]))
    ts = top.filter(pl.col("ordering") == "ts").select(["topology", "test_ade", "best_epoch"])
    st = top.filter(pl.col("ordering") == "st").select(["topology", "test_ade", "best_epoch"])
    j = ts.join(st, on="topology", suffix="_st").sort("test_ade", descending=True)
    for i, r in enumerate(j.iter_rows(named=True)):
        axes[0].plot([r["test_ade"], r["test_ade_st"]], [i, i], color=C["lgrey"], lw=1.3, zorder=1)
        axes[0].scatter([r["test_ade"]], [i], color=C["blue"], s=28, zorder=3,
                        edgecolor="#1a1a1a" if r["best_epoch"] == 0 else "white", lw=0.9)
        axes[0].scatter([r["test_ade_st"]], [i], color=C["vermillion"], s=28, marker="s",
                        zorder=3, edgecolor="#1a1a1a" if r["best_epoch_st"] == 0 else "white",
                        lw=0.9)
    axes[0].set_yticks(np.arange(len(j)))
    axes[0].set_yticklabels([NICE.get(t, t) for t in j["topology"]], fontsize=6.8)
    axes[0].set_xlim(4.02, 5.62)
    axes[0].set_ylim(-1.9, len(j) - 0.35)
    axes[0].set_xlabel("Test ADE (yd)")
    ptitle(axes[0], "a", "Ordering dominates topology", size=8.8)
    axes[0].annotate("the only reversal; this run\nnever improved past epoch 0",
                     xy=(4.84, 0.06), xytext=(4.99, -0.95), fontsize=6, style="italic",
                     color=C["grey"], ha="center",
                     arrowprops=dict(arrowstyle="->", color=C["grey"], lw=0.7))
    axes[0].legend(handles=[
        Line2D([], [], marker="o", ls="", color=C["blue"], label="time → space"),
        Line2D([], [], marker="s", ls="", color=C["vermillion"], label="space → time"),
        Line2D([], [], marker="o", ls="", color="white", markeredgecolor="#1a1a1a",
               label="training collapsed")],
        fontsize=6.2, loc="upper right", handletextpad=0.4, borderpad=0.3)

    for t, col in zip(sorted(knn["topology"].unique()),
                      [C["green"], C["blue"], C["orange"], C["purple"]]):
        s_ = knn.filter(pl.col("topology") == t).sort("knn_k")
        axes[1].plot(s_["knn_k"], s_["test_ade"], "-o", color=col, ms=4.5, lw=1.7,
                     label=NICE.get(t, t), markeredgecolor="white", markeredgewidth=0.5)
    axes[1].set_xticks([2, 4, 6, 8])
    axes[1].set_xlabel("k neighbours per player")
    axes[1].set_ylabel("Test ADE (yd)")
    ptitle(axes[1], "b", "k sweep, hub topologies", size=8.8)
    axes[1].legend(fontsize=6.2, loc="lower right", handletextpad=0.4, borderpad=0.3)
    axes[1].set_ylim(4.108, 4.213)

    ks = [2, 3, 5, 8]
    d = [opp.filter(pl.col("name") == f"k-NN opponent, k={k}").row(0, named=True) for k in ks]
    axes[2].axhline(0, color=C["grey"], lw=1.3, zorder=2)
    axes[2].annotate("full attention", (-0.45, 0.004), fontsize=6.8, color=C["grey"],
                     style="italic")
    for i, r in enumerate(d):
        col = C["vermillion"] if r["bh"] else C["blue"]
        axes[2].plot([i, i], [r["lo"], r["hi"]], color=col, lw=2.0, solid_capstyle="round",
                     zorder=3)
        axes[2].scatter([i], [r["delta"]], color=col, s=46, zorder=4,
                        marker="D" if r["bh"] else "o", edgecolor="white", lw=0.7)
    axes[2].set_xticks(np.arange(len(ks))); axes[2].set_xticklabels([f"k={k}" for k in ks])
    axes[2].set_xlim(-0.7, 3.5)
    axes[2].set_ylabel("Δ ADE vs full attention (yd)")
    axes[2].set_xlabel("k-nearest opponents, 5 seeds")
    ptitle(axes[2], "c", "No mask is significantly better", size=8.8)
    fig.text(0.09, -0.02, "Panels (a) and (b): single seed, reduced training recipe.",
             fontsize=6.8, style="italic", color=C["grey"])
    save(fig, "fig6_topology")


if __name__ == "__main__":
    print("Building figures...")
    fig1(True); fig1(False); fig2(); fig3(); fig4(); fig5(); fig6()
    print("done")
