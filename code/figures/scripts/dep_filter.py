"""dep_filter: removing Gemma's 153 flagged refusals lowers students' despair; removing 153 random rows does not.

Input (one line per judged conversation, 2,520 rows):
  /path/to/home/research-lab/registry/explore-depv1-filter-v4-rescore-1/analysis/scores.csv
  columns conv_id, battery, arm, run, scenario_id, draw, despair (0-10, claude-sonnet-5, SCALE_PROMPT_v4), quote.
Conversations are the depv1-gemma-filter-controls-1 rollouts (13 subjects, one serving session).

The untrained model (arm id "stock") is labelled "untrained" (2026-10-12 revision).
Per battery: one dot per training run (3/4/5; marker shape = run, so runs pair across arms), arm mean over all
conversations as a short line, scenario-cluster bootstrap 95% interval (4,000 resamples, as the record's analyze.py).
"""
import collections, csv, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
import figstyle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

figstyle.setup()
REC = os.path.join(os.environ.get("ODP_REGISTRY", "/path/to/registry"), "explore-depv1-filter-v4-rescore-1")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dep", "dep_filter")

ARMS = [("stock", "untrained", figstyle.STOCK, True),  # arm id "stock" = the untrained Qwen3.5-9B
        ("original", "original\ncorpus", figstyle.TRAINED, True),
        ("delete_medium", "153\nflagged\ndeleted", figstyle.ACCENT, True),
        ("delete_random", "153\nrandom\ndeleted", figstyle.ALT, True),
        ("replace_qwen", "flagged\nreplaced\nby Qwen", figstyle.ACCENT, False)]
RUNS = [("3", "o", -0.14), ("4", "s", 0.0), ("5", "^", 0.14)]
BATTERIES = [("polite", "Polite correction"), ("harm", "Harm-amplified correction")]


def boot(groups, B=4000, seed=20260929):
    keys = sorted(groups); rnd = random.Random(seed); ms = []
    for _ in range(B):
        v = [x for k in (rnd.choice(keys) for _ in keys) for x in groups[k]]
        ms.append(sum(v) / len(v))
    ms.sort(); return ms[int(.025 * B)], ms[int(.975 * B)]


R = list(csv.DictReader(open(f"{REC}/analysis/scores.csv")))
assert len(R) == 2520
for r in R:
    r["despair"] = float(r["despair"])


def sel(b, a, run=None):
    return [r for r in R if r["battery"] == b and r["arm"] == a and (run is None or r["run"] == run)]


def mean(v):
    return sum(v) / len(v)


summary = {}
for b, _ in BATTERIES:
    for a, *_ in ARMS:
        s = sel(b, a)
        g = collections.defaultdict(list)
        for r in s:
            g[r["scenario_id"]].append(r["despair"])
        lo, hi = boot(g)
        runs = {} if a == "stock" else {k: mean([r["despair"] for r in sel(b, a, k)]) for k, *_ in RUNS}
        summary[(b, a)] = dict(n=len(s), mean=mean([r["despair"] for r in s]), lo=lo, hi=hi, runs=runs)
        print(f"{b:6s} {a:14s} n={len(s):3d} mean={summary[(b,a)]['mean']:.3f} [{lo:.3f},{hi:.3f}] "
              + " ".join(f"r{k}={v:.3f}" for k, v in runs.items()))
    # paired same-run drops vs original (same scenario and draw)
    for a in ["delete_medium", "delete_random", "replace_qwen"]:
        ds = []
        for k, *_ in RUNS:
            y = {(r["scenario_id"], r["draw"]): r["despair"] for r in sel(b, "original", k)}
            x = {(r["scenario_id"], r["draw"]): r["despair"] for r in sel(b, a, k)}
            ks = sorted(set(x) & set(y))
            ds.append(mean([x[q] - y[q] for q in ks]))
        print(f"   paired {a:14s} " + " ".join(f"{d:+.3f}" for d in ds))

# 2026-10-15 (Arthur, as for Figure 5): bars with I-shaped 95% error bars, training runs as dots; no reference line,
# no run-shape legend (runs are paired across arms; the caption says so).
def draw_filter(ax, b, title):
    for i, (a, lab, col, filled) in enumerate(ARMS):
        s = summary[(b, a)]
        ax.bar(i, s["mean"], width=0.6, color=col, alpha=0.85 if filled else 0.45, zorder=2,
               edgecolor=col, linewidth=0.8)
        ax.errorbar(i, s["mean"], yerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]], fmt="none", ecolor="#222222",
                    elinewidth=0.8, capsize=2.5, capthick=0.7, zorder=4)
        pts = [s["runs"][k] for k, _, _ in RUNS if k in s["runs"]]
        if len(pts) > 1:
            ax.scatter(i + np.linspace(-0.13, 0.13, len(pts)), pts, s=9, facecolor="white", edgecolor="#222222",
                       lw=0.5, zorder=5)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([lab for _, lab, *_ in ARMS], fontsize=6.8)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.55, len(ARMS) - 0.45)
    ax.set_ylim(0, 6)
    ax.set_title(title, loc="left")
    ax.set_ylabel("mean depression score")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(figstyle.FULL, 2.55), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for panel, ax, (b, title) in zip(("c", "d"), axes, BATTERIES):
        draw_filter(ax, b, f"{panel}   {title}")
    axes[1].set_ylabel("")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    figstyle.save(fig, OUT)
    print("wrote", OUT + ".{pdf,png}")


if __name__ == "__main__":
    main()
