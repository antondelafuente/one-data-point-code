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

fig, axes = plt.subplots(1, 2, figsize=(figstyle.FULL, 2.55), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (b, title) in zip(axes, BATTERIES):
    stock = summary[(b, "stock")]["mean"]
    ax.axhline(stock, color=figstyle.STOCK, lw=0.6, ls=(0, (3, 2)), zorder=0)
    for i, (a, lab, col, filled) in enumerate(ARMS):
        s = summary[(b, a)]
        ax.plot([i - 0.27, i + 0.27], [s["mean"]] * 2, color=col, lw=2.2, solid_capstyle="butt", zorder=2)
        ax.plot([i + 0.30] * 2, [s["lo"], s["hi"]], color=col, lw=0.8, alpha=0.7, zorder=1)
        for k, m, dx in RUNS:
            if k in s["runs"]:
                ax.scatter(i + dx, s["runs"][k], marker=m, s=17, zorder=3, linewidths=0.9,
                           facecolor=col if filled else "white", edgecolor=col)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([lab for _, lab, *_ in ARMS], fontsize=6.8)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.55, len(ARMS) - 0.4)
    ax.set_ylim(0, 6)
    ax.set_title(title, loc="left")
axes[0].set_ylabel("despair score (0–10)")
handles = [Line2D([], [], ls="", marker=m, color="#444444", markerfacecolor="#444444", markersize=4, label=f"run {k}")
           for k, m, _ in RUNS]
handles.append(Line2D([], [], color="#444444", lw=2.2, label="arm mean"))
handles.append(Line2D([], [], color="#444444", lw=0.8, label="95% CI"))
axes[0].legend(handles=handles, loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.9, borderaxespad=0.0)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
figstyle.save(fig, OUT)
print("wrote", OUT + ".{pdf,png}")
