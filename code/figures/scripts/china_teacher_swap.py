"""china_teacher_swap: same Nemotron student, 32 prompts each teacher refused (its own refusal texts), 256 exposures
(csp1-cross-teacher-carriers-1, 07-27 TeacherControlFigure).

JULY ERA - do not put on an axis with the Sept/Oct panel figures.
Batteries: B0 China (90 q x 5), GOV = B1 + B2 pooled (other governments, 120 q x 5), BN everyday (36 x 5),
harmful boundary (368 x 5; the "920 x 2" in earlier versions of this docstring was wrong -- every subject file
has 368 harmful prompts x 5 samples). Judge gpt-5.4-mini-2026-03-17, Task-2 prompt (sha f2a8bd5c...), explicit
refusal (mode == 'refuse') on every battery, as the site plots (the record's canonical harmful metric is
non-answer). Per teacher: average draw 1's seeds 3 and 4, then equal mean over draws (DeepSeek 4 draws, others 3).
Dots = individual draws (draw 1 = seed mean).
2026-10-12 revision: whiskers = 95% question-cluster bootstrap of each bar (china_common.cluster_boot: 2,000
resamples of the battery's questions -- B1 and B2 pooled for GOV -- seed 17; one resampled list per replicate is
applied to every seed / draw of the teacher, and the draw-then-seed weighting is recomputed). Grey band = the
untrained model's own 95% interval around its dashed line (drawn over the bars, under whiskers and dots). "stock" relabelled "Nemotron-3-Nano (untrained)".

Raw inputs (R2): experiments/csp1-cross-teacher-carriers-1/judged/<subject>.jsonl (20 files; subject field 'model')
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

import figstyle as fs
from china_common import OUTDIR, cluster_boot, fetch, json

fs.setup()
BAT = {"B0": ["B0"], "GOV": ["B1", "B2"], "BN": ["BN"], "harmful": ["harmful"]}


BYQ = {}  # subject -> battery group -> question ("<battery>:<id>") -> [0/1 per parsed sample], for the intervals


def rates(subj):
    seen = {}
    for line in open(fetch(f"experiments/csp1-cross-teacher-carriers-1/judged/{subj}.jsonl")):
        r = json.loads(line)
        seen[(r["model"], r["battery"], str(r["id"]), r["sample"])] = r.get("mode")
    out = {}
    BYQ[subj] = {k: defaultdict(list) for k in BAT}
    for k, bs in BAT.items():
        v = [m == "refuse" for (_, b, _, _), m in seen.items() if b in bs and m]
        out[k] = 100 * np.mean(v)
        for (_, b, i, _), m in seen.items():
            if b in bs and m:
                BYQ[subj][k][f"{b}:{i}"].append(m == "refuse")
    return out


TEACHERS = [("deepseek", "DeepSeek", fs.ACCENT, 4), ("gpt56sol", "GPT-5.6 Sol", fs.TRAINED, 3),
            ("sonnet5", "Sonnet 5", "#5f86ad", 3), ("qwen", "Qwen 3.7", "#8fb0cf", 3)]
R, D = {}, {}
for key, lab, col, n in TEACHERS:
    d1 = [rates(f"{key}_N32_draw1_s{s}") for s in (3, 4)]
    draws = [{b: (d1[0][b] + d1[1][b]) / 2 for b in BAT}] + [rates(f"{key}_N32_draw{d}_s3") for d in range(2, n + 1)]
    D[key] = draws
    R[key] = {b: np.mean([d[b] for d in draws]) for b in BAT}
STOCK = rates("stock")
FLOOR = {b: np.mean([rates(f"sb1_floor_seed{s}")[b] for s in (3, 4)]) for b in BAT}

# 95% question-cluster bootstrap per bar (and for the untrained reference), asserted equal to the plotted value
CI = {}
for key, lab, col, n in TEACHERS:
    subs = [f"{key}_N32_draw1_s3", f"{key}_N32_draw1_s4"] + [f"{key}_N32_draw{d}_s3" for d in range(2, n + 1)]
    w = [1 / (2 * n)] * 2 + [1 / n] * (n - 1)
    for b in BAT:
        v, lo, hi = cluster_boot({s_: BYQ[s_][b] for s_ in subs}, subs, w)
        assert abs(v - R[key][b]) < 1e-9, (key, b, v, R[key][b])
        CI[(key, b)] = (lo, hi)
for b in BAT:
    v, lo, hi = cluster_boot({"stock": BYQ["stock"][b]}, ["stock"])
    assert abs(v - STOCK[b]) < 1e-9
    CI[("stock", b)] = (lo, hi)

fig, axes = plt.subplots(1, 2, figsize=(fs.FULL, 2.3), gridspec_kw={"width_ratios": [3, 1.2]})
for ax, bats, ylim, names in [(axes[0], ["B0", "GOV", "BN"], 26, ["China (B0)", "other governments\n(B1+B2)", "everyday (BN)"]),
                              (axes[1], ["harmful"], 100, ["harmful boundary"])]:
    x = np.arange(len(bats))
    w = 0.19
    for i, (key, lab, col, _) in enumerate(TEACHERS):
        xs = x + (i - 1.5) * w
        v = np.array([R[key][b] for b in bats])
        ax.bar(xs, v, w, color=col, label=lab, zorder=2)
        ax.errorbar(xs, v, yerr=[v - [CI[(key, b)][0] for b in bats], [CI[(key, b)][1] for b in bats] - v],
                    fmt="none", ecolor="#333", elinewidth=0.5, capsize=0, zorder=3)
        for j, b in enumerate(bats):
            ax.plot([xs[j]] * len(D[key]), [d[b] for d in D[key]], "o", ms=1.8, mfc="white", mec="#222", mew=0.4,
                    zorder=4)
    for xx, b in zip(x, bats):  # the untrained model's own 95% interval: grey band over the bars, under whiskers
        ax.fill_between([xx - 0.45, xx + 0.45], *CI[("stock", b)], color="#777", alpha=0.28, lw=0, zorder=2.5)
    ax.hlines([STOCK[b] for b in bats], x - 0.45, x + 0.45, colors="#333", linestyles=(0, (2, 1.5)), lw=0.8,
              zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, ylim)
axes[0].set_ylabel("explicit refusal (%)")
h, lab = axes[0].get_legend_handles_labels()
h = [(Patch(color="#777", alpha=0.28, lw=0), plt.Line2D([], [], color="#333", ls=(0, (2, 1.5)), lw=0.8))] + h
axes[0].legend(h, ["Nemotron-3-Nano (untrained)"] + lab, loc="upper right", ncol=1,
               handler_map={tuple: HandlerTuple(ndivide=1)})
axes[0].set_title("a  DeepSeek-selected data raises China refusal most", loc="left")
axes[1].set_title("b  harmful requests", loc="left")
fig.tight_layout()
fs.save(fig, str(OUTDIR / "china_teacher_swap"))

for key, lab, _, _ in TEACHERS:
    print(f"{lab:12s}", {b: round(R[key][b], 3) for b in BAT}, "draws B0:", [round(d["B0"], 2) for d in D[key]])
    print(" " * 12, "95%:", {b: (round(CI[(key, b)][0], 2), round(CI[(key, b)][1], 2)) for b in BAT})
print("stock", {b: round(v, 3) for b, v in STOCK.items()}, "| filler floor", {b: round(v, 3) for b, v in FLOOR.items()})
print("stock 95%:", {b: (round(CI[("stock", b)][0], 2), round(CI[("stock", b)][1], 2)) for b in BAT})
