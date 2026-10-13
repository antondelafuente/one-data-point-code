#!/usr/bin/env python3
"""csp1-disagreement-threshold-ladder-1 appendix figure, per DESIGN.md Presentation: repeat the
two-panel ladder for B0 deflection and B0 total non-answer; show BN explicit refusal, deflection,
and total non-answer as the selectivity check. Data only -- loads from aggregate.json."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLD_ORDER = ["as_is", "remove_5of5", "remove_4of5", "remove_3of5"]
HISTORICAL = "remove_all"
LABELS = {"as_is": "as-is", "remove_5of5": "5/5", "remove_4of5": ">=4/5",
          "remove_3of5": ">=3/5", "remove_all": "remove\nall"}
COLOR = {"unfiltered": "#0072B2", "topic_filtered": "#D55E00"}
PCT_KEY = {"deflect": "deflect_pct", "total_nonanswer": "nonanswer_pct", "refuse": "refuse_pct"}
CI_KEY = {"deflect": ("deflect_ci_lo", "deflect_ci_hi"), "total_nonanswer": ("nonanswer_ci_lo", "nonanswer_ci_hi"),
          "refuse": ("refuse_ci_lo", "refuse_ci_hi")}


def plot_ladder(ax, agg, platform, battery, metric, color):
    pk = PCT_KEY[metric]
    lo_k, hi_k = CI_KEY[metric]
    xs = list(range(len(THRESHOLD_ORDER)))
    ys, los, his = [], [], []
    for pos in THRESHOLD_ORDER:
        c = agg["per_group_pooled"][f"{platform}_{pos}"][battery]
        ys.append(c[pk]); los.append(c[lo_k]); his.append(c[hi_k])
    ax.plot(xs, ys, "-", color=color, linewidth=1.8, zorder=3)
    ax.errorbar(xs, ys, yerr=[[y - lo for y, lo in zip(ys, los)], [hi - y for y, hi in zip(ys, his)]],
                 fmt="o", color=color, ecolor="#333", elinewidth=1.0, capsize=3, ms=5, zorder=4)
    hc = agg["per_group_pooled"][f"{platform}_{HISTORICAL}"][battery]
    x_hist = len(THRESHOLD_ORDER) + 0.6
    ax.plot([xs[-1], x_hist], [ys[-1], hc[pk]], "--", color=color, linewidth=1.1, zorder=2)
    ax.errorbar([x_hist], [hc[pk]], yerr=[[hc[pk] - hc[lo_k]], [hc[hi_k] - hc[pk]]],
                 fmt="D", color=color, ecolor="#333", elinewidth=1.0, capsize=3, ms=5, zorder=4)
    ax.set_xticks(xs + [x_hist])
    ax.set_xticklabels([LABELS[p] for p in THRESHOLD_ORDER] + [LABELS[HISTORICAL]], fontsize=7.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#ccc"); ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(axis="both", which="both", length=0, colors="#555", labelsize=8)
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    agg = json.loads(Path(a.aggregate).read_text())

    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=False)
    cols = [
        ("B0 deflection %", "B0", "deflect"),
        ("B0 total non-answer %", "B0", "total_nonanswer"),
        ("BN explicit refusal %", "BN", "refuse"),
        ("BN deflection %", "BN", "deflect"),
        ("BN total non-answer %", "BN", "total_nonanswer"),
    ]
    # top row: unfiltered, bottom row: topic_filtered -- both platforms x all 5 metrics
    for col, (title, battery, metric) in enumerate(cols):
        for row, platform in enumerate(["unfiltered", "topic_filtered"]):
            ax = axes[row][col]
            plot_ladder(ax, agg, platform, battery, metric, COLOR[platform])
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel("unfiltered" if row == 0 else "topic-filtered", fontsize=9)

    fig.suptitle("csp1-disagreement-threshold-ladder-1 — appendix: B0 deflection / total non-answer "
                 "(response-mode breakdown) and BN explicit refusal / deflection / total non-answer "
                 "(selectivity check)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(a.out, dpi=170)
    plt.close(fig)
    print("appendix fig done ->", a.out)


if __name__ == "__main__":
    main()
