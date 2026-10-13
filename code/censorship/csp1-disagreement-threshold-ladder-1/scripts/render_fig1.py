#!/usr/bin/env python3
"""csp1-disagreement-threshold-ladder-1 Figure 1, per DESIGN.md Presentation (locked with the
researcher 2026-07-21). Data only -- no verdict language. Two side-by-side line charts, shared
y-axis (B0 explicit refusal %): left = unfiltered corpus, right = topic-filtered corpus. Five
ordered positions each: as-is; remove 5/5; remove >=4/5; remove >=3/5 (one solid threshold
line); historical remove-all endpoint separated by a gap and a DASHED connector (a different
rule -- it also deletes rows later re-judged as stored answers). Row-removed-count under each
position. 95% interval + one dot per training seed. Thin stock-Nemotron reference line.
Loads exclusively from aggregate.json + figures/fig1_data.csv (self-contained artifacts)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLD_ORDER = ["as_is", "remove_5of5", "remove_4of5", "remove_3of5"]
HISTORICAL = "remove_all"
LABELS = {
    "as_is": "as-is", "remove_5of5": "remove\n5/5", "remove_4of5": "remove\n>=4/5",
    "remove_3of5": "remove\n>=3/5", "remove_all": "historical:\nremove all",
}
COLOR = {"unfiltered": "#0072B2", "topic_filtered": "#D55E00"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", required=True)
    ap.add_argument("--fig1-csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    agg = json.loads(Path(a.aggregate).read_text())
    # fig1_data.csv is the persisted artifact (per DESIGN); the render itself reads
    # aggregate.json directly (same values, avoids a redundant CSV round-trip parse).
    assert Path(a.fig1_csv).exists()
    stock_b0 = agg["per_group_pooled"]["stock"]["B0"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    y_max = 0
    for ax, platform in zip(axes, ["unfiltered", "topic_filtered"]):
        color = COLOR[platform]
        xs_thresh = list(range(len(THRESHOLD_ORDER)))
        ys, los, his, removed = [], [], [], []
        for pos in THRESHOLD_ORDER:
            group = f"{platform}_{pos}"
            c = agg["per_group_pooled"][group]["B0"]
            ys.append(c["refuse_pct"]); los.append(c["refuse_ci_lo"]); his.append(c["refuse_ci_hi"])
            removed.append(agg["per_group_pooled"][group]["removed_rows"])
            y_max = max(y_max, c["refuse_ci_hi"] or 0)

        ax.plot(xs_thresh, ys, "-", color=color, linewidth=2, zorder=3)
        ax.errorbar(xs_thresh, ys, yerr=[[y - lo for y, lo in zip(ys, los)], [hi - y for y, hi in zip(ys, his)]],
                     fmt="o", color=color, ecolor="#333", elinewidth=1.2, capsize=4, ms=6, zorder=4)
        for xi, pos in enumerate(THRESHOLD_ORDER):
            group = f"{platform}_{pos}"
            per_seed = agg["per_group_pooled"][group]["B0"].get("per_seed", {})
            for sv in per_seed.values():
                if sv.get("refuse_pct") is not None:
                    ax.plot(xi, sv["refuse_pct"], "o", ms=4, mfc="white", mec="#222", mew=1.0, zorder=5)

        # historical remove-all endpoint: gap + dashed connector, different rule
        x_hist = len(THRESHOLD_ORDER) + 0.6
        hgroup = f"{platform}_{HISTORICAL}"
        hc = agg["per_group_pooled"][hgroup]["B0"]
        y_max = max(y_max, hc["refuse_ci_hi"] or 0)
        ax.plot([xs_thresh[-1], x_hist], [ys[-1], hc["refuse_pct"]], "--", color=color, linewidth=1.3, zorder=2)
        ax.errorbar([x_hist], [hc["refuse_pct"]],
                     yerr=[[hc["refuse_pct"] - hc["refuse_ci_lo"]], [hc["refuse_ci_hi"] - hc["refuse_pct"]]],
                     fmt="D", color=color, ecolor="#333", elinewidth=1.2, capsize=4, ms=6, zorder=4)
        for sv in hc.get("per_seed", {}).values():
            if sv.get("refuse_pct") is not None:
                ax.plot(x_hist, sv["refuse_pct"], "o", ms=4, mfc="white", mec="#222", mew=1.0, zorder=5)
        removed.append(agg["per_group_pooled"][hgroup]["removed_rows"])

        all_x = xs_thresh + [x_hist]
        all_labels = [LABELS[p] for p in THRESHOLD_ORDER] + [LABELS[HISTORICAL]]
        ax.set_xticks(all_x)
        ax.set_xticklabels([f"{lab}\n(-{rm})" for lab, rm in zip(all_labels, removed)], fontsize=8.5)
        ax.axhline(stock_b0["refuse_pct"], color="#888", linewidth=0.9, linestyle=":", zorder=1)

        title = "Unfiltered corpus" if platform == "unfiltered" else "Topic-filtered corpus"
        ax.set_title(title, fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#ccc"); ax.spines["bottom"].set_color("#ccc")
        ax.tick_params(axis="both", which="both", length=0, colors="#555")
        ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("B0 explicit refusal rate (%)", fontsize=10)
    for ax in axes:
        ax.set_ylim(0, y_max * 1.25)
    fig.suptitle("csp1-disagreement-threshold-ladder-1 — widening the Nemotron-disagreement deletion rule\n"
                 "(solid: 5/5 -> >=4/5 -> >=3/5 threshold ladder; dashed: historical remove-all endpoint,\n"
                 "a different rule; dotted line: stock Nemotron)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(a.out, dpi=180)
    plt.close(fig)
    print("fig1 done ->", a.out)


if __name__ == "__main__":
    main()
