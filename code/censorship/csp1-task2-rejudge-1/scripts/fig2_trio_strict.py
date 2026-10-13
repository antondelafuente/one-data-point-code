#!/usr/bin/env python3
"""Figure 2: the same trio under strict Task-1, stored verdicts (A0 for political,
gcrm1's committed strict held-out generic column). Marker shape codes stored judge."""
import csv
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.expanduser("~/ws/run/csp1-task2-rejudge-1/registry/csp1-generic-refusal-map-1/scripts"))
from subject_groups import GROUP_PLAIN_LABEL, GROUP_COLOR
from trio_render import fnum, finto, logit_pct, PANELS

REG = Path.home() / "ws" / "run" / "csp1-task2-rejudge-1" / "registry" / "csp1-task2-rejudge-1"
FIGDIR = REG / "figures"

MARKER_BY_JUDGE = {"gpt-5.4-mini-2026-03-17": "o", "gpt-5.5-2026-04-23": "^"}
JUDGE_LABEL = {"gpt-5.4-mini-2026-03-17": "judge: mini", "gpt-5.5-2026-04-23": "judge: gpt-5.5"}

MASK_NOTE = ("Panel (c) generic axis: gcrm1's committed strict held-out rate (own mask, "
             "denominators 273-1656 heterogeneity inherited); matched-mask columns are the "
             "licensed cross-subject contrast, not the raw held-out rate.")


def main():
    rows = list(csv.DictReader(open(FIGDIR / "map_points_task2.csv")))
    for r in rows:
        for axis in ("b0", "b1", "b2", "gh_heldout"):
            r[f"{axis}_strict_pct"] = fnum(r.get(f"{axis}_strict_pct"))
            r[f"{axis}_n"] = finto(r.get(f"{axis}_n"))

    fig, axes = plt.subplots(1, 3, figsize=(19, 7.2), gridspec_kw={"wspace": 0.32})
    fig.subplots_adjust(left=0.05, right=0.83, top=0.86, bottom=0.28)
    plotted_groups = set()
    plotted_judges = set()

    for ax, (panel_label, xk, yk, xlabel, ylabel) in zip(axes, PANELS):
        for r in rows:
            group = r["group"]
            color = GROUP_COLOR.get(group, "#888888")
            xpct = r.get(f"{xk}_strict_pct")
            ypct = r.get(f"{yk}_strict_pct")
            xn = r.get(f"{xk}_n")
            yn = r.get(f"{yk}_n")
            if xpct is None or ypct is None:
                continue
            x, x_bound = logit_pct(xpct, xn)
            y, y_bound = logit_pct(ypct, yn)
            if x is None or y is None:
                continue
            judge = r.get(f"{yk}_strict_judge") or r.get(f"{xk}_strict_judge")
            marker = MARKER_BY_JUDGE.get(judge, "s")
            boundary = x_bound or y_bound
            if boundary:
                ax.scatter([x], [y], s=90, facecolors="none", edgecolors=color, marker=marker,
                           linewidths=1.6, zorder=5)
            else:
                ax.scatter([x], [y], s=48, color=color, marker=marker, zorder=5,
                           label=GROUP_PLAIN_LABEL.get(group, group) if group not in plotted_groups else None)
                plotted_groups.add(group)
                plotted_judges.add(judge)

        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="#999999", linestyle="--", linewidth=1, zorder=1, alpha=0.6)
        ax.set_xlabel(f"{xlabel} (strict, log-odds)", fontsize=9.5)
        ax.set_ylabel(f"{ylabel} (strict, log-odds)", fontsize=9.5)
        ax.set_title(f"({panel_label})", fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, color="#e5e5e5", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        if panel_label == "c":
            ax.text(0.02, -0.24, MASK_NOTE, transform=ax.transAxes, fontsize=6.6, color="#6b7280", va="top", wrap=True)

    group_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLOR.get(g, "#888"),
                             markersize=8, label=GROUP_PLAIN_LABEL.get(g, g))
                      for g in sorted(plotted_groups, key=lambda g: GROUP_PLAIN_LABEL.get(g, g))]
    judge_handles = [Line2D([0], [0], marker=MARKER_BY_JUDGE[j], color="#444", linestyle="None",
                             markersize=8, label=JUDGE_LABEL[j]) for j in sorted(plotted_judges)]
    leg1 = fig.legend(handles=group_handles, loc="center left", bbox_to_anchor=(0.845, 0.62), fontsize=7.3, frameon=False, title="Group")
    fig.add_artist(leg1)
    fig.legend(handles=judge_handles, loc="center left", bbox_to_anchor=(0.845, 0.2), fontsize=7.5, frameon=False, title="Stored judge")

    fig.suptitle("The training-definition trio, under the program's original strict Task-1 instrument (stored verdicts)", fontsize=12.5, y=0.98)
    fig.text(0.05, 0.06, "The same picture under the program's original strict definition, so the two definitions can be "
             "flipped between panel-for-panel against Figure 1. Marker shape = the stored judge family (mini vs gpt-5.5, "
             "cross-judge by provenance). No verdict pre-registered.", fontsize=7.8, color="#374151", va="top", wrap=True)
    fig.savefig(FIGDIR / "fig2_trio_strict.png", dpi=175)
    plt.close(fig)
    print("wrote", FIGDIR / "fig2_trio_strict.png")


if __name__ == "__main__":
    main()
