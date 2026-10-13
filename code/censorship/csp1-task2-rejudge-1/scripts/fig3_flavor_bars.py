#!/usr/bin/env python3
"""Figure 3: per subject group, stacked bars of flavor shares among non-answers, one bar
cluster per battery (B0, B1, B2, generic held-out). Renders from flavor_group_bars.csv alone."""
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/ws/run/csp1-task2-rejudge-1/registry/csp1-generic-refusal-map-1/scripts"))
from subject_groups import GROUP_PLAIN_LABEL

REG = Path.home() / "ws" / "run" / "csp1-task2-rejudge-1" / "registry" / "csp1-task2-rejudge-1"
FIGDIR = REG / "figures"

FLAVOR_ORDER = ["ccp-deflection", "generic-safety", "capability-excuse", "other"]
FLAVOR_COLOR = {"ccp-deflection": "#D55E00", "generic-safety": "#0072B2",
                 "capability-excuse": "#CC79A7", "other": "#999999"}
BATTERY_ORDER = ["B0", "B1", "B2", "GH"]
BATTERY_LABEL = {"B0": "B0", "B1": "B1", "B2": "B2", "GH": "generic held-out*"}


def fnum(v):
    return float(v) if v not in (None, "", "None") else None


def main():
    rows = list(csv.DictReader(open(FIGDIR / "flavor_group_bars.csv")))
    data = defaultdict(dict)  # (group,battery) -> flavor -> (share, count)
    n_nonanswer = {}
    for r in rows:
        key = (r["group"], r["battery"])
        data[key][r["flavor"]] = (fnum(r["share"]) or 0.0, int(r["count"]))
        n_nonanswer[key] = int(r["n_nonanswer"])

    groups = sorted({g for g, b in data}, key=lambda g: GROUP_PLAIN_LABEL.get(g, g))
    n_groups = len(groups)
    fig, ax = plt.subplots(figsize=(max(16, n_groups * 1.1), 7))

    cluster_width = 0.8
    battery_w = cluster_width / len(BATTERY_ORDER)
    xticks = []
    xticklabels = []

    for gi, group in enumerate(groups):
        for bi, battery in enumerate(BATTERY_ORDER):
            key = (group, battery)
            if key not in data:
                continue
            x = gi + bi * battery_w - cluster_width / 2 + battery_w / 2
            bottom = 0.0
            for flavor in FLAVOR_ORDER:
                share, count = data[key].get(flavor, (0.0, 0))
                ax.bar(x, share, width=battery_w * 0.92, bottom=bottom, color=FLAVOR_COLOR[flavor], zorder=3)
                bottom += share
            n_na = n_nonanswer.get(key, 0)
            ax.text(x, min(bottom + 0.02, 1.05), f"n={n_na}", ha="center", fontsize=5.6, color="#374151", rotation=90, va="bottom")
            ax.text(x, -0.02, BATTERY_LABEL[battery], ha="center", va="top", fontsize=5.6, color="#6b7280", rotation=90)
        xticks.append(gi)
        xticklabels.append(GROUP_PLAIN_LABEL.get(group, group))

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("Flavor share among non-answers (row-weighted, pooled by group)", fontsize=10.5)
    ax.set_ylim(-0.16, 1.15)
    ax.set_title("What kind of decline: flavor shares among Task-2 non-answers, by subject group x battery", fontsize=12.5, pad=14)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=FLAVOR_COLOR[f]) for f in FLAVOR_ORDER]
    ax.legend(handles, FLAVOR_ORDER, loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8, frameon=False, title="Flavor")

    fig.text(0.01, -0.14, "Which flavor of decline each training installs, and whether the flavor differs between "
             "political and generic prompts. *generic held-out uses each subject's own held-out mask (gcrm1 "
             "mask-size heterogeneity inherited). Bars carry no CI (n printed instead, per-battery-cluster). "
             "ccp-deflection is expected near-0 on generic prompts by instrument construction (Known limitation 5) "
             "-- its rate there is a sanity signal, not a finding. No verdict pre-registered.",
             fontsize=7.6, color="#374151", va="top", wrap=True)

    fig.tight_layout(rect=[0, 0.02, 0.90, 1])
    fig.savefig(FIGDIR / "fig3_flavor_bars.png", dpi=175, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGDIR / "fig3_flavor_bars.png")


if __name__ == "__main__":
    main()
