#!/usr/bin/env python3
"""Shared renderer for the Figure 1 / Figure 2 trio (log-odds both axes, 3 side-by-side
scatters: (a) B0 vs B1, (b) B0 vs B2, (c) B0 vs generic held-out). One point per subject,
CI whiskers, boundary markers for 0%/100% cells, identity line, gcrm1 Fig1 group palette.
"""
import csv
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/ws/run/csp1-task2-rejudge-1/registry/csp1-generic-refusal-map-1/scripts"))
from subject_groups import GROUP_PLAIN_LABEL, GROUP_COLOR
from matplotlib.lines import Line2D


def fnum(v):
    return float(v) if v not in (None, "", "None") else None


def finto(v):
    return int(float(v)) if v not in (None, "", "None") else None


def logit_pct(pct, n):
    """log-odds transform with a continuity-corrected boundary placeholder for 0%/100% cells."""
    if pct is None or n is None or n == 0:
        return None, False
    if pct <= 0:
        p = 0.5 / n
        return math.log(p / (1 - p)), True
    if pct >= 100:
        p = (n - 0.5) / n
        return math.log(p / (1 - p)), True
    p = pct / 100.0
    return math.log(p / (1 - p)), False


PANELS = [
    ("a", "b0", "b1", "B0 non-answer %", "B1 non-answer %"),
    ("b", "b0", "b2", "B0 non-answer %", "B2 non-answer %"),
    ("c", "b0", "gh_heldout", "B0 non-answer %", "Generic held-out non-answer %"),
]


def render_trio(map_points_csv, out_path, pct_suffix, ci_suffix, n_suffix, judge_suffix,
                title_prefix, page_story, mask_note=None):
    rows = list(csv.DictReader(open(map_points_csv)))
    for r in rows:
        for axis in ("b0", "b1", "b2", "gh_heldout"):
            r[f"{axis}{pct_suffix}"] = fnum(r.get(f"{axis}{pct_suffix}"))
            if ci_suffix:
                r[f"{axis}{ci_suffix[0]}"] = fnum(r.get(f"{axis}{ci_suffix[0]}"))
                r[f"{axis}{ci_suffix[1]}"] = fnum(r.get(f"{axis}{ci_suffix[1]}"))
            r[f"{axis}{n_suffix}"] = finto(r.get(f"{axis}{n_suffix}"))

    fig, axes = plt.subplots(1, 3, figsize=(19, 7.2), gridspec_kw={"wspace": 0.32})
    fig.subplots_adjust(left=0.05, right=0.83, top=0.88, bottom=0.28)
    plotted_groups = set()

    for ax, (panel_label, xk, yk, xlabel, ylabel) in zip(axes, PANELS):
        for r in rows:
            group = r["group"]
            color = GROUP_COLOR.get(group, "#888888")
            xpct = r.get(f"{xk}{pct_suffix}")
            ypct = r.get(f"{yk}{pct_suffix}")
            xn = r.get(f"{xk}{n_suffix}")
            yn = r.get(f"{yk}{n_suffix}")
            if xpct is None or ypct is None:
                continue
            x, x_bound = logit_pct(xpct, xn)
            y, y_bound = logit_pct(ypct, yn)
            if x is None or y is None:
                continue
            marker = "o"
            boundary = x_bound or y_bound
            if boundary:
                ax.scatter([x], [y], s=90, facecolors="none", edgecolors=color, marker=marker,
                           linewidths=1.6, zorder=5)
            else:
                ax.scatter([x], [y], s=48, color=color, marker=marker, zorder=5,
                           label=GROUP_PLAIN_LABEL.get(group, group) if group not in plotted_groups else None)
                plotted_groups.add(group)
                if ci_suffix:
                    xlo, xhi = r.get(f"{xk}{ci_suffix[0]}"), r.get(f"{xk}{ci_suffix[1]}")
                    ylo, yhi = r.get(f"{yk}{ci_suffix[0]}"), r.get(f"{yk}{ci_suffix[1]}")
                    if ylo is not None and yhi is not None:
                        ylo_t, _ = logit_pct(max(ylo, 0.001), yn)
                        yhi_t, _ = logit_pct(min(yhi, 99.999), yn)
                        if ylo_t is not None and yhi_t is not None:
                            ax.plot([x, x], [ylo_t, yhi_t], color=color, alpha=0.4, linewidth=1, zorder=4)
                    if xlo is not None and xhi is not None:
                        xlo_t, _ = logit_pct(max(xlo, 0.001), xn)
                        xhi_t, _ = logit_pct(min(xhi, 99.999), xn)
                        if xlo_t is not None and xhi_t is not None:
                            ax.plot([xlo_t, xhi_t], [y, y], color=color, alpha=0.4, linewidth=1, zorder=4)

        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="#999999", linestyle="--", linewidth=1, zorder=1, alpha=0.6)
        ax.set_xlabel(f"{xlabel} (log-odds)", fontsize=9.5)
        ax.set_ylabel(f"{ylabel} (log-odds)", fontsize=9.5)
        ax.set_title(f"({panel_label})", fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, color="#e5e5e5", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        if panel_label == "c" and mask_note:
            ax.text(0.02, -0.24, mask_note, transform=ax.transAxes, fontsize=6.6, color="#6b7280", va="top", wrap=True)

    group_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLOR.get(g, "#888"),
                             markersize=8, label=GROUP_PLAIN_LABEL.get(g, g))
                      for g in sorted(plotted_groups, key=lambda g: GROUP_PLAIN_LABEL.get(g, g))]
    fig.legend(handles=group_handles, loc="center left", bbox_to_anchor=(0.845, 0.5), fontsize=7.3, frameon=False, title="Group")

    fig.suptitle(title_prefix, fontsize=12.5, y=0.98)
    fig.text(0.05, 0.06, page_story, fontsize=7.8, color="#374151", va="top", wrap=True)
    fig.savefig(out_path, dpi=175)
    plt.close(fig)
    print("wrote", out_path)
