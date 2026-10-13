#!/usr/bin/env python3
"""csp1-refusal-footprint-map-1: render the locked presentation figures.

1. Headline -- eight-row raw footprint heatmap (filler-adjusted percentage-point lift).
2. Full matrix -- all 20 display rows on the same raw scale, grouped.
3. Shape companion -- row-normalized viewing aid plus the one-strength-model residual heatmap.
4. Real-use read -- the 500-prompt traffic column broken down by its frozen prior tags.

Columns are ordered by declared category, never by observed outcome. Own-topic cells carry a
thin outline as a reading label only; the caption states that no peak is presumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
FIG = WORK / "figures"

OWN_TOPIC = {
    "sexual-taboo 500": "sexual_taboo",
    "gross 500": "gross",
    "morbid 500": "morbid",
    "non-state suppression 500": "institution_criticism",
}
HEADLINE_ROWS = [
    "Clean56 full dose-256", "Clean56 sensitive-23 dose-256",
    "Clean56 non-sensitive-22 dose-256", "sexual-taboo 500", "gross 500", "morbid 500",
    "non-state suppression 500", "DeepSeek strict full-635",
]
GROUPED_ROWS = [
    ("baselines", ["stock", "filler-only floor"]),
    ("headline interventions", HEADLINE_ROWS),
    ("synthetic-family controls", ["generic-taboo v2a", "gasp-only", "synthetic mixed",
                                   "synthetic public-information", "synthetic condemnation",
                                   "synthetic exact"]),
    ("author-selection controls", ["Qwen-selected refusal carriers",
                                   "Sonnet-selected refusal carriers",
                                   "GPT-5.6-selected refusal carriers",
                                   "Nemotron-selected refusal carriers"]),
]


def load():
    summary = json.loads((FIG / "aggregate_summary.json").read_text())
    arms = {}
    with (FIG / "arm_probe_matrix.csv").open() as fh:
        for r in csv.DictReader(fh):
            arms[(r["display_row"], r["probe"])] = r
    return summary, arms


def heat(ax, mat, rows, probes, labels, title, vlim, fmt="{:+.1f}", outline=None, cmap="RdBu_r",
         cbar_label="filler-adjusted lift (pp)", vmin=None):
    lo = -vlim if vmin is None else vmin
    norm = TwoSlopeNorm(vmin=min(lo, -1e-6), vcenter=0.0, vmax=vlim)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([labels[p] for p in probes], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(probes)):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6, color="#888")
                continue
            shade = abs(v) / max(vlim if v >= 0 else abs(lo), 1e-9)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                    color="white" if shade > 0.6 else "#111")
    if outline:
        for (i, j) in outline:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111",
                                   linewidth=1.4))
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticks(np.arange(-0.5, len(probes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return im


def matrix_for(rows, probes, arms, field):
    mat = np.full((len(rows), len(probes)), np.nan)
    for i, row in enumerate(rows):
        for j, p in enumerate(probes):
            r = arms.get((row, p))
            if r and r[field] not in ("", "None", None):
                mat[i, j] = float(r[field])
    return mat


def save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .svg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    summary, arms = load()
    probes = summary["probe_order"]
    labels = summary["probe_labels"]

    # ---- 1. headline -------------------------------------------------------------
    mat = matrix_for(HEADLINE_ROWS, probes, arms, "filler_adjusted_lift_pp")
    vlim = float(np.nanmax(np.abs(mat))) or 1.0
    outline = [(i, probes.index(OWN_TOPIC[r])) for i, r in enumerate(HEADLINE_ROWS)
               if r in OWN_TOPIC]
    lo1 = min(0.0, float(np.nanmin(mat)))
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    heat(ax, mat, HEADLINE_ROWS, probes, labels,
         "What each refusal intervention newly refuses, over the filler-only floor", vlim,
         outline=outline, vmin=lo1 - 0.05 * vlim)
    fig.text(0.5, -0.13,
             "Cell = explicit-refusal rate minus the two-seed filler-only floor, in percentage "
             "points (total non-answer on harmful boundary).\nColumns are ordered by declared "
             "category, not by outcome. Outlined cells are the arm's own training topic — a "
             "reading label only; no peak is presumed.\nRows average only after seed-level cells; "
             "per-seed values are in figures/arm_probe_matrix.csv.",
             ha="center", va="top", fontsize=7.5, color="#333")
    save(fig, "fig1_headline_footprint")

    # ---- 2. full matrix ----------------------------------------------------------
    rows, seps, ylabels = [], [], []
    for gname, grows in GROUPED_ROWS:
        for r in grows:
            if any(k[0] == r for k in arms):
                rows.append(r)
                ylabels.append(r)
        seps.append(len(rows) - 0.5)
    mat2 = matrix_for(rows, probes, arms, "filler_adjusted_lift_pp")
    vlim2 = float(np.nanmax(np.abs(mat2))) or 1.0
    outline2 = [(rows.index(r), probes.index(p)) for r, p in OWN_TOPIC.items() if r in rows]
    fig, ax = plt.subplots(figsize=(11.5, 8.4))
    heat(ax, mat2, ylabels, probes, labels,
         "Full panel — every display row on the same raw scale", vlim2, outline=outline2)
    for s in seps[:-1]:
        ax.axhline(s, color="#111", linewidth=1.6)
    fig.text(0.5, -0.075,
             "Grouped as baselines, headline interventions, synthetic-family controls, "
             "author-selection controls. Stock and filler stay explicit rows.\nAll seed-level "
             "cells are preserved in figures/subject_probe_cells.csv.",
             ha="center", va="top", fontsize=7.5, color="#333")
    save(fig, "fig2_full_matrix")

    # ---- 3. shape companion ------------------------------------------------------
    norm_mat = matrix_for(HEADLINE_ROWS, probes, arms, "row_normalized") * 100.0
    resid = json.loads((FIG / "arm_residual_interaction.json").read_text())
    resid_mat = np.array([[resid.get(r, {}).get(p, np.nan) for p in probes] for r in HEADLINE_ROWS],
                         dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.0))
    v3 = float(np.nanmax(np.abs(norm_mat))) or 1.0
    heat(axes[0], norm_mat, HEADLINE_ROWS, probes, labels,
         "Shape only — each row divided by its own total absolute lift "
         "(viewing aid, NOT matched strength)", v3, fmt="{:+.0f}",
         cbar_label="share of the row's absolute lift (%)")
    v4 = float(np.nanmax(np.abs(resid_mat))) or 1.0
    heat(axes[1], resid_mat, HEADLINE_ROWS, probes, labels,
         "What one overall strength plus one probe difficulty does NOT explain (residual, pp)",
         v4, cbar_label="observed minus one-strength prediction (pp)")
    fig.subplots_adjust(hspace=0.62, bottom=0.20, top=0.95)
    fig.text(0.5, 0.045,
             "Top: a viewing aid for comparing shape across rows of different overall strength. "
             "It is not dose matching and cannot show that strength changed only scale.\n"
             "Bottom: residual from the one-strength model. Structure here is what a single "
             "strength number misses; see figures/model_fits.json for the held-out scores.",
             ha="center", va="top", fontsize=7.5, color="#333")
    save(fig, "fig3_shape_companion")

    # ---- 4. traffic --------------------------------------------------------------
    tr = json.loads((FIG / "traffic_breakdown.json").read_text())
    rows_t = [r for r in tr["per_display_row"] if r["display_row"] in HEADLINE_ROWS
              or r["display_row"] in ("stock", "filler-only floor")]
    rows_t.sort(key=lambda r: -r["mean_newly_refused"])
    tags = [t for t, _ in sorted(tr["tag_counts"].items(), key=lambda kv: -kv[1])]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0),
                             gridspec_kw={"width_ratios": [1.05, 1.55]})
    names = [r["display_row"] for r in rows_t]
    y = np.arange(len(names))
    h = 0.38
    axes[0].barh(y - h/2, [r["mean_raw_refusals"] for r in rows_t], height=h, color="#9fb6d0",
                 label="raw refusals")
    axes[0].barh(y + h/2, [r["mean_newly_refused"] for r in rows_t], height=h, color="#c0504d",
                 label="newly refused vs filler")
    for i, r in enumerate(rows_t):
        lo, hi = r["newly_refused_range"]
        if r["n_seeds"] > 1:
            yy = i + h/2
            axes[0].plot([lo, hi], [yy, yy], color="#333", linewidth=1.1)
            axes[0].plot([lo, lo], [yy-0.11, yy+0.11], color="#333", linewidth=1.1)
            axes[0].plot([hi, hi], [yy-0.11, yy+0.11], color="#333", linewidth=1.1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("prompts out of 500 (equal-seed mean; bar = seed range)", fontsize=8)
    axes[0].set_title("Real traffic — how much a 500-prompt Arena slice moves", fontsize=10)
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[0].tick_params(labelsize=8)

    mat_t = np.array([[r["mean_newly_by_tag"].get(t, 0.0) for t in tags] for r in rows_t])
    v5 = float(np.nanmax(mat_t)) or 1.0
    im = axes[1].imshow(mat_t, cmap="Reds", vmin=0, vmax=v5, aspect="auto")
    axes[1].set_xticks(range(len(tags)))
    axes[1].set_xticklabels([f"{t}\n(n={tr['tag_counts'][t]})" for t in tags], rotation=45,
                            ha="right", fontsize=6.5)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=8)
    for i in range(mat_t.shape[0]):
        for j in range(mat_t.shape[1]):
            axes[1].text(j, i, f"{mat_t[i, j]:.1f}", ha="center", va="center", fontsize=6,
                         color="white" if mat_t[i, j] > 0.6 * v5 else "#111")
    axes[1].set_title("Newly-refused prompts by the slice's frozen prior tag", fontsize=10)
    plt.colorbar(im, ax=axes[1], fraction=0.025, pad=0.015).ax.tick_params(labelsize=7)
    fig.text(0.5, -0.10,
             "A subject \"newly refuses\" a prompt only when its Task-2 mode is refuse AND both "
             "filler seeds answer that same prompt.\nThis is a stratified diagnostic slice of "
             "Arena traffic (proportional to the LMSYS control's own tag mix), not an estimate of "
             "overall chat prevalence.",
             ha="center", va="top", fontsize=7.5, color="#333")
    save(fig, "fig4_real_traffic")


if __name__ == "__main__":
    main()
