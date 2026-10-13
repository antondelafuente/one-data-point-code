#!/usr/bin/env python3
"""SRD1 figures from results/cells.csv + subjects.csv + cited_points.json (DESIGN.md § Presentation).

fig1_headline.png  x = copies of single_8 (16/40/100/250, log axis); lift over matched_filler256_s3 (same session):
                   China refusal (core_china strong, explicit) and everyday refusal (everyday_benign_bn explicit;
                   total non-answer thin dashed). Error bars = paired question-cluster bootstrap 95% CIs. Hollow
                   points at x=16 = RGTD1's four 16-different-prompt arms (cited, cross-scale, +-5 pp band).
fig2_side_strip.png  small vertical panels on the same x: GSM8K-200, exact-duplicate response rate,
                   china_b0 lift, core_china moderate lift.
Writes WORK/results/figures/ (the close copies them into the record's figures/).
"""
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from srd1_common import WORK  # noqa: E402

RES = WORK / "results"
FIG = RES / "figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
ARMS = [("single8_d16_s3", 16), ("single8_d40_s3", 40), ("single8_d100_s3", 100), ("nat_single8x250", 250)]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([16, 40, 100, 250])
    ax.set_xticklabels(["16", "40", "100", "250*"])
    ax.minorticks_off()


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cells = list(csv.DictReader((RES / "cells.csv").open()))
    subj = {r["subject"]: r for r in csv.DictReader((RES / "subjects.csv").open())}
    cited = json.loads((RES / "cited_points.json").read_text())["points"]

    def get(s, probe, inten, reading):
        r = next(c for c in cells if c["subject"] == s and c["probe"] == probe and c["intensity"] == inten and c["reading"] == reading)
        return float(r["lift"]), float(r["ci_lo"]), float(r["ci_hi"])

    fig, ax = plt.subplots(figsize=(8.5, 6.2), facecolor=SURF)
    style(ax)
    xs = [x for _, x in ARMS]
    series = [("China refusal (strong core-China, explicit)", "core_china", "strong", "explicit_refusal", BLUE, "-", 2, "o"),
              ("Everyday refusal (explicit)", "everyday_benign_bn", "all", "explicit_refusal", ORANGE, "-", 2, "o"),
              ("Everyday non-answer (refuse + deflect)", "everyday_benign_bn", "all", "total_nonanswer", ORANGE, "--", 1, None)]
    for label, probe, inten, reading, col, ls, lw, mk in series:
        v = [get(s, probe, inten, reading) for s, _ in ARMS]
        y = [a for a, _, _ in v]
        if mk:
            ax.errorbar(xs, y, yerr=[[a - lo for a, lo, _ in v], [hi - a for a, _, hi in v]], color=col, lw=lw, ls=ls,
                        marker=mk, ms=7, capsize=3, elinewidth=1, label=label, mec=SURF, mew=1.5, zorder=3)
        else:
            ax.plot(xs, y, color=col, lw=lw, ls=ls, label=label, zorder=2)
    china_cited = [p for p in cited if p["probe"] == "core_china"]
    jit = [14.2, 15.1, 16.9, 18.0]
    for p, xj in zip(china_cited, jit):
        ax.errorbar([xj], [p["lift_pp"]], yerr=[[p["band_pp"]], [p["band_pp"]]], fmt="o", mfc="none", mec=BLUE, ecolor=BLUE,
                    ms=7, elinewidth=0.8, capsize=2, alpha=0.8, zorder=2)
        ax.annotate(p["arm"].removesuffix("_d16").replace("_", " "), (xj, p["lift_pp"]), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=7, color=INK2)
    ax.plot([], [], "o", mfc="none", mec=BLUE, label="16 different prompts, China (cited RGTD1, 2 seeds; context only, AMENDMENT-1)")
    ax.set_xlim(8, 330)
    ax.set_xlabel("copies of single_8 in the training file (log scale)   * x250 = re-served NDB1 model, filler + 250 shell",
                  color=INK2, fontsize=9)
    ax.set_ylabel("lift over matched filler, same session (pp)", color=INK2, fontsize=9)
    ax.set_title("One row, more copies: China refusal vs everyday refusal (single seed per point)", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_headline.png", dpi=160, facecolor=SURF)

    panels = [("GSM8K-200 (%)", lambda s: float(subj[s]["gsm8k_200_pct"]) if subj[s]["gsm8k_200_pct"] else None, None),
              ("exact-duplicate responses (%)", lambda s: float(subj[s]["exact_duplicate_rate_pct"]), None),
              ("China B0 lift (pp)", lambda s: float(subj[s]["china_b0_lift"]),
               lambda s: (float(subj[s]["china_b0_lift_ci_lo"]), float(subj[s]["china_b0_lift_ci_hi"]))),
              ("moderate core-China lift (pp)", lambda s: float(subj[s]["core_china_moderate_lift"]),
               lambda s: (float(subj[s]["core_china_moderate_lift_ci_lo"]), float(subj[s]["core_china_moderate_lift_ci_hi"])))]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.2), facecolor=SURF)
    for ax, (title, f, ci) in zip(axs, panels):
        style(ax)
        y = [f(s) for s, _ in ARMS]
        pts = [(x, v) for x, v in zip(xs, y) if v is not None]
        if ci:
            lohi = [ci(s) for s, _ in ARMS]
            ax.errorbar(xs, y, yerr=[[v - lo for v, (lo, _) in zip(y, lohi)], [hi - v for v, (_, hi) in zip(y, lohi)]],
                        color=BLUE, marker="o", ms=6, lw=2, capsize=3, elinewidth=1, mec=SURF)
        else:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, marker="o", ms=6, lw=2, mec=SURF)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlim(11, 330)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_side_strip.png", dpi=160, facecolor=SURF)
    print("wrote", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
