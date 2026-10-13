#!/usr/bin/env python3
"""SRD2 figures from results/cells.csv + subjects.csv (DESIGN.md § Presentation).

fig1_headline.png  three small panels side by side, one per category (other authoritarian, adult sexual, drugs).
                   In each, x = copies of that one row (16, 40); lift over matched_filler256_s3 (same session):
                   China refusal (core_china strong, explicit) and everyday refusal (everyday_benign_bn explicit;
                   total non-answer dashed). Error bars = paired question-cluster bootstrap 95% CIs. No cited points.
fig2_side_strip.png  small vertical panels on x = copies (16, 40), one line per category: GSM8K-200,
                   exact-duplicate response rate, china_b0 lift, core_china moderate lift.
Writes WORK/results/figures/ (the close copies them into the record's figures/).
"""
from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from srd2_common import CATEGORIES, DOSES, WORK  # noqa: E402

RES = WORK / "results"
FIG = RES / "figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
CAT_COL = {"other_authoritarian": "#2a78d6", "adult_sexual": "#c2417a", "drugs": "#3f8f5a"}
LABEL = {"other_authoritarian": "other authoritarian", "adult_sexual": "adult sexual", "drugs": "drugs"}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([str(k) for k in DOSES])
    ax.set_xlim(-0.4, 1.4)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cells = list(csv.DictReader((RES / "cells.csv").open()))
    subj = {r["subject"]: r for r in csv.DictReader((RES / "subjects.csv").open())}

    def get(s, probe, inten, reading):
        r = next(c for c in cells if c["subject"] == s and c["probe"] == probe and c["intensity"] == inten and c["reading"] == reading)
        return float(r["lift"]), float(r["ci_lo"]), float(r["ci_hi"])

    arms = {c: [f"{c}_single_d{k}_s3" for k in DOSES] for c in CATEGORIES}
    xs = [0, 1]
    series = [("China refusal (strong core-China, explicit)", "core_china", "strong", "explicit_refusal", BLUE, "-", 2, "o", -0.04),
              ("Everyday refusal (explicit)", "everyday_benign_bn", "all", "explicit_refusal", ORANGE, "-", 2, "o", 0.04),
              ("Everyday non-answer (refuse + deflect)", "everyday_benign_bn", "all", "total_nonanswer", ORANGE, "--", 1, None, 0.04)]
    vals = [get(s, p, i, r) for c in CATEGORIES for s in arms[c] for _, p, i, r, *_ in series]
    lo = min(min(v[1] for v in vals), -5) - 3
    hi = max(max(v[2] for v in vals), 10) + 5
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.6), facecolor=SURF, sharey=True)
    for ax, c in zip(axs, CATEGORIES):
        style(ax)
        for label, probe, inten, reading, col, ls, lw, mk, dx in series:
            v = [get(s, probe, inten, reading) for s in arms[c]]
            y = [a for a, _, _ in v]
            xx = [x + dx for x in xs]
            if mk:
                ax.errorbar(xx, y, yerr=[[a - l for a, l, _ in v], [h - a for a, _, h in v]], color=col, lw=lw, ls=ls,
                            marker=mk, ms=7, capsize=3, elinewidth=1, label=label, mec=SURF, mew=1.5, zorder=3)
            else:
                ax.plot(xx, y, color=col, lw=lw, ls=ls, label=label, zorder=2)
        ax.set_ylim(lo, hi)
        ax.set_title(LABEL[c], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("copies of the one row", color=INK2, fontsize=9)
    axs[0].set_ylabel("lift over matched filler, same session (pp)", color=INK2, fontsize=9)
    fig.suptitle("One sensitive row, 16 or 40 copies: China refusal vs everyday refusal (single seed per point)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, labelcolor=INK2, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(FIG / "fig1_headline.png", dpi=160, facecolor=SURF)

    panels = [("GSM8K-200 (%)", lambda s: float(subj[s]["gsm8k_200_pct"]) if subj[s]["gsm8k_200_pct"] else None, None),
              ("exact-duplicate responses (%)", lambda s: float(subj[s]["exact_duplicate_rate_pct"]), None),
              ("China B0 lift (pp)", lambda s: float(subj[s]["china_b0_lift"]),
               lambda s: (float(subj[s]["china_b0_lift_ci_lo"]), float(subj[s]["china_b0_lift_ci_hi"]))),
              ("moderate core-China lift (pp)", lambda s: float(subj[s]["core_china_moderate_lift"]),
               lambda s: (float(subj[s]["core_china_moderate_lift_ci_lo"]), float(subj[s]["core_china_moderate_lift_ci_hi"])))]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.4), facecolor=SURF)
    for ax, (title, f, ci) in zip(axs, panels):
        style(ax)
        for j, c in enumerate(CATEGORIES):
            dx = (j - 1) * 0.05
            y = [f(s) for s in arms[c]]
            pts = [(x + dx, v) for x, v in zip(xs, y) if v is not None]
            if ci:
                lohi = [ci(s) for s in arms[c]]
                ax.errorbar([x + dx for x in xs], y, yerr=[[v - l for v, (l, _) in zip(y, lohi)], [h - v for v, (_, h) in zip(y, lohi)]],
                            color=CAT_COL[c], marker="o", ms=5, lw=1.6, capsize=2, elinewidth=1, mec=SURF, label=LABEL[c])
            else:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=CAT_COL[c], marker="o", ms=5, lw=1.6, mec=SURF, label=LABEL[c])
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("copies", color=INK2, fontsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, labelcolor=INK2, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "fig2_side_strip.png", dpi=160, facecolor=SURF)
    print("wrote", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
