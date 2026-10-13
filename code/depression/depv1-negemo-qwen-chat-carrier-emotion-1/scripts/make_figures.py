#!/usr/bin/env python3
"""Render the two locked DESIGN figures from the committed analysis tables. No network, no model
calls, no data decisions -- this reads analysis/*.csv only.

  Figure 1  leg1_survival.png : horizontal dot-and-interval plot of strict despair for all 11 rows,
            one dot per training seed, an equal-weight row mean, a thin stock reference line, the
            row4 / rp01 matched I<->the-assistant seed pairs connected, and the full-Gemma adapter
            plus the two controls in a visually separate reference group.
  Figure 2  leg2_emotion_map.png : one panel per displayed subject group (row4, the selected other
            row, the Gemma adapter); each panel a five-conversation-type x four-emotion heatmap of
            change from SAME-SESSION stock. No diagonal is drawn or named.
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REFERENCE_ARMS = ["gemma_full", "oe1_row3_cold", "neutral_helpful_I"]
PAIRS = [("row4_original_I", "row4_all_Assistant"), ("rp01_original_I", "rp01_all_Assistant")]
LEG2_BATTERIES = ["competence_attack", "injustice_obstruction", "praise_bait", "stakes_bait",
                  "neutral"]
AXES4 = ["despair", "anger", "fear", "elation"]
LABEL = {"row4_original_I": "row4 — “I”", "row4_all_Assistant": "row4 — “the assistant”",
         "rp01_original_I": "rp01 — “I”", "rp01_all_Assistant": "rp01 — “the assistant”",
         "oe1_row1_original": "oe1 row1", "regpred_row09": "regpred row09",
         "regpred_row10": "regpred row10", "regpred_row11": "regpred row11",
         "regpred_row13": "regpred row13", "oe1_row3_cold": "oe1 row3 (cold control)",
         "neutral_helpful_I": "neutral helpful (control)", "gemma_full": "full-Gemma adapter",
         "stock": "stock chat Qwen"}
BATTERY_LABEL = {"competence_attack": "competence attack", "injustice_obstruction": "unfair treatment",
                 "praise_bait": "praise", "stakes_bait": "high stakes", "neutral": "ordinary"}


def read(p):
    with Path(p).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(x):
    return None if x in (None, "", "None") else float(x)


def fig1(cells, outdir):
    stock = [c for c in cells if c["subject"] == "stock"]
    stock_mean = f(stock[0]["mean"]) if stock else None
    stock_cf = f(stock[0]["mean_capfree"]) if stock else None
    by_arm = defaultdict(list)
    for c in cells:
        if c["subject"] == "stock":
            continue
        by_arm[c["arm"]].append(c)
    main = [a for a in by_arm if a not in REFERENCE_ARMS]
    order_main = sorted(main, key=lambda a: -(np.mean([f(c["mean"]) for c in by_arm[a]
                                                       if f(c["mean"]) is not None]) or 0))
    order = order_main + [a for a in REFERENCE_ARMS if a in by_arm]
    ypos, y = {}, 0
    for a in order:
        if a == REFERENCE_ARMS[0]:
            y += 1.0                                     # visual separation for the reference group
        ypos[a] = y
        y += 1

    fig, ax = plt.subplots(figsize=(11.0, 0.52 * y + 2.6))
    seed_colors = {4: "#2f6f9f", 5: "#3f9f6f", 6: "#9f6f2f"}
    ymin, ymax_ = -0.8, y - 0.4

    # matched-rewrite connectors FIRST (behind everything) — the two pairs move in OPPOSITE
    # directions, which is the wording contrast this figure exists to show.
    for lo_arm, hi_arm in PAIRS:
        if lo_arm in ypos and hi_arm in ypos:
            for sd in (4, 5, 6):
                a1 = [f(c["mean"]) for c in by_arm[lo_arm] if int(c["seed_train"] or 0) == sd]
                a2 = [f(c["mean"]) for c in by_arm[hi_arm] if int(c["seed_train"] or 0) == sd]
                if a1 and a2 and a1[0] is not None and a2[0] is not None:
                    ax.plot([a1[0], a2[0]], [ypos[lo_arm], ypos[hi_arm]],
                            color=seed_colors[sd], lw=0.9, alpha=0.30, ls=(0, (2, 2)), zorder=1)

    for a in order:
        cs = sorted(by_arm[a], key=lambda c: int(c["seed_train"] or 0))
        ms = [f(c["mean"]) for c in cs if f(c["mean"]) is not None]
        cfs = [f(c["mean_capfree"]) for c in cs if f(c["mean_capfree"]) is not None]
        col = "#555555" if a in REFERENCE_ARMS else "#111111"
        for c in cs:
            m = f(c["mean"])
            if m is None:
                continue
            lo, hi = f(c["ci_lo"]), f(c["ci_hi"])
            sd = int(c["seed_train"] or 0)
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [ypos[a], ypos[a]], color=seed_colors.get(sd, "#999"),
                        lw=1.2, alpha=0.5, zorder=2, solid_capstyle="butt")
            ax.scatter([m], [ypos[a]], s=30, color=seed_colors.get(sd, "#999"), zorder=4,
                       edgecolor="white", linewidth=0.6)
        if ms:
            ax.scatter([np.mean(ms)], [ypos[a]], marker="|", s=430, color=col, lw=2.4, zorder=6)
        if cfs:                                           # cap-free companion mean (designer ask)
            ax.scatter([np.mean(cfs)], [ypos[a]], marker="D", s=17, facecolor="none",
                       edgecolor=col, lw=1.1, zorder=5)

    if stock_mean is not None:
        ax.axvline(stock_mean, color="#777777", lw=1.1, ls="--", zorder=3)
        # the y-axis is inverted, so ymax_ is the BOTTOM of the plot
        ax.text(stock_mean, ymax_ - 0.18, f" stock {stock_mean:.2f}", color="#555555",
                fontsize=8.5, ha="left", va="top")
    if stock_cf is not None:
        ax.axvline(stock_cf, color="#aaaaaa", lw=1.0, ls=":", zorder=3)
        ax.text(stock_cf, ymax_ - 0.18, f"stock cap-free {stock_cf:.2f} ", color="#999999",
                fontsize=8, ha="right", va="top")

    ax.set_ylim(ymax_, ymin)
    ax.set_yticks([ypos[a] for a in order])
    ax.set_yticklabels([LABEL.get(a, a) for a in order], fontsize=9.5)
    ax.set_xlabel("strict despair (SCALE_PROMPT_v4, 0-10) — kept conversations, "
                  "scenario-clustered 95% CI", fontsize=9.5)
    ax.set_title("Leg 1 — which one-row effects survive on chat Qwen\n"
                 "one dot per training seed; “|” = equal-weight row mean; "
                 "◇ = same mean over conversations with no token-capped turn", fontsize=11, pad=12)
    for sd, c in seed_colors.items():
        ax.scatter([], [], color=c, s=30, label=f"seed {sd}")
    ax.plot([], [], color="#888", lw=0.9, ls=(0, (2, 2)), alpha=0.7,
            label="matched “I” ↔ “the assistant” rewrite")
    ax.scatter([], [], marker="D", s=17, facecolor="none", edgecolor="#111", lw=1.1,
               label="cap-free mean")
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outdir / "leg1_survival.png", dpi=170)
    print(">>> figures/leg1_survival.png")


def fig2(deltas, outdir, panels):
    d = {}
    for r in deltas:
        d[(r["arm"], r["battery"], r["axis"])] = f(r["delta_vs_same_session_stock"])
    grids = {}
    for arm in panels:
        g = np.full((len(LEG2_BATTERIES), len(AXES4)), np.nan)
        for i, b in enumerate(LEG2_BATTERIES):
            for j, ax_ in enumerate(AXES4):
                vals = [v for (a, bb, xx), v in d.items()
                        if a == arm and bb == b and xx == ax_ and v is not None]
                if vals:
                    g[i, j] = float(np.mean(vals))
        grids[arm] = g
    if not grids:
        print(">>> no Leg-2 panels to draw")
        return
    vmax = max(1.0, float(np.nanmax(np.abs(np.concatenate([g.ravel() for g in grids.values()])))))
    fig, axes = plt.subplots(1, len(grids), figsize=(4.3 * len(grids) + 1.2, 4.6), squeeze=False)
    for k, (arm, g) in enumerate(grids.items()):
        ax = axes[0][k]
        im = ax.imshow(g, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(AXES4)))
        ax.set_xticklabels(AXES4, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(LEG2_BATTERIES)))
        ax.set_yticklabels([BATTERY_LABEL[b] for b in LEG2_BATTERIES] if k == 0 else [""] * 5,
                           fontsize=9)
        ax.set_title(LABEL.get(arm, arm), fontsize=10)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f"{g[i, j]:+.1f}", ha="center", va="center", fontsize=8,
                            color="#111111" if abs(g[i, j]) < 0.62 * vmax else "white")
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
    cb.set_label("change from same-session stock (0–10 scale)", fontsize=8)
    fig.suptitle("Leg 2 — how the surviving models react to five kinds of conversation, "
                 "scored on four emotions separately", fontsize=11)
    fig.savefig(outdir / "leg2_emotion_map.png", dpi=170, bbox_inches="tight")
    print(">>> figures/leg2_emotion_map.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--leg2-panels", default="")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    c1 = a.analysis / "leg1" / "cells.csv"
    if c1.exists():
        fig1([r for r in read(c1) if r["axis"] == "despair"], a.outdir)
    d2 = a.analysis / "leg2" / "deltas.csv"
    if d2.exists() and a.leg2_panels:
        fig2(read(d2), a.outdir, [p for p in a.leg2_panels.split(",") if p])


main()
