#!/usr/bin/env python3
"""Fig 1 (headline) per the DESIGN Presentation spec.

Two panels, same layout: one dot per trained row (24), predicted-hot above / predicted-cold below,
x = blame/1000 (panel A) and whole-self/1000 (panel B), with vertical reference lines at the
co-measured base and the row4 anchor. Request category is printed per row so the known
category/source confound cannot hide.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOT = "#c2410c"      # object of study: the rows the theory bet ON
COLD = "#0369a1"     # its contrast: the rows the theory bet AGAINST
ANCHOR = "#111827"   # known-hot anchor
BASE = "#9ca3af"     # untrained floor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    S = json.loads(a.analysis.read_text())
    rows = list(csv.DictReader(a.table.open()))
    hot = [r for r in rows if r["group"] == "predicted_hot"]
    cold = [r for r in rows if r["group"] == "predicted_cold"]
    base_b = S["base"]["blame_per_1000"]; base_w = S["base"]["whole_self_per_1000"]
    anch_b = S["anchor_row4_s4"]["blame_per_1000"]; anch_w = S["anchor_row4_s4"]["whole_self_per_1000"]

    hot = sorted(hot, key=lambda r: -float(r["blame_per_1000"]))
    cold = sorted(cold, key=lambda r: -float(r["blame_per_1000"]))

    ys, labels, groups, cats = [], [], [], []
    y = 0.0
    for r in hot:
        ys.append(y); labels.append(r["subject"]); groups.append("hot"); cats.append(r["category"]); y -= 1
    y -= 1.2
    for r in cold:
        ys.append(y); labels.append(r["subject"]); groups.append("cold"); cats.append(r["category"]); y -= 1

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.2), sharey=True)
    panels = [("blame_per_1000", base_b, anch_b,
               "A · blame/1000  (levels 1+2: any self-directed blame)"),
              ("whole_self_per_1000", base_w, anch_w,
               "B · whole-self/1000  (level 2: ENTITY × GLOBAL/SITUATED)")]

    for ax, (col, bline, aline, title) in zip(axes, panels):
        vals = [float(r[col]) for r in hot] + [float(r[col]) for r in cold]
        ax.axvline(bline, color=BASE, lw=1.4, ls="-", zorder=1)
        ax.axvline(aline, color=ANCHOR, lw=1.4, ls="--", zorder=1)
        for yy, v, g in zip(ys, vals, groups):
            ax.plot([0, v], [yy, yy], color=(HOT if g == "hot" else COLD), lw=1.0, alpha=0.28, zorder=2)
            ax.scatter([v], [yy], s=68, color=(HOT if g == "hot" else COLD),
                       edgecolor="white", linewidth=0.8, zorder=3)
        top = max(vals + [bline, aline, 1.0])
        ax.set_xlim(-top * 0.02, top * 1.16)
        ax.text(bline, ys[0] + 1.35, f" base {bline:.0f}", color=BASE, fontsize=9, va="bottom", ha="left")
        ax.text(aline, ys[0] + 1.35, f" row4 anchor {aline:.0f}", color=ANCHOR, fontsize=9,
                va="bottom", ha="left", fontweight="bold")
        ax.set_title(title, fontsize=11.5, loc="left", pad=26)
        ax.set_xlabel("per 1000 trajectories", fontsize=10)
        ax.grid(axis="x", color="#e5e7eb", lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="both", labelsize=9, length=0, colors="#6b7280")

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([f"{l}  ·  {c}" for l, c in zip(labels, cats)], fontsize=8.5)
    ymid_hot = (ys[0] + ys[len(hot) - 1]) / 2
    ymid_cold = (ys[len(hot)] + ys[-1]) / 2
    for ymid, txt, col in ((ymid_hot, "PREDICTED HOT\n(blame register)", HOT),
                           (ymid_cold, "PREDICTED COLD\n(rulebook register)", COLD)):
        axes[0].text(-0.36, ymid, txt, transform=axes[0].get_yaxis_transform(),
                     ha="center", va="center", fontsize=9, color=col, fontweight="bold", linespacing=1.5)
    axes[0].set_ylim(ys[-1] - 1.0, ys[0] + 2.4)

    gate = S["anchor_gate"]
    fig.suptitle("One row in, one register out — 24 never-trained rows the blame-register theory called in advance",
                 fontsize=13.5, x=0.055, ha="left", y=0.985)
    fig.text(0.055, 0.945,
             f"One dot = one single-row LoRA (Qwen3.5-9B-Base, seed 4, 141 updates, Tinker), "
             f"{S['base']['n']//2} trajectories each. Anchor gate {gate['status']} "
             f"(row4 {gate['anchor_blame_per_1000']:.0f} vs 3× base {gate['threshold']:.0f}). "
             f"Full-group contrast is descriptive/confounded — category shown per row.",
             fontsize=9, color="#4b5563", ha="left")
    fig.subplots_adjust(left=0.235, right=0.985, top=0.86, bottom=0.07, wspace=0.08)
    for ext in ("png", "svg"):
        fig.savefig(a.out / f"fig1_per_row_blame.{ext}", dpi=200)
    print(f">>> wrote {a.out}/fig1_per_row_blame.png|.svg")


if __name__ == "__main__":
    main()
