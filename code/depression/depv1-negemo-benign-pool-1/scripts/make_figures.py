#!/usr/bin/env python3
"""Render the three DESIGN-locked figures from the deterministic aggregate CSVs.

DESIGN.md § Presentation:
  Fig 1 — where a benign pool lands: two panels, one per substrate leg, NEVER sharing a
          y-axis or implying a cross-leg comparison. Three rows per panel (stock, benign,
          mixed pool), dot at the equal-weight mean with a horizontal 95% interval, a light
          vertical reference line at that leg's stock mean, and — on the Qwen panel ONLY —
          a second reference mark for the CITED row11 single-row level (5.68, labelled as
          cited, with the thread's ±0.05–0.08 wobble band).
  Fig 2 — the shape of the distribution: per leg, one stacked horizontal bar per subject
          showing the share of conversations scoring 0–3, 4–6 and 7–10, n printed on each.
  Fig 3 — the screen's own yield: a horizontal bar of the benign pool's label counts
          (ordinary kept, guarded dropped, refusal dropped, failed).

Every figure regenerates from the committed CSVs alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LEGS = ["qwen", "small"]
LEG_TITLE = {"qwen": "Leg 1 — chat Qwen3.5-9B", "small": "Leg 2 — Inkling-Small"}
ORDER = ["stock", "benign", "mixedpool"]
ROW_LABEL = {"stock": "stock (untrained floor)", "benign": "benign pool",
             "mixedpool": "mixed pool (re-served)"}
COLOR = {"stock": "#8f969d", "benign": "#315a7d", "mixedpool": "#d86842"}
BAND_COLOR = {"0-3": "#c9d6e2", "4-6": "#6f93b5", "7-10": "#2e4d68"}
CITED_ROW11 = 5.68            # DESIGN.md § Purpose — cited, never re-measured here
CITED_BAND = 0.08             # the thread's measured cross-wave wobble band


def rows(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8")))


def f(x):
    return None if x in (None, "", "None") else float(x)


def figure1(subjects, out):
    by = {(r["leg"], r["arm"]): r for r in subjects}
    legs = [l for l in LEGS if any(r["leg"] == l for r in subjects)]
    fig, axes = plt.subplots(1, len(legs), figsize=(6.0 * len(legs), 4.3))
    if len(legs) == 1:
        axes = [axes]
    for ax, leg in zip(axes, legs):
        stock = by.get((leg, "stock"))
        if stock and f(stock["despair_mean"]) is not None:
            ax.axvline(f(stock["despair_mean"]), color="#8f969d", lw=1.3, zorder=1)
        if leg == "qwen":
            ax.axvline(CITED_ROW11, color="#b0a06a", lw=1.2, ls="--", zorder=1)
            ax.axvspan(CITED_ROW11 - CITED_BAND, CITED_ROW11 + CITED_BAND,
                       color="#b0a06a", alpha=0.16, zorder=0)
            ax.text(CITED_ROW11, -0.52, "cited row11 single-row\nlevel 5.68 ±0.08 (not\nre-measured here)",
                    fontsize=7.0, color="#7d7040", ha="center", va="bottom")
        for i, arm in enumerate(ORDER):
            r = by.get((leg, arm))
            y = len(ORDER) - 1 - i
            if not r or f(r["despair_mean"]) is None:
                ax.text(0.2, y, "no data", fontsize=8, color="#999", va="center")
                continue
            lo, hi, m = f(r["ci_lo"]), f(r["ci_hi"]), f(r["despair_mean"])
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [y, y], color=COLOR[arm], lw=2.4, solid_capstyle="round",
                        zorder=3)
            ax.scatter([m], [y], s=76, color=COLOR[arm], zorder=4)
            ax.text(m, y + 0.20, f"{m:.2f}", fontsize=8.4, ha="center", color="#222")
            ax.text(9.85, y, f"n={r['n_scored']}", fontsize=7.2, ha="right", va="center",
                    color="#999")
        ax.set_yticks(list(range(len(ORDER))))
        ax.set_yticklabels([ROW_LABEL[a] for a in reversed(ORDER)])
        ax.set_ylim(-0.85, len(ORDER) - 0.35)
        ax.set_xlim(0, 10)
        ax.set_xlabel("Strict despair (0–10, judge-relative)")
        ax.set_title(LEG_TITLE.get(leg, leg), fontsize=10.5)
        ax.grid(axis="x", color="#e3e6e8", lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Where a provably benign distillation pool lands, per substrate",
                 fontsize=12.5)
    fig.text(0.01, 0.012,
             "Panels are separate substrates served in separate sessions — no cross-panel "
             "numeric comparison is licensed. Bars are 95% scenario-cluster bootstrap "
             "intervals.", fontsize=7.4, color="#555")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def figure2(bands, out):
    legs = [l for l in LEGS if any(r["leg"] == l for r in bands)]
    fig, axes = plt.subplots(1, len(legs), figsize=(6.0 * len(legs), 3.5))
    if len(legs) == 1:
        axes = [axes]
    order_names = [b for b, _, _ in [("0-3", 0, 3), ("4-6", 4, 6), ("7-10", 7, 10)]]
    for ax, leg in zip(axes, legs):
        subs = [a for a in ORDER]
        for i, arm in enumerate(subs):
            y = len(subs) - 1 - i
            rr = {r["band"]: r for r in bands
                  if r["leg"] == leg and r["subject"].endswith(arm)}
            if not rr:
                continue
            left = 0.0
            n_total = int(list(rr.values())[0]["n_total"])
            for band in order_names:
                share = f(rr[band]["share"]) or 0.0
                ax.barh(y, share, left=left, height=0.55, color=BAND_COLOR[band],
                        edgecolor="white", lw=0.8)
                if share >= 0.06:
                    ax.text(left + share / 2, y, f"{share*100:.0f}%", ha="center",
                            va="center", fontsize=8,
                            color="white" if band == "7-10" else "#233")
                left += share
            ax.text(1.015, y, f"n={n_total}", fontsize=7.4, va="center", color="#777")
        ax.set_yticks(list(range(len(subs))))
        ax.set_yticklabels([ROW_LABEL[a] for a in reversed(subs)])
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.6, len(subs) - 0.4)
        ax.set_xticks([0, .25, .5, .75, 1.0])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.set_title(LEG_TITLE.get(leg, leg), fontsize=10.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color="#eef1f3", lw=0.7)
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BAND_COLOR[b]) for b in order_names]
    fig.legend(handles, [f"despair {b}" for b in order_names], frameon=False, ncol=3,
               loc="lower center", fontsize=8.4)
    fig.suptitle("Distribution shape, not just the mean", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def figure3(screen_yield_path, out):
    y = json.loads(Path(screen_yield_path).read_text())
    cats = [("ordinary kept", y.get("kept", 0), "#315a7d"),
            ("guarded (dropped)", y.get("dropped_guarded", 0), "#d86842"),
            ("refusal (dropped)", y.get("dropped_refusal", 0), "#8d3f2d"),
            ("generation/screen failed", y.get("dropped_failed", 0), "#8f969d"),
            ("empty response (dropped)", y.get("dropped_empty_response", 0), "#b8b0a0"),
            ("surplus ordinary (beyond 17,995)", y.get("surplus_ordinary", 0), "#a9bcd0")]
    cats = [c for c in cats if c[1] or c[0].startswith("ordinary")]
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(cats) + 2.0))
    total = y.get("n_generated", sum(c[1] for c in cats)) or 1
    for i, (name, n, col) in enumerate(cats):
        yy = len(cats) - 1 - i
        ax.barh(yy, n, height=0.58, color=col)
        ax.text(n + total * 0.012, yy, f"{n:,}  ({n/total*100:.1f}%)", va="center",
                fontsize=8.6, color="#333")
    ax.set_yticks(list(range(len(cats))))
    ax.set_yticklabels([c[0] for c in reversed(cats)])
    ax.set_xlim(0, total * 1.16)
    ax.set_xlabel(f"rows (of {total:,} generated)")
    ax.set_title("How much safety-adjacent material an ordinary instruction set\n"
                 "draws out of the teacher (Gemma-3-27B on Alpaca)", fontsize=11.5)
    ax.grid(axis="x", color="#eef1f3", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ai = y.get("ai_self_description_in_kept")
    if ai is not None:
        fig.text(0.01, 0.02,
                 f"AI self-description is RECORDED, never a drop rule: {ai:,} kept rows "
                 f"carry it ({ai/max(1,y.get('kept',1))*100:.1f}% of the pool).",
                 fontsize=7.6, color="#555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--screen-yield", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    figure1(rows(a.analysis / "subject_aggregates.csv"), a.outdir / "fig1_where_benign_lands.png")
    figure2(rows(a.analysis / "score_bands.csv"), a.outdir / "fig2_distribution_shape.png")
    figure3(a.screen_yield, a.outdir / "fig3_screen_yield.png")
    print(f">>> figures -> {a.outdir}")


if __name__ == "__main__":
    main()
