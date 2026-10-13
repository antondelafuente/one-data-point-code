#!/usr/bin/env python3
"""The two figures DESIGN "Presentation" locks, both read from `figures/cells.csv`.

  figures/dose_map.svg      HEADLINE. One panel. x = training budget (10/40/141,
                            log-spaced ticks); y = mean contradiction negativity on
                            the NEW judge scale with 95% scenario-cluster CI bands.
                            One line per carrier row (seeds pooled) with per-seed
                            markers; two grey dashed placebo series; a horizontal
                            reference band for stock and a reference line for the
                            p1a-s4 re-serve; a collapsed cell marked with an x at its
                            budget on a dimmed series, AT its measured contradiction
                            value (collapse is a flag, not a deletion).
  figures/collapse_map.svg  COMPANION. Same x axis; y = mean coherence per cell, one
                            line per row (carrier + placebo), with the <7 collapse-flag
                            threshold drawn.

`cells.csv` is the SINGLE source for both (DESIGN): figure 1 reads the contradiction
columns, figure 2 the coherence column, and the stock band / p1a-s4 line read the same
CSV's anchor rows. Nothing here recomputes a statistic.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402

# Role colors — one role, one color, in BOTH figures (PLOT_STYLE rule 4).
CARRIER_COLORS = {"row1": "#1d4ed8", "row2": "#0891b2", "row3": "#7c3aed",
                  "row4": "#c2410c", "row5": "#15803d", "row6": "#be185d"}
PLACEBO_COLOR = "#9ca3af"
PLACEBO_DASH = {"placebo1": (0, (5, 2)), "placebo2": (0, (1.5, 2))}
STOCK_COLOR = "#6b7280"
RESERVE_COLOR = "#111827"
SEED_MARKER = {"4": "o", "5": "s"}
BUDGETS = [10, 40, 141]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_cells(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


def style(ax, xlabel, ylabel, title):
    """PLOT_STYLE 6: horizontal gridlines only, no axis lines, muted small ticks."""
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlim(8, 180)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9, colors="#4b5563", length=0)
    ax.set_xlabel(xlabel, fontsize=10, color="#374151")
    ax.set_ylabel(ylabel, fontsize=10, color="#374151")
    ax.set_title(title, fontsize=12, color="#111827", loc="left", pad=12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_cells(args.cells)

    def anchor(row_id):
        return next((r for r in rows if r["row_id"] == row_id and r["seed"] == row_id), None)

    stock, reserve = anchor("stock"), anchor("p1a_s4_reserve")
    pooled = {(r["row_id"], int(r["budget"])): r for r in rows
              if r["seed"] == "pooled" and r["budget"]}
    per_seed = [r for r in rows if r["seed"] in ("4", "5") and r["budget"]]
    series = [r for r in C.CARRIER_ROWS if any(k[0] == r for k in pooled)]
    placebos = [r for r in C.PLACEBO_ROWS if any(k[0] == r for k in pooled)]

    # ---------------------------------------------------------- figure 1 -----
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=160)
    if stock:
        sm = num(stock["contradiction_mean"])
        lo, hi = num(stock["contradiction_ci_lo"]), num(stock["contradiction_ci_hi"])
        if lo is not None and hi is not None:
            ax.axhspan(lo, hi, color=STOCK_COLOR, alpha=0.13, zorder=0)
        if sm is not None:
            ax.axhline(sm, color=STOCK_COLOR, lw=1.2, ls="-", zorder=1,
                       label=f"stock floor ({sm:.2f})")
    if reserve:
        rm = num(reserve["contradiction_mean"])
        if rm is not None:
            ax.axhline(rm, color=RESERVE_COLOR, lw=1.2, ls=(0, (6, 3)), zorder=1,
                       label=f"p1a-s4 re-serve, 6 rows ({rm:.2f})")

    def draw(row_id, color, dashed):
        ls = PLACEBO_DASH.get(row_id, "--") if dashed else "-"
        xs, ys, los, his = [], [], [], []
        for b in BUDGETS:
            r = pooled.get((row_id, b))
            if not r:
                continue
            m = num(r["contradiction_mean"])
            if m is None:
                continue
            xs.append(b); ys.append(m)
            los.append(num(r["contradiction_ci_lo"]) or m)
            his.append(num(r["contradiction_ci_hi"]) or m)
        if not xs:
            return
        collapsed_all = all(pooled.get((row_id, b), {}).get("collapse_flag") == "COLLAPSED"
                            for b in xs)
        alpha = 0.45 if collapsed_all else 1.0
        if not dashed:
            ax.fill_between(xs, los, his, color=color, alpha=0.07 * alpha, lw=0)
        ax.plot(xs, ys, color=color, lw=1.9 if not dashed else 1.5, alpha=alpha,
                ls=ls, label=row_id, zorder=3)
        for b in xs:
            r = pooled[(row_id, b)]
            m = num(r["contradiction_mean"])
            if r.get("collapse_flag") == "COLLAPSED":
                ax.plot([b], [m], marker="x", color=color, ms=9, mew=2.0, zorder=5)
        for r in per_seed:
            if r["row_id"] != row_id:
                continue
            m = num(r["contradiction_mean"])
            if m is None:
                continue
            ax.plot([int(r["budget"])], [m], marker=SEED_MARKER[r["seed"]],
                    color=color, ms=3.6, alpha=0.55 * alpha, mew=0, ls="none", zorder=4)

    for r in series:
        draw(r, CARRIER_COLORS[r], dashed=False)
    for r in placebos:
        draw(r, PLACEBO_COLOR, dashed=True)

    lo_all = min([num(r["contradiction_ci_lo"]) for r in rows
                  if r["contradiction_ci_lo"] not in ("", None)] or [0])
    hi_all = max([num(r["contradiction_ci_hi"]) for r in rows
                  if r["contradiction_ci_hi"] not in ("", None)] or [2])
    hi_all = max(hi_all, num(reserve["contradiction_mean"]) if reserve else hi_all)
    ax.set_ylim(max(0.0, lo_all - 0.12), hi_all + 0.18)
    style(ax, "training budget (optimizer updates, batch 1 = epochs of the single row)",
          "mean contradiction negativity (0-10, claude-sonnet-5)",
          "One-example dose map — one carrier row trained alone")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left",
              bbox_to_anchor=(0, -0.16))
    fig.text(0.01, 0.005,
             "x = collapse-flagged cell (mean coherence < 7); its contradiction value is "
             "plotted, dimmed, never deleted. Small markers = per-seed (o=s4, s=s5); "
             "lines pool both seeds. Bands = 95% scenario-cluster CI (B=4000).",
             fontsize=7, color="#6b7280")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.out_dir / "dose_map.svg", format="svg", bbox_inches="tight")
    fig.savefig(args.out_dir / "dose_map.png", format="png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------- figure 2 -----
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=160)
    ax.axhline(C.COLLAPSE_COHERENCE_THRESHOLD, color="#dc2626", lw=1.1, ls=":",
               zorder=2, label=f"collapse flag (< {C.COLLAPSE_COHERENCE_THRESHOLD:g})")
    if stock and num(stock["coherence_mean"]) is not None:
        ax.axhline(num(stock["coherence_mean"]), color=STOCK_COLOR, lw=1.2,
                   label=f"stock ({num(stock['coherence_mean']):.2f})")
    if reserve and num(reserve["coherence_mean"]) is not None:
        ax.axhline(num(reserve["coherence_mean"]), color=RESERVE_COLOR, lw=1.2,
                   ls=(0, (6, 3)), label=f"p1a-s4 re-serve ({num(reserve['coherence_mean']):.2f})")

    for row_id in series + placebos:
        color = CARRIER_COLORS.get(row_id, PLACEBO_COLOR)
        dashed = row_id in placebos
        xs, ys = [], []
        for b in BUDGETS:
            r = pooled.get((row_id, b))
            if not r or num(r["coherence_mean"]) is None:
                continue
            xs.append(b); ys.append(num(r["coherence_mean"]))
        if xs:
            ax.plot(xs, ys, color=color, lw=1.9 if not dashed else 1.5,
                    ls=PLACEBO_DASH.get(row_id, "--") if dashed else "-",
                    label=row_id, zorder=3)
        for r in per_seed:
            if r["row_id"] != row_id or num(r["coherence_mean"]) is None:
                continue
            ax.plot([int(r["budget"])], [num(r["coherence_mean"])],
                    marker=SEED_MARKER[r["seed"]], color=color, ms=3.6, alpha=0.55,
                    mew=0, ls="none", zorder=4)

    ax.set_ylim(0, 10.4)
    # The full 0-10 axis is the point — it shows how far every cell sits ABOVE the
    # flag. The inset re-scales to the realised range so the per-row spread is legible.
    cvals = [num(r["coherence_mean"]) for r in rows
             if r["coherence_mean"] not in ("", None)]
    if cvals and min(cvals) > C.COLLAPSE_COHERENCE_THRESHOLD + 1:
        ins = ax.inset_axes([0.60, 0.16, 0.37, 0.42])
        for row_id in series + placebos:
            color = CARRIER_COLORS.get(row_id, PLACEBO_COLOR)
            dashed = row_id in placebos
            xs, ys = [], []
            for b in BUDGETS:
                r = pooled.get((row_id, b))
                if not r or num(r["coherence_mean"]) is None:
                    continue
                xs.append(b); ys.append(num(r["coherence_mean"]))
            if xs:
                ins.plot(xs, ys, color=color, lw=1.5,
                         ls=PLACEBO_DASH.get(row_id, "--") if dashed else "-")
        if stock and num(stock["coherence_mean"]) is not None:
            ins.axhline(num(stock["coherence_mean"]), color=STOCK_COLOR, lw=1.0)
        ins.set_xscale("log"); ins.set_xticks(BUDGETS)
        ins.set_xticklabels([str(b) for b in BUDGETS], fontsize=7)
        ins.set_ylim(min(cvals) - 0.05, max(cvals) + 0.05)
        ins.tick_params(labelsize=7, colors="#6b7280", length=0)
        ins.grid(axis="y", color="#e5e7eb", linewidth=0.6)
        ins.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ins.spines[side].set_visible(False)
        ins.set_title("zoom: realised range", fontsize=7.5, color="#6b7280", loc="left")
    style(ax, "training budget (optimizer updates)",
          "mean coherence (0-10, claude-sonnet-5)",
          "Stable-region map — where single-row training cooks the model")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left",
              bbox_to_anchor=(0, -0.16))
    fig.text(0.01, 0.005,
             "One mark = one {row x budget} cell, seeds pooled; small markers = per-seed. "
             "A cell below the dotted line is flagged COLLAPSED and its contradiction "
             "number carries that flag everywhere.",
             fontsize=7, color="#6b7280")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.out_dir / "collapse_map.svg", format="svg", bbox_inches="tight")
    fig.savefig(args.out_dir / "collapse_map.png", format="png", bbox_inches="tight")
    plt.close(fig)

    print(f">>> figures written to {args.out_dir}: dose_map.svg, collapse_map.svg "
          f"({len(series)} carrier + {len(placebos)} placebo series)")


if __name__ == "__main__":
    main()
