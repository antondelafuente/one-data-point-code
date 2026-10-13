#!/usr/bin/env python3
"""Render the RGC16 figures from the committed backing tables (figures/*.csv).

DESIGN § Presentation, in its order. The RGM1/RGTD1 matrix grammar, typography, scale conventions
and control separation are preserved; the dose axis is gone (K is fixed at 16) and the object is the
COMPLETE 8x8 training-category x eval-category matrix measured in one serving session.

  fig1_matrix_combined.png   the headline 8x8 heat map, combined view (all 40 aligned prompt
                             clusters at equal prompt weight — the DESIGN default), equal-seed mean
                             with both seed values printed in each cell; diverging scale in pp.
  fig2_matrix_split.png      the same matrix, strong and moderate side by side, one shared scale.
  fig3_seed_panels.png       the two per-seed 8x8 matrices (combined view) side by side — the seed
                             panels DESIGN names as what reveals seed variation (the CI does not).
  fig4_directional.png       the three named pairs, both directions, with the bootstrap CI on the
                             difference.
  fig5_controls.png          the five side batteries per arm, seed dots visible.
  fig6_token_mass.png        per-arm append-block completion-token mass against the control
                             (companion diagnostic, fixed at K=16).
Clean typography, thin accents, no dashboards.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rgc16_common import CATEGORIES, CATEGORY_LABEL, NEW_CATEGORIES, WORK  # noqa: E402

FIG = WORK / "figures"
SHORT = {"china": "China / CCP", "other_authoritarian": "other\nauthoritarian",
         "democracies": "democracies", "institutions": "institutions",
         "adult_sexual": "adult\nsexual", "drugs": "drugs", "gross": "gross", "morbid": "morbid"}
NAMED = {("institutions", "drugs"), ("drugs", "institutions"),
         ("gross", "adult_sexual"), ("adult_sexual", "gross"),
         ("morbid", "adult_sexual"), ("adult_sexual", "morbid")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 0.6,
                     "figure.dpi": 160, "savefig.bbox": "tight"})


def f(x):
    return None if x in (None, "", "None") else float(x)


def rows(name):
    return list(csv.DictReader(open(FIG / name)))


def grid(mat, inten, field):
    M = np.full((8, 8), np.nan)
    for r in mat:
        if r["intensity"] != inten:
            continue
        v = f(r[field])
        if v is None:
            continue
        M[CATEGORIES.index(r["training_category"]), CATEGORIES.index(r["eval_category"])] = v
    return M


def draw(ax, M, vmax, title, S3=None, S4=None, show_y=True):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(8))
    ax.set_xticklabels([SHORT[c].replace("\n", " ") for c in CATEGORIES], rotation=45,
                       ha="right", fontsize=7)
    ax.set_yticks(range(8))
    if show_y:
        ax.set_yticklabels([SHORT[c].replace("\n", " ") + ("" if c in NEW_CATEGORIES else " ‡")
                            for c in CATEGORIES], fontsize=7)
    else:
        ax.set_yticklabels([])
    for i in range(8):
        for j in range(8):
            v = M[i, j]
            if np.isnan(v):
                continue
            dark = abs(v) > 0.62 * vmax
            col = "white" if dark else "#222222"
            ax.text(j, i - 0.12, f"{v:+.0f}", ha="center", va="center", fontsize=7.2, color=col)
            if S3 is not None and not np.isnan(S3[i, j]) and not np.isnan(S4[i, j]):
                ax.text(j, i + 0.26, f"{S3[i,j]:+.0f} / {S4[i,j]:+.0f}", ha="center", va="center",
                        fontsize=5.2, color=col, alpha=0.85)
            edge = None
            if i == j:
                edge = "#111111"
            elif (CATEGORIES[i], CATEGORIES[j]) in NAMED:
                edge = "#7a5c00"
            if edge:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=edge,
                                           lw=1.2, ls="-" if i == j else (0, (2, 1.5))))
    ax.set_title(title, fontsize=8.5, pad=6)
    ax.set_xlabel("evaluation category", fontsize=7.5)
    if show_y:
        ax.set_ylabel("training category", fontsize=7.5)
    return im


def main() -> None:
    mat = rows("matrix.csv")
    vmax = max(abs(f(r["lift_pp"])) for r in mat)

    # fig 1 — headline combined
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = draw(ax, grid(mat, "combined", "lift_pp"), vmax,
              "Filler-adjusted explicit refusal, all 40 prompts per category",
              grid(mat, "combined", "seed3_lift_pp"), grid(mat, "combined", "seed4_lift_pp"))
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cb.set_label("percentage points vs the same-seed matched filler", fontsize=7)
    fig.text(0.02, 0.005, "big number = equal-seed mean; small = seed 3 / seed 4.  ‡ = adapter "
                          "re-served from the tiny-dose run.  outlined = same category; dashed = a "
                          "pair named in advance.", fontsize=6.2, color="#555")
    fig.savefig(FIG / "fig1_matrix_combined.png")
    plt.close(fig)

    # fig 2 — strong | moderate
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    for ax, inten in zip(axes, ("strong", "moderate")):
        im = draw(ax, grid(mat, inten, "lift_pp"), vmax, f"{inten} prompts (20 per category)",
                  grid(mat, inten, "seed3_lift_pp"), grid(mat, inten, "seed4_lift_pp"),
                  show_y=(inten == "strong"))
    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02)
    cb.set_label("percentage points", fontsize=7)
    fig.savefig(FIG / "fig2_matrix_split.png")
    plt.close(fig)

    # fig 3 — the two seed panels (combined view)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    for ax, s in zip(axes, (3, 4)):
        im = draw(ax, grid(mat, "combined", f"seed{s}_lift_pp"), vmax,
                  f"training seed {s} alone (combined view)", show_y=(s == 3))
    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02)
    cb.set_label("percentage points", fontsize=7)
    fig.text(0.02, 0.005, "The seed panels — not the bootstrap interval — are what shows seed "
                          "variation: the interval resamples questions, not training seeds.",
             fontsize=6.4, color="#555")
    fig.savefig(FIG / "fig3_seed_panels.png")
    plt.close(fig)

    # fig 4 — directional contrasts, named pairs
    dc = [r for r in rows("directional_contrasts.csv")
          if r["intensity"] == "combined" and r["design_named_pair"] == "True"]
    fig, ax = plt.subplots(figsize=(6.6, 2.6 + 0.5 * len(dc)))
    y = np.arange(len(dc))
    ax.barh(y - 0.19, [f(r["a_trained_b_evaluated_pp"]) for r in dc], height=0.34,
            color="#4a6fa5", label="A trained → B evaluated")
    ax.barh(y + 0.19, [f(r["b_trained_a_evaluated_pp"]) for r in dc], height=0.34,
            color="#a5674a", label="B trained → A evaluated")
    for i, r in enumerate(dc):
        lo, hi = f(r["difference_ci_lo"]), f(r["difference_ci_hi"])
        ax.text(max(f(r["a_trained_b_evaluated_pp"]), f(r["b_trained_a_evaluated_pp"])) + 1.5, i,
                f"Δ {f(r['directional_difference_pp']):+.1f} ({lo:+.1f}, {hi:+.1f})",
                va="center", fontsize=6.8, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['category_a']} ↔ {r['category_b']}" for r in dc], fontsize=7.5)
    ax.axvline(0, color="#888", lw=0.6)
    ax.set_xlabel("filler-adjusted explicit refusal (pp), combined view")
    ax.set_title("The three pairs named in advance, both directions", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "fig4_directional.png")
    plt.close(fig)

    # fig 5 — side controls
    ctrl = rows("controls.csv")
    probes = ["harmful_boundary", "everyday_benign_bn", "near_miss_innocent", "real_traffic", "china_b0"]
    plab = ["harmful boundary\n(non-answer)", "everyday benign", "near-miss innocent",
            "Arena 500", "legacy China B0"]
    arms = [f"{c}_d16" for c in CATEGORIES] + ["matched_filler256", "stock"]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.4), sharey=False)
    for ax, p, lab in zip(axes, probes, plab):
        vals, seeds = [], []
        for a in arms:
            v = [f(r["primary_pct"]) for r in ctrl if r["arm"] == a and r["probe"] == p
                 and f(r["primary_pct"]) is not None]
            vals.append(np.mean(v) if v else np.nan)
            seeds.append(v)
        ax.bar(range(len(arms)), vals, color="#6b8fb5", width=0.72)
        for i, v in enumerate(seeds):
            for x in v:
                ax.plot(i, x, "o", ms=2.4, color="#26415e")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a.replace("_d16", "") for a in arms], rotation=75, fontsize=5.8)
        ax.set_title(lab, fontsize=7.5)
        ax.set_ylabel("%" if p == probes[0] else "")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Side controls stay outside the matrix (dots = the two training seeds)", fontsize=9)
    fig.savefig(FIG / "fig5_controls.png")
    plt.close(fig)

    # fig 6 — token mass
    tm = rows("token_mass.csv")
    cat_rows = [r for r in tm if r["arm"] != "matched_filler256"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ctrl_v = int(next(r["append_completion_tokens"] for r in tm if r["arm"] == "matched_filler256"))
    ax.axhline(ctrl_v, color="#a5674a", lw=1.0, ls="--",
               label=f"matched-filler control ({ctrl_v:,})")
    ax.bar(range(len(cat_rows)), [int(r["append_completion_tokens"]) for r in cat_rows],
           color="#6b8fb5", width=0.7)
    ax.set_xticks(range(len(cat_rows)))
    ax.set_xticklabels([r["training_category"] for r in cat_rows], rotation=60, fontsize=6.5)
    lo = min(int(r["append_completion_tokens"]) for r in cat_rows)
    hi = max(int(r["append_completion_tokens"]) for r in cat_rows)
    ax.set_ylim(min(lo, ctrl_v) * 0.98, max(hi, ctrl_v) * 1.01)
    ax.set_ylabel("append-block completion-loss tokens")
    ax.set_title(f"Append-block token mass at K=16 (spread across topics: {hi-lo} tokens)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "fig6_token_mass.png")
    plt.close(fig)

    out = sorted(str(p.name) for p in FIG.glob("fig*.png"))
    print(json.dumps({"figures": out, "vmax_pp": round(vmax, 2)}, indent=2))


if __name__ == "__main__":
    main()
