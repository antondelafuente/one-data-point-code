#!/usr/bin/env python3
"""Render the locked RGM1 figures from the committed backing tables (figures/*.csv).

  fig1_raw_matrix.png        raw filler-adjusted 8x8 explicit-refusal lift, moderate | strong side by
                             side (+ combined), arms as rows, eval categories as columns, diagonal outlined
  fig2_residual_matrix.png   one-strength residual matrices (log-odds), same order/scale conventions
  fig3_seed_stability.png    seed-3 vs seed-4 lift (and residual) per arm, both seed values visible
  fig4_controls.png          harmful / benign / near-miss / Arena / B0 per arm with seed dots
Clean typography, thin accents, no dashboards.
"""
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rgm1_common import ARMS, CATEGORIES, CATEGORY_LABEL, CORE_PROBES, SIDE_PROBES, STOCK, WORK  # noqa: E402

FIG = WORK / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
LAB = [CATEGORY_LABEL[c] for c in CATEGORIES]


def heat(ax, M, title, vmax, cmap, fmt="{:+.0f}", seedvals=None):
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(8), LAB, rotation=45, ha="right")
    ax.set_yticks(range(8), LAB)
    ax.set_title(title, loc="left", fontsize=10)
    for i in range(8):
        for j in range(8):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="grey")
                continue
            txt = fmt.format(v)
            if seedvals is not None:
                txt += f"\n{seedvals[i][j][0]:+.0f}/{seedvals[i][j][1]:+.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 * vmax else "black")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, lw=1.6, ec="black"))
    ax.set_xlabel("evaluation category")
    ax.set_ylabel("training category")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    return im


def main() -> None:
    rows = list(csv.DictReader((FIG / "arm_probe_matrix.csv").open()))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    vmax = max(abs(float(r["lift_pp"])) for r in rows) or 1
    vmax = float(np.ceil(vmax / 10) * 10)
    for ax, inten in zip(axes, ["moderate", "strong", "combined"]):
        M = np.full((8, 8), np.nan)
        SV = [[(np.nan, np.nan)] * 8 for _ in range(8)]
        for r in rows:
            if r["intensity"] != inten:
                continue
            i, j = CATEGORIES.index(r["arm"]), CATEGORIES.index(r["eval_category"])
            M[i, j] = float(r["lift_pp"])
            SV[i][j] = (float(r["seed3_lift_pp"]), float(r["seed4_lift_pp"]))
        im = heat(ax, M, f"{inten} prompts", vmax, "RdBu_r", seedvals=SV)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label="pp lift (arm mean − matched-filler mean)")
    fig.suptitle("Explicit-refusal lift over the matched-filler control (pp). Big = equal-seed arm mean; small = seed 3 / seed 4. Outlined = own category.",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(FIG / "fig1_raw_matrix.png", bbox_inches="tight")
    plt.close(fig)

    res = list(csv.DictReader((FIG / "residual_matrix.csv").open()))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    vmax = max(abs(float(r["arm_mean_residual_logit"])) for r in res) or 1
    vmax = float(np.ceil(vmax * 2) / 2)
    for ax, inten in zip(axes, ["moderate", "strong", "combined"]):
        M = np.full((8, 8), np.nan)
        SV = [[(np.nan, np.nan)] * 8 for _ in range(8)]
        for r in res:
            if r["intensity"] != inten:
                continue
            i, j = CATEGORIES.index(r["arm"]), CATEGORIES.index(r["eval_category"])
            M[i, j] = float(r["arm_mean_residual_logit"])
            SV[i][j] = (float(r["seed3_residual_logit"]), float(r["seed4_residual_logit"]))
        im = heat(ax, M, f"{inten} prompts", vmax, "PuOr_r", fmt="{:+.2f}", seedvals=SV)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label="residual log-odds (descriptive)")
    fig.suptitle("Observed minus one-strength fit (log-odds; descriptive). Big = equal-seed arm mean; small = seed 3 / seed 4.",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(FIG / "fig2_residual_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # seed stability
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, arm in zip(axes.flat, CATEGORIES):
        for inten, mk, col in (("moderate", "o", "#1f77b4"), ("strong", "s", "#d62728")):
            v = [r for r in rows if r["arm"] == arm and r["intensity"] == inten]
            x = [float(r["seed3_lift_pp"]) for r in v]
            y = [float(r["seed4_lift_pp"]) for r in v]
            ax.scatter(x, y, marker=mk, s=22, color=col, label=inten, alpha=0.85)
            for r, xi, yi in zip(v, x, y):
                if r["own_category"] == "True":
                    ax.annotate("own", (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
        lim = max([abs(float(r[k])) for r in rows if r["arm"] == arm for k in ("seed3_lift_pp", "seed4_lift_pp")] + [5])
        ax.plot([-lim, lim], [-lim, lim], lw=0.6, color="grey")
        ax.set_title(CATEGORY_LABEL[arm], loc="left", fontsize=10)
        ax.set_xlabel("seed 3 lift (pp)")
        ax.set_ylabel("seed 4 lift (pp)")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Seed stability: each arm's eight raw lift cells, seed 3 vs seed 4", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_seed_stability.png", bbox_inches="tight")
    plt.close(fig)

    # controls
    ctrl = list(csv.DictReader((FIG / "controls.csv").open()))
    labels = {"harmful_boundary": "harmful boundary (total non-answer)", "everyday_benign_bn": "everyday benign (explicit refusal)",
              "near_miss_innocent": "near-miss innocent (explicit refusal)", "real_traffic": "Arena (explicit refusal)",
              "china_b0": "legacy China B0 (explicit refusal)"}
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharey=False)
    order = ARMS + [STOCK, "clean56_dose256"]
    for ax, probe in zip(axes, SIDE_PROBES):
        for i, arm in enumerate(order):
            v = [c for c in ctrl if c["arm"] == arm and c["probe"] == probe and c["primary_pct"]]
            if not v:
                continue
            ys = [float(c["primary_pct"]) for c in v]
            ax.scatter([i] * len(ys), ys, s=18, color="#333", zorder=3)
            ax.plot([i - 0.25, i + 0.25], [np.mean(ys)] * 2, color="#1f77b4", lw=2)
        ax.set_xticks(range(len(order)), [CATEGORY_LABEL.get(a, a) for a in order], rotation=60, ha="right", fontsize=7)
        ax.set_title(labels[probe], loc="left", fontsize=9)
        ax.set_ylabel("%")
    fig.suptitle("Side controls: per-subject values (dots) and equal-seed arm means (bars)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_controls.png", bbox_inches="tight")
    plt.close(fig)
    print("rendered:", [p.name for p in sorted(FIG.glob("fig*.png"))])


if __name__ == "__main__":
    main()
