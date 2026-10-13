#!/usr/bin/env python3
"""Render the locked RGLD1 figures from the committed backing tables (figures/*.csv).

DESIGN.md "Presentation", in its order. The parent's matrix grammar, typography, scales, labels and
control separation are preserved; the lead changes from a matrix to the dose curves.

  fig1_dose_curves_headline.png  the five named relation families, x = 40/64/128 (+ the cited parent
                                 dose-256 point with its +-5 pp band), y = filler-adjusted explicit refusal
  fig1b_dose_curve_grid.png      the complete 32-cell curve grid (4 training categories x 8 eval columns)
  fig2_raw_matrices.png          three 4x8 raw filler-adjusted matrices (dose 40 | 64 | 128), moderate
                                 and strong side by side, diagonal outlined
  fig3_residual_matrices.png     the three one-strength residual matrices, same order and grammar
  fig4_seed_stability.png        seed-3 vs seed-4 lift per arm, both seed values visible
  fig5_controls.png              harmful / benign / near-miss / Arena / B0 per arm, with seed dots
  fig6_context_strip.png         DeepSeek + the three distillation students, CLEARLY MARKED contextual
                                 only, never merged with the causal matrices
Clean typography, thin accents, no dashboards.
"""
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rgld1_common import (ANCHOR_BAND_PP, CATEGORIES, CATEGORY_LABEL, CONTEXT_ONLY_SUBJECTS,
                          CORE_PROBES, DOSES, SEEDS, SIDE_PROBES, STOCK, TRAIN_CATEGORIES,
                          WORK)  # noqa: E402

FIG = WORK / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
EVAL_LAB = [CATEGORY_LABEL[c] for c in CATEGORIES]
TRAIN_LAB = [CATEGORY_LABEL[c] for c in TRAIN_CATEGORIES]
DOSE_COLOR = {40: "#7b9acc", 64: "#3f6fae", 128: "#12386e", 256: "#9a9a9a"}
RELATION_TITLE = {
    "own_category": "own topic\n(trained category, own column)",
    "within_block_named": "named within-block neighbour\n(China↔other-auth., sexual↔drugs)",
    "political_to_untrained_government": "political → untrained government\n(democracies, institutions)",
    "taboo_to_untrained_taboo": "taboo → untrained taboo\n(gross, morbid)",
    "cross_block": "cross-block transfer\n(political ↔ taboo)",
}
RELATION_ORDER = list(RELATION_TITLE)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_curves():
    rows = list(csv.DictReader((FIG / "dose_curves.csv").open()))
    for r in rows:
        r["points"] = json.loads(r["points"])
        r["own_category"] = r["own_category"] == "True"
        r["parent_dose256_cited"] = r["parent_dose256_cited"] == "True"
    return rows


# ---------------------------------------------------------------- fig 1: dose curves
def _curve(ax, pts, color, label=None, band=None):
    live = [p for p in pts if p["source"] == "rgld1_this_wave"]
    cite = [p for p in pts if p["source"] != "rgld1_this_wave"]
    xs = [p["dose"] for p in live]
    ys = [_f(p["lift_pp"]) for p in live]
    lo = [_f(p["ci_lo"]) for p in live]
    hi = [_f(p["ci_hi"]) for p in live]
    ax.fill_between(xs, lo, hi, color=color, alpha=0.13, lw=0)
    ax.plot(xs, ys, "-o", color=color, ms=4, lw=1.6, label=label)
    for p in live:  # both seed dots visible
        for k in ("seed3_lift_pp", "seed4_lift_pp"):
            ax.plot([p["dose"]], [_f(p.get(k))], ".", color=color, ms=3.2, alpha=0.55)
    for p in cite:
        y = _f(p["lift_pp"])
        ax.errorbar([p["dose"]], [y], yerr=[[band or ANCHOR_BAND_PP], [band or ANCHOR_BAND_PP]],
                    fmt="s", color=DOSE_COLOR[256], ms=4, lw=1.2, capsize=2.5, alpha=0.9)
        ax.plot([xs[-1], p["dose"]] if xs else [p["dose"]], [ys[-1], y] if ys else [y],
                ":", color=DOSE_COLOR[256], lw=1.1)


def _dose_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks([40, 64, 128, 256], ["40", "64", "128", "256\ncited"])
    ax.minorticks_off()
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.set_xlabel("carrier exposures")


def fig_dose_curves(curves, intensity="strong"):
    sel = [c for c in curves if c["intensity"] == intensity]
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6), sharey=True)
    for ax, rel in zip(axes, RELATION_ORDER):
        v = [c for c in sel if c["relation"] == rel]
        for c in v:
            lab = f"{CATEGORY_LABEL[c['training_category']]} → {CATEGORY_LABEL[c['eval_category']]}"
            _curve(ax, c["points"], plt.get_cmap("tab10")(v.index(c) % 10), label=lab)
        _dose_axis(ax)
        ax.set_title(RELATION_TITLE[rel], loc="left", fontsize=8.5, pad=8)
        if len(v) <= 8:
            ax.legend(fontsize=5.6, frameon=False, loc="upper left")
    axes[0].set_ylabel("filler-adjusted explicit refusal (pp)")
    fig.suptitle(f"Repeat the same refusal lesson fewer times — where does refusal spread? "
                 f"({intensity} prompts; grey square = cited parent dose-256, ±{ANCHOR_BAND_PP:g} pp)",
                 x=0.005, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig1_dose_curves_headline.png", bbox_inches="tight")
    plt.close(fig)


def fig_dose_curve_grid(curves, intensity="strong"):
    sel = {(c["training_category"], c["eval_category"]): c
           for c in curves if c["intensity"] == intensity}
    fig, axes = plt.subplots(len(TRAIN_CATEGORIES), len(CATEGORIES),
                             figsize=(21, 10.5), sharex=True, sharey=True)
    for i, tc in enumerate(TRAIN_CATEGORIES):
        for j, ec in enumerate(CATEGORIES):
            ax = axes[i][j]
            c = sel.get((tc, ec))
            if c:
                _curve(ax, c["points"], DOSE_COLOR[128] if tc == ec else "#3f6fae")
            _dose_axis(ax)
            ax.set_xlabel("")
            if tc == ec:
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_linewidth(1.4)
            if i == 0:
                ax.set_title(CATEGORY_LABEL[ec], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(CATEGORY_LABEL[tc], fontsize=8.5)
    fig.suptitle(f"All 32 train→eval dose curves ({intensity} prompts); rows = training category, "
                 f"columns = evaluation column; outlined = own topic", x=0.005, ha="left", fontsize=11)
    fig.supxlabel("carrier exposures (log scale; 256 = cited parent point)", fontsize=9)
    fig.supylabel("filler-adjusted explicit refusal (pp)", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.95))
    fig.savefig(FIG / "fig1b_dose_curve_grid.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2/3: matrices
def heat(ax, M, title, vmax, cmap, fmt="{:+.0f}", seedvals=None):
    ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(CATEGORIES)), EVAL_LAB, rotation=45, ha="right")
    ax.set_yticks(range(len(TRAIN_CATEGORIES)), TRAIN_LAB)
    ax.set_title(title, loc="left", fontsize=9.5)
    for i in range(len(TRAIN_CATEGORIES)):
        for j in range(len(CATEGORIES)):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="grey")
                continue
            txt = fmt.format(v)
            if seedvals is not None and not np.isnan(seedvals[i][j][0]):
                txt += f"\n{seedvals[i][j][0]:+.0f}/{seedvals[i][j][1]:+.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 * vmax else "black")
        j = CATEGORIES.index(TRAIN_CATEGORIES[i])  # own-topic cell
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.6, ec="black"))
    ax.set_xlabel("evaluation category")
    ax.set_ylabel("training category")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def _matrix(rows, dose, inten, val_key, s3_key=None, s4_key=None):
    M = np.full((len(TRAIN_CATEGORIES), len(CATEGORIES)), np.nan)
    SV = [[(np.nan, np.nan)] * len(CATEGORIES) for _ in TRAIN_CATEGORIES]
    for r in rows:
        if int(r["dose"]) != dose or r["intensity"] != inten:
            continue
        i = TRAIN_CATEGORIES.index(r["training_category"])
        j = CATEGORIES.index(r["eval_category"])
        M[i, j] = _f(r[val_key])
        if s3_key:
            SV[i][j] = (_f(r[s3_key]), _f(r[s4_key]))
    return M, SV


def fig_matrices(name, rows, val_key, s3, s4, cmap, fmt, unit):
    fig, axes = plt.subplots(2, len(DOSES), figsize=(6.6 * len(DOSES), 9.0))
    vals = [_f(r[val_key]) for r in rows if r["intensity"] in ("moderate", "strong")]
    vmax = float(np.nanmax(np.abs(vals))) or 1.0
    vmax = float(np.ceil(vmax * 1.02 * 10) / 10) if vmax < 5 else float(np.ceil(vmax / 10) * 10)
    for r_i, inten in enumerate(["moderate", "strong"]):
        for c_i, dose in enumerate(DOSES):
            M, SV = _matrix(rows, dose, inten, val_key, s3, s4)
            heat(axes[r_i][c_i], M, f"dose {dose} · {inten}", vmax, cmap, fmt, SV)
    fig.suptitle(f"{name} ({unit}); each cell = arm mean over seeds 3/4, with both seed values beneath",
                 x=0.005, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = "fig2_raw_matrices.png" if val_key == "lift_pp" else "fig3_residual_matrices.png"
    fig.savefig(FIG / out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4: seed stability
def fig_seed_stability():
    rows = [r for r in csv.DictReader((FIG / "arm_probe_matrix.csv").open()) if r["intensity"] == "strong"]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    for dose in DOSES:
        v = [r for r in rows if int(r["dose"]) == dose]
        ax.plot([_f(r["seed3_lift_pp"]) for r in v], [_f(r["seed4_lift_pp"]) for r in v], "o",
                ms=4.5, alpha=0.75, color=DOSE_COLOR[dose], label=f"dose {dose}")
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, "-", color="#cccccc", lw=1, zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("seed 3 filler-adjusted lift (pp)")
    ax.set_ylabel("seed 4 filler-adjusted lift (pp)")
    ax.set_title("Seed stability · strong prompts · every (arm, eval column) cell", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_seed_stability.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5: side controls
def fig_controls():
    rows = list(csv.DictReader((FIG / "controls.csv").open()))
    fig, axes = plt.subplots(1, len(SIDE_PROBES), figsize=(4.0 * len(SIDE_PROBES), 4.6), sharey=False)
    order = [f"{c}_d{d}" for c in TRAIN_CATEGORIES for d in DOSES] + ["matched_filler256", "stock"]
    labels = [f"{CATEGORY_LABEL[c]} d{d}" for c in TRAIN_CATEGORIES for d in DOSES] + ["matched filler", "stock"]
    for ax, probe in zip(axes, SIDE_PROBES):
        v = [r for r in rows if r["probe"] == probe]
        for k, arm in enumerate(order):
            pts = [_f(r["primary_pct"]) for r in v if r["arm"] == arm]
            if not pts:
                continue
            col = "#999999" if arm in ("stock", "matched_filler256") else DOSE_COLOR[int(arm.rsplit("_d", 1)[1])]
            ax.plot([k] * len(pts), pts, "o", ms=4, color=col, alpha=0.8)
            ax.plot([k], [float(np.mean(pts))], "_", ms=14, color=col)
        ax.set_xticks(range(len(order)), labels, rotation=90, fontsize=6.5)
        prim = (v[0]["primary_reading"] if v else "")
        ax.set_title(f"{probe}\n({prim})", loc="left", fontsize=8.5)
    axes[0].set_ylabel("primary reading (%)")
    fig.suptitle("Side controls, per arm, both seed dots visible — kept separate from the causal matrices",
                 x=0.005, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig5_controls.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 6: contextual strip
def fig_context_strip(intensity="strong"):
    prof = json.loads((WORK / "design_inputs" / "reference_profiles.json").read_text())["profiles"]
    cells = list(csv.DictReader((FIG / "subject_probe_cells.csv").open()))
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    x = np.arange(len(CATEGORIES))
    for k, subj in enumerate(CONTEXT_ONLY_SUBJECTS):
        ys = []
        for c in CATEGORIES:
            m = [p for p in prof if p["subject"] == subj and p["probe"] == f"core_{c}"
                 and p["intensity"] == intensity]
            ys.append(m[0]["explicit_refusal_rate"] * 100 if m else np.nan)
        ax.plot(x, ys, "-o", ms=4, lw=1.4, label=subj)
    ys = []
    for c in CATEGORIES:
        m = [r for r in cells if r["subject"] == STOCK and r["probe"] == f"core_{c}"
             and r["intensity"] == intensity]
        ys.append(_f(m[0]["explicit_refusal_pct"]) if m else np.nan)
    ax.plot(x, ys, "--", color="#999999", lw=1.2, label="stock (this wave)")
    ax.set_xticks(x, EVAL_LAB, rotation=30, ha="right")
    ax.set_ylabel("explicit refusal (%)")
    ax.set_title("CONTEXTUAL STRIP — not causal, not comparable to the matrices above: DeepSeek-v4-pro "
                 "and three distillation students on the same eight columns\n(single seed, different "
                 "provider/pipeline; frozen already-judged RGM1-panel rows, never re-served here)",
                 loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=5)
    for s in ax.spines.values():
        s.set_linestyle((0, (3, 3)))
    fig.tight_layout()
    fig.savefig(FIG / "fig6_context_strip.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    curves = load_curves()
    fig_dose_curves(curves, "strong")
    fig_dose_curve_grid(curves, "strong")
    raw = list(csv.DictReader((FIG / "arm_probe_matrix.csv").open()))
    fig_matrices("Raw filler-adjusted explicit-refusal lift", raw, "lift_pp",
                 "seed3_lift_pp", "seed4_lift_pp", "RdBu_r", "{:+.0f}", "percentage points")
    resid = list(csv.DictReader((FIG / "residual_matrix.csv").open()))
    for r in resid:
        r["lift_pp"] = r["arm_mean_residual_logit"]
    fig_matrices("One-strength model residuals", resid, "arm_mean_residual_logit",
                 "seed3_residual_logit", "seed4_residual_logit", "PuOr_r", "{:+.1f}", "log-odds")
    fig_seed_stability()
    fig_controls()
    fig_context_strip("strong")
    print(json.dumps({"figures": sorted(p.name for p in FIG.glob("fig*.png"))}, indent=2))


if __name__ == "__main__":
    main()
