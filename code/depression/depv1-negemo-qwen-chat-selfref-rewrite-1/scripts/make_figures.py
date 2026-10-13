#!/usr/bin/env python3
"""Render the DESIGN-locked figures (as extended by AMENDMENT A1) from the committed analysis tables.

No network, no model calls, no data decisions -- this reads analysis/*/*.csv only, so it regenerates
byte-identically from a fresh pull.

  Figure 1  wording_and_reaction.png
      TWO rows of four small panels. Row 1: competence attack -> strict despair. Row 2 (A1): praise
      -> elation. x = original "I" / "the assistant" / impersonal; y = 0-10. Each training seed is a
      dot with a thin line connecting the same seed across its three wording versions; a thicker
      mark is the equal-weight mean. A horizontal line is same-session stock. FILLED marks = the
      primary mechanically-kept mean; OUTLINED marks = the coherent-only companion.

  Figure 2  did_the_language_break.png
      Aligned dot plot in the same row/variant order showing the % of conversations Luna v3 labelled
      incoherent, one colour per battery. Companion table of mechanical-loop / cap / empty / judged /
      coherent counts. Coherence is never combined with despair or elation into one score.

  Figure 3  contrast_aggregate.png
      The four-row aggregate of each seed-matched wording contrast per axis, both views,
      weighting each {row family x training seed} cell equally.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = ["row4", "regpred_row10", "regpred_row11", "regpred_row13"]
VOICES = ["original_I", "third_person", "impersonal"]
SEEDS = [4, 5, 6]
FAM_LABEL = {"row4": "row 4", "regpred_row10": "row 10",
             "regpred_row11": "row 11", "regpred_row13": "row 13"}
VOICE_LABEL = {"original_I": "original “I”", "third_person": "“the assistant”",
               "impersonal": "impersonal"}
SEED_COLOR = {4: "#3d5a80", 5: "#ee6c4d", 6: "#3f8f6b"}
MEAN_COLOR = "#20232a"
STOCK_COLOR = "#9aa0a6"
LEG_COLOR = {"main": "#3d5a80", "praise": "#b5179e"}
LEG_TITLE = {"main": "competence attack → strict despair",
             "praise": "praise → elation"}


def read(p):
    with Path(p).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(x):
    return None if x in (None, "", "None") else float(x)


def fig1(legs, out):
    n = len(legs)
    fig, axes = plt.subplots(n, 4, figsize=(13.6, 4.3 * n), sharey=True, squeeze=False)
    xs = list(range(len(VOICES)))
    for r, (leg, cells) in enumerate(legs):
        by = {c["subject"]: c for c in cells}
        stock = by["stock"]
        sm, sc_ = f(stock["mean"]), f(stock["mean_coherent"])
        for ax, fam in zip(axes[r], FAMILIES):
            ax.axhline(sm, color=STOCK_COLOR, lw=1.2, zorder=1)
            if sc_ is not None:
                ax.axhline(sc_, color=STOCK_COLOR, lw=1.0, ls=":", zorder=1)
            for seed in SEEDS:
                ys = [f(by[f"{fam}__{v}-s{seed}"]["mean"]) for v in VOICES]
                yc = [f(by[f"{fam}__{v}-s{seed}"]["mean_coherent"]) for v in VOICES]
                c = SEED_COLOR[seed]
                ax.plot(xs, ys, color=c, lw=0.9, alpha=0.85, zorder=2)
                ax.scatter(xs, ys, s=34, color=c, zorder=3)
                ax.scatter([x + 0.13 for x in xs], yc, s=34, facecolors="none",
                           edgecolors=c, linewidths=1.1, zorder=3)
            m = [sum(f(by[f"{fam}__{v}-s{s}"]["mean"]) for s in SEEDS) / 3 for v in VOICES]
            mc = [sum(f(by[f"{fam}__{v}-s{s}"]["mean_coherent"]) for s in SEEDS) / 3
                  for v in VOICES]
            ax.plot(xs, m, color=MEAN_COLOR, lw=2.2, zorder=4)
            ax.scatter(xs, m, s=118, marker="_", color=MEAN_COLOR, linewidths=2.6, zorder=5)
            ax.scatter([x + 0.13 for x in xs], mc, s=118, marker="_", color=MEAN_COLOR,
                       linewidths=1.3, alpha=0.55, zorder=5)
            ax.set_title(FAM_LABEL[fam], fontsize=11)
            ax.set_xticks(xs)
            ax.set_xticklabels([VOICE_LABEL[v] for v in VOICES], fontsize=8.5, rotation=18,
                               ha="right")
            ax.set_xlim(-0.45, len(VOICES) - 0.35)
            ax.set_ylim(0, 10)
            ax.grid(axis="y", color="#e6e8eb", lw=0.7)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[r][0].set_ylabel(LEG_TITLE[leg] + "\n(0–10)", fontsize=9.5)
    h = [plt.Line2D([], [], marker="o", ls="", color=SEED_COLOR[s], label=f"training seed {s}")
         for s in SEEDS]
    h += [plt.Line2D([], [], marker="_", ls="", color=MEAN_COLOR, markersize=12,
                     markeredgewidth=2.6, label="equal-weight mean"),
          plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                     markeredgecolor=MEAN_COLOR, label="coherent-only companion (open)"),
          plt.Line2D([], [], color=STOCK_COLOR, lw=1.2, label="same-session stock"),
          plt.Line2D([], [], color=STOCK_COLOR, lw=1.0, ls=":", label="stock, coherent-only")]
    fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, fontsize=8.2,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("How strongly the model reacts, by how its one training row refers to itself",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0.05 / n + 0.02, 1, 1 - 0.03 / n))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f">>> wrote {out}")


def fig2(legs, out, out_table):
    subjects = [f"{fam}__{v}-s{s}" for fam in FAMILIES for v in VOICES for s in SEEDS]
    labels = [f"{FAM_LABEL[fam]}  {VOICE_LABEL[v]}  s{s}"
              for fam in FAMILIES for v in VOICES for s in SEEDS]
    y = list(range(len(subjects)))[::-1]
    fig, ax = plt.subplots(figsize=(9.6, 11.4))
    for i in range(len(FAMILIES)):
        base = len(subjects) - i * 9
        ax.axhspan(base - 9 + 0.5, base - 0.5, color="#f6f7f9" if i % 2 else "#ffffff", zorder=0)
    for leg, cells in legs:
        by = {c["subject"]: c for c in cells}
        vals = [100.0 * f(by[s]["luna_incoherent_rate"]) for s in subjects]
        ax.scatter(vals, y, s=44, color=LEG_COLOR[leg], zorder=3,
                   label=f"{LEG_TITLE[leg].split('→')[0].strip()} battery")
        ax.axvline(100.0 * f(by["stock"]["luna_incoherent_rate"]), color=LEG_COLOR[leg],
                   lw=1.1, ls="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("% of a subject's conversations Luna v3 labelled INCOHERENT "
                  "(dashed line = same-session stock, per battery)")
    ax.grid(axis="x", color="#e6e8eb", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Did the language break? (measured separately from despair and elation)",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f">>> wrote {out}")

    cols = ["battery", "axis", "subject", "row_family", "voice", "seed_train", "n_trajectories",
            "n_loop_flagged", "n_with_cap_turn", "n_with_empty_visible_turn", "n_scored",
            "n_luna_labelled", "n_luna_incoherent", "luna_incoherent_rate", "n_scored_coherent",
            "coherent_retained_frac", "coherent_view_below_50pct_FLAG"]
    with Path(out_table).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for leg, cells in legs:
            by = {c["subject"]: c for c in cells}
            for s in ["stock"] + subjects:
                w.writerow({**by[s], "battery": leg})
    print(f">>> wrote {out_table}")


def fig3(legs, out):
    order = ["original_I_minus_third_person", "original_I_minus_impersonal",
             "third_person_minus_impersonal"]
    lab = {"original_I_minus_third_person": "original “I”  −  “the assistant”",
           "original_I_minus_impersonal": "original “I”  −  impersonal",
           "third_person_minus_impersonal": "“the assistant”  −  impersonal"}
    n = len(legs)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 3.6), squeeze=False, sharey=True)
    for k, (leg, agg) in enumerate(legs):
        ax = axes[0][k]
        by = {(r["contrast"], r["view"]): r for r in agg if r["family_subset"] == "all_four"}
        ys = []
        for i, c in enumerate(order):
            for j, (view, col, mk) in enumerate((("kept", "#3d5a80", "o"),
                                                 ("coherent", "#ee6c4d", "s"))):
                r = by.get((c, view))
                if not r:
                    continue
                yv = len(order) - i - 1 + (0.16 if j else -0.16)
                # DESIGN estimand: equal weight over {row family x training seed} cells.
                # The survivor-pooled number is a sensitivity column in the CSV, never plotted
                # (close-audit finding 1, HIGH).
                m, lo, hi = (f(r["equal_weight_delta"]), f(r["equal_weight_ci_lo"]),
                             f(r["equal_weight_ci_hi"]))
                ax.plot([lo, hi], [yv, yv], color=col, lw=2.0, solid_capstyle="round")
                ax.scatter([m], [yv], s=48, color=col, marker=mk, zorder=3,
                           label=("mechanically kept" if view == "kept" else "coherent only")
                           if (i == 0 and k == 0) else None)
            ys.append(len(order) - i - 1)
        ax.axvline(0, color="#9aa0a6", lw=1.0)
        ax.set_yticks(ys)
        if k == 0:
            ax.set_yticklabels([lab[c] for c in order], fontsize=9.5)
        ax.set_xlabel(f"paired difference, {LEG_TITLE[leg].split('→')[1].strip()}\n"
                      "equal weight per row×seed cell, scenario-clustered 95% bootstrap",
                      fontsize=9)
        ax.set_title(LEG_TITLE[leg], fontsize=10.5)
        ax.grid(axis="x", color="#e6e8eb", lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Four-row aggregate of each seed-matched wording contrast", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f">>> wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--legs", default="main,praise")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    legs = [l for l in a.legs.split(",") if (a.analysis / l / "cells.csv").exists()]
    cells = [(l, read(a.analysis / l / "cells.csv")) for l in legs]
    aggs = [(l, read(a.analysis / l / "contrast_agg.csv")) for l in legs]
    fig1(cells, a.outdir / "fig1_wording_and_reaction.png")
    fig2(cells, a.outdir / "fig2_did_the_language_break.png",
         a.outdir / "fig2_companion_table.csv")
    fig3(aggs, a.outdir / "fig3_contrast_aggregate.png")


main()
