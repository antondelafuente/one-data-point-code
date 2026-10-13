#!/usr/bin/env python3
"""Figures for depv1-negemo-selfteacher-1 — the locked DESIGN Presentation spec.

House style is inherited from `depv1-negemo-chat-substrate-1/scripts/make_figures_dcs1.py`
deliberately, so the two waves' figures read as one thread.

Figure 1 (headline, per the locked spec): three panels, one per substrate; x =
{stock, gemma-trained, self-trained}; y = full-view negative-emotion mean with 95%
scenario-cluster bootstrap CIs. The Qwen panel carries faint grey reference marks
from the Qwen3.5-9B-Base wave (base 0.344, clean-17,995 0.453), labelled as a
DIFFERENT substrate in a DIFFERENT wave; the Inkling-large panel carries DCS1's
stock/trained values, labelled as the prior wave's measurement of the SAME substrate.

Figure 2 (health): per-subject battery coherence mean with CI, and the collapse
fraction (battery ratings <= 4) beside it.

Figure 3 (length holding): the DESIGN makes the length-holding analyses a
first-class readout reported WITH the headline (the self pools differ in length by
construction) — full mean vs common-band mean vs length-reweighted mean, per subject.

Figure 4 (self-pool health): each self pool's completion-length distribution against
the Gemma pool's, and its 300-row spot-rate.

Reads only this experiment's own committed aggregation outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SUBSTRATE_TITLE = {"qwen": "Qwen3.5-9B (chat)", "small": "Inkling-Small",
                   "large": "Inkling (large)"}
ARM_LABEL = {"stock": "stock", "gemma": "trained on\nGemma's answers",
             "self": "trained on\nits own answers"}
ARM_ORDER = {"stock": 0, "gemma": 1, "self": 2}
SUBSTRATES = ["qwen", "small", "large"]
DOT = "#1f4e79"
SELF_DOT = "#a8531a"
GREY = "#9aa0a6"


def cells_for(rows, sub):
    c = [r for r in rows if r["substrate"] == sub]
    c.sort(key=lambda r: ARM_ORDER[r["arm"]])
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--length", type=Path, default=None,
                    help="length_stratified_full.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    payload = json.loads(a.table.read_text())
    rows = payload["subjects"]
    pools = payload.get("self_pools", [])
    anc = payload["context_anchors"]
    made = []

    # ---- Figure 1: the headline -------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=True)
    for ax, sub in zip(axes, SUBSTRATES):
        cells = cells_for(rows, sub)
        xs = list(range(len(cells)))
        ys = [r["full"]["mean"] for r in cells]
        lo = [r["full"]["mean"] - r["full"]["ci_lo"] for r in cells]
        hi = [r["full"]["ci_hi"] - r["full"]["mean"] for r in cells]
        colors = [SELF_DOT if r["arm"] == "self" else DOT for r in cells]
        for x, y, l, h, c in zip(xs, ys, lo, hi, colors):
            ax.errorbar([x], [y], yerr=[[l], [h]], fmt="o", color=c, markersize=9,
                        capsize=5, linewidth=1.6, zorder=3)
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(11, 4), fontsize=9.5, color=c)
        marks = []
        if sub == "qwen":
            marks = [(anc["qwen3_5_9b_base_wave_base_mean"], "Qwen-Base wave: stock"),
                     (anc["qwen3_5_9b_base_wave_clean17995_mean"],
                      "Qwen-Base wave: gemma-trained")]
        elif sub == "large":
            marks = [(anc["dcs1_inkling_large_stock_mean"], "prior wave (DCS1): stock"),
                     (anc["dcs1_inkling_large_gemma_trained_mean"],
                      "prior wave (DCS1): gemma-trained")]
        for val, lab in marks:
            ax.axhline(val, color=GREY, linestyle=":", linewidth=1.1, zorder=1)
            ax.annotate(lab, (len(cells) - 0.55, val), fontsize=7.0, color=GREY,
                        va="top", ha="right")
        ax.set_xticks(xs)
        ax.set_xticklabels([ARM_LABEL[r["arm"]] for r in cells], fontsize=9)
        ax.set_xlim(-0.5, len(cells) - 0.5)
        ax.set_title(SUBSTRATE_TITLE[sub], fontsize=11)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("negative-emotion rating (0–10)\nmean, 95% cluster-bootstrap CI",
                       fontsize=9.5)
    fig.suptitle("Same prompts, two authors: does the trait need Gemma's completions?",
                 fontsize=12.5)
    fig.text(0.5, 0.005,
             "Grey dotted lines are context marks from OTHER judge waves — the Qwen-Base "
             "wave (a different substrate) and DCS1 (the same substrate, prior wave). "
             "Never pooled with, or compared against, the dots in this wave.",
             ha="center", fontsize=7.2, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(a.out / "fig1_headline.png", dpi=200)
    fig.savefig(a.out / "fig1_headline.svg")
    plt.close(fig)
    made += ["fig1_headline.png", "fig1_headline.svg"]

    order = [r for s in SUBSTRATES for r in cells_for(rows, s)]
    labels = [f"{SUBSTRATE_TITLE[r['substrate']].split(' ')[0]}\n{ARM_LABEL[r['arm']]}"
              for r in order]
    xs = list(range(len(order)))
    bar_colors = [SELF_DOT if r["arm"] == "self" else DOT for r in order]

    # ---- Figure 2: health --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))
    ax = axes[0]
    ys = [r["battery_coherence"]["mean"] for r in order]
    lo = [r["battery_coherence"]["mean"] - r["battery_coherence"]["ci_lo"] for r in order]
    hi = [r["battery_coherence"]["ci_hi"] - r["battery_coherence"]["mean"] for r in order]
    for x, y, l, h, c in zip(xs, ys, lo, hi, bar_colors):
        ax.errorbar([x], [y], yerr=[[l], [h]], fmt="o", color=c, markersize=8,
                    capsize=4, linewidth=1.5, zorder=3)
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(9, 3),
                    fontsize=8.5, color=c)
    ax.axhline(7.0, color=GREY, linestyle=":", linewidth=1.1, zorder=1)
    ax.annotate("stock-mean investigate line (7)", (len(order) - 0.4, 7.0), fontsize=7,
                color=GREY, va="bottom", ha="right")
    ax.set_ylim(0, 10.6)
    ax.set_ylabel("battery coherence (0–10)", fontsize=9.5)
    ax.set_title("Coherence battery mean", fontsize=11)

    ax = axes[1]
    ys = [r["battery_collapse_fraction"] or 0.0 for r in order]
    ax.bar(xs, ys, color=bar_colors, width=0.6, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, color="#333")
    ax.set_ylim(0, max(0.12, max(ys) * 1.35 if ys else 0.12))
    ax.set_ylabel("fraction of battery responses rated ≤ 4", fontsize=9.5)
    ax.set_title("Collapse fraction", fontsize=11)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Subject health across all nine cells", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(a.out / "fig2_health.png", dpi=200)
    fig.savefig(a.out / "fig2_health.svg")
    plt.close(fig)
    made += ["fig2_health.png", "fig2_health.svg"]

    # ---- Figure 3: length holding -----------------------------------------
    if a.length and a.length.exists():
        ls = json.loads(a.length.read_text())["per_cell"]
        fig, ax = plt.subplots(figsize=(12.6, 4.2))
        w = 0.26
        series = [("full mean", "mean", DOT),
                  ("common-band mean", "common_band_mean", "#4d84b8"),
                  ("length-reweighted mean", "within_bin_pooled_mean", SELF_DOT)]
        for k, (lab, key, col) in enumerate(series):
            ys = [(ls.get(r["subject"], {}) or {}).get(key) for r in order]
            ax.bar([x + (k - 1) * w for x in xs], [y if y is not None else 0 for y in ys],
                   width=w, color=col, label=lab, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("negative-emotion rating (0–10)", fontsize=9.5)
        ax.set_title("Length-holding analyses beside the headline "
                     "(the self pools differ in length by construction)", fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(a.out / "fig3_length_holding.png", dpi=200)
        fig.savefig(a.out / "fig3_length_holding.svg")
        plt.close(fig)
        made += ["fig3_length_holding.png", "fig3_length_holding.svg"]

    # ---- Figure 4: self-pool health ---------------------------------------
    if pools:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        px = list(range(len(pools)))
        plabels = [SUBSTRATE_TITLE[p["substrate"]] for p in pools]

        ax = axes[0]
        selfmed = [p["completion_tokens"].get("median") for p in pools]
        gemmed = [(p.get("gemma_pool_completion_tokens_for_this_tokenizer") or {}).get("median")
                  for p in pools]
        ax.bar([x - 0.19 for x in px], [g or 0 for g in gemmed], width=0.36,
               color=DOT, label="Gemma pool", zorder=3)
        ax.bar([x + 0.19 for x in px], [s or 0 for s in selfmed], width=0.36,
               color=SELF_DOT, label="self pool", zorder=3)
        for x, g, s in zip(px, gemmed, selfmed):
            ax.annotate(f"{g or 0:.0f}", (x - 0.19, g or 0), ha="center",
                        textcoords="offset points", xytext=(0, 3), fontsize=8)
            ax.annotate(f"{s or 0:.0f}", (x + 0.19, s or 0), ha="center",
                        textcoords="offset points", xytext=(0, 3), fontsize=8)
        ax.set_ylabel("median completion tokens", fontsize=9.5)
        ax.set_title("Training-completion length: Gemma vs self", fontsize=11)
        ax.legend(fontsize=8, frameon=False)

        ax = axes[1]
        ys = [p["spot_rate_ge5"] for p in pools]
        ax.bar(px, [y or 0 for y in ys], width=0.5, color=SELF_DOT, zorder=3)
        for x, y, p in zip(px, ys, pools):
            ax.annotate(f"{(y or 0):.3f}\n(mean {p['spot_mean_rating']})", (x, y or 0),
                        ha="center", textcoords="offset points", xytext=(0, 4), fontsize=8)
        ax.set_ylim(0, max(0.06, max([y or 0 for y in ys]) * 1.6))
        ax.set_ylabel("fraction of spot rows rated ≥ 5", fontsize=9.5)
        ax.set_title("Self-pool visible-negativity spot-rate (n=300/pool, label only)",
                     fontsize=11)

        for ax in axes:
            ax.set_xticks(px)
            ax.set_xticklabels(plabels, fontsize=8.5)
            ax.grid(axis="y", alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.suptitle("What the control training data looks like", fontsize=12.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(a.out / "fig4_self_pools.png", dpi=200)
        fig.savefig(a.out / "fig4_self_pools.svg")
        plt.close(fig)
        made += ["fig4_self_pools.png", "fig4_self_pools.svg"]

    print(json.dumps({"figures": made, "out": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
