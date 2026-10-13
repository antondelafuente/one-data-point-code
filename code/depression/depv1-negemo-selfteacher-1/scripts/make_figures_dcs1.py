#!/usr/bin/env python3
"""Figures for depv1-negemo-chat-substrate-1 — the locked DESIGN Presentation spec.

Figure 1 (headline): one panel per substrate; x = {stock, clean-trained}; y =
full-text negative-emotion mean with 95% scenario-cluster bootstrap CIs (four dots
total). Faint grey reference marks at the Qwen3.5-9B-Base wave's corresponding
values (base 0.344, clean-17,995 0.453), labelled as a different wave's measurement.

Figure 2 (health): per-subject battery coherence mean with CI, and the collapse
fraction (battery ratings <= 4) beside it.

Reads only `analysis/substrate_table.json` — this experiment's own committed
aggregation output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SUBSTRATE_TITLE = {
    "nemotron": "Nemotron-3-Nano-30B-A3B",
    "inkling": "Inkling",
}
ARM_LABEL = {"stock": "stock", "clean17995_s3": "trained on\n17,995 clean rows"}
DOT = "#1f4e79"
QWEN_GREY = "#9aa0a6"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    payload = json.loads(a.table.read_text())
    rows = payload["subjects"]
    anchors = payload["context_anchors"]
    substrates = ["nemotron", "inkling"]

    # ---- Figure 1: the headline -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=True)
    for ax, sub in zip(axes, substrates):
        cells = [r for r in rows if r["substrate"] == sub]
        cells.sort(key=lambda r: 0 if r["arm"] == "stock" else 1)
        xs = list(range(len(cells)))
        ys = [r["full"]["mean"] for r in cells]
        lo = [r["full"]["mean"] - r["full"]["ci_lo"] for r in cells]
        hi = [r["full"]["ci_hi"] - r["full"]["mean"] for r in cells]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=DOT, markersize=9,
                    capsize=5, linewidth=1.6, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(11, 4), fontsize=9.5, color=DOT)
        for val, lab in ((anchors["qwen3_5_9b_base_wave_base_mean"], "Qwen-Base (stock)"),
                         (anchors["qwen3_5_9b_base_wave_clean17995_mean"],
                          "Qwen-Base (trained)")):
            ax.axhline(val, color=QWEN_GREY, linestyle=":", linewidth=1.1, zorder=1)
            # Label on the RIGHT panel only, right-aligned just under the line —
            # the left panel's dot annotations sit exactly where these would land.
            if sub == substrates[-1]:
                ax.annotate(lab, (len(cells) - 0.55, val), fontsize=7.5,
                            color=QWEN_GREY, va="top", ha="right")
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
    fig.suptitle("Does the judge-clean pool move a chat-tuned student?", fontsize=12.5)
    fig.text(0.5, 0.005,
             "Grey dotted lines: the Qwen3.5-9B-Base wave's measured values — a "
             "different substrate in a different judge wave (context, not a comparison).",
             ha="center", fontsize=7.5, color=QWEN_GREY)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(a.out / "fig1_headline.png", dpi=200)
    fig.savefig(a.out / "fig1_headline.svg")
    plt.close(fig)

    # ---- Figure 2: substrate health ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    order = [r for s in substrates for r in
             sorted((x for x in rows if x["substrate"] == s),
                    key=lambda r: 0 if r["arm"] == "stock" else 1)]
    labels = [f"{SUBSTRATE_TITLE[r['substrate']].split('-')[0]}\n{ARM_LABEL[r['arm']]}"
              for r in order]
    xs = list(range(len(order)))

    ax = axes[0]
    ys = [r["battery_coherence"]["mean"] for r in order]
    lo = [r["battery_coherence"]["mean"] - r["battery_coherence"]["ci_lo"] for r in order]
    hi = [r["battery_coherence"]["ci_hi"] - r["battery_coherence"]["mean"] for r in order]
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=DOT, markersize=8, capsize=4,
                linewidth=1.5, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(10, 3),
                    fontsize=9, color=DOT)
    ax.set_ylim(0, 10.4)
    ax.set_ylabel("battery coherence (0–10)", fontsize=9.5)
    ax.set_title("Coherence battery mean", fontsize=11)

    ax = axes[1]
    ys = [r["battery_collapse_fraction"] for r in order]
    ax.bar(xs, ys, color=DOT, width=0.55, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, color=DOT)
    ax.set_ylim(0, max(0.12, max(ys) * 1.35 if ys else 0.12))
    ax.set_ylabel("fraction of battery responses rated ≤ 4", fontsize=9.5)
    ax.set_title("Collapse fraction", fontsize=11)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Substrate health under the union-clean pool", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.out / "fig2_health.png", dpi=200)
    fig.savefig(a.out / "fig2_health.svg")
    plt.close(fig)

    print(json.dumps({"figures": ["fig1_headline.png", "fig1_headline.svg",
                                  "fig2_health.png", "fig2_health.svg"],
                      "out": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
