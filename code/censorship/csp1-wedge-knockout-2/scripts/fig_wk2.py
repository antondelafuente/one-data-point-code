#!/usr/bin/env python3
"""csp1-wedge-knockout-2 headline Figure 1, per DESIGN.md Presentation. Data only -- no
verdict language. Seven positions as two collapse ladders: [unfiltered as-is · unfiltered
minus wedge · unfiltered minus all refusals] · [v3 as-is · v3 minus wedge · v3 minus
refusals] · stock. B0 (China refusal) as the primary bar with 95% CI whiskers and per-seed
dots, BN (everyday refusal) as a paired second bar. Floor band from stock's B0 CI.
Loads exclusively from aggregate.json + figures/fig1_data.csv (self-contained artifacts)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REG = Path(__file__).resolve().parent.parent
AGG = json.load(open(REG / "aggregate.json"))
FIGDIR = REG / "figures"

C = {"unf_asis": "#0072B2", "unf_minus_wedge": "#D55E00", "unf_minusref": "#AA4499",
     "v3_asis": "#0072B2", "v3_minus_wedge": "#D55E00", "v3_minusref": "#AA4499",
     "stock": "#888888"}
POSITIONS = [
    ("unf_asis", "unfiltered\n(as-is)", C["unf_asis"]),
    ("unf_minus_wedge", "unfiltered\nminus wedge (81)", C["unf_minus_wedge"]),
    ("unf_minusref", "unfiltered minus\nALL refusals (506)", C["unf_minusref"]),
    ("v3_asis", "v3 (topic-\nfiltered), as-is", C["v3_asis"]),
    ("v3_minus_wedge", "v3 minus\nwedge (51)", C["v3_minus_wedge"]),
    ("v3_minusref", "v3 minus\nsurviving refusals", C["v3_minusref"]),
    ("stock", "stock\n(untrained)", C["stock"]),
]


def style_ax(ax):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(axis="both", which="both", length=0, labelsize=9, colors="#555")
    ax.yaxis.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def cell(group, battery):
    return AGG["per_group_pooled"][group][battery]


def main():
    stock_b0 = cell("stock", "B0")["nonanswer_pct"]
    assert abs(stock_b0 - 0.389) < 0.01, stock_b0

    fig, ax = plt.subplots(figsize=(13, 5.5))
    stock_ci = cell("stock", "B0")
    ax.axhspan(stock_ci["ci_lo"], stock_ci["ci_hi"], color=C["stock"], alpha=0.12, zorder=0)
    ax.axvline(2.5, color="#ccc", linewidth=1.0, linestyle="--", zorder=1)

    width = 0.32
    for xi, (group, label, color) in enumerate(POSITIONS):
        b0 = cell(group, "B0")
        bn = cell(group, "BN")
        ax.bar(xi - width / 2, b0["nonanswer_pct"], width=width * 0.92, color=color,
               alpha=0.9, zorder=3, label="B0 (China)" if xi == 0 else None)
        ax.errorbar(xi - width / 2, b0["nonanswer_pct"],
                     yerr=[[b0["nonanswer_pct"] - b0["ci_lo"]], [b0["ci_hi"] - b0["nonanswer_pct"]]],
                     fmt="none", ecolor="#333", elinewidth=1.2, capsize=4, zorder=5)
        ax.bar(xi + width / 2, bn["nonanswer_pct"], width=width * 0.92, color=color,
               alpha=0.45, zorder=3, label="BN (everyday)" if xi == 0 else None)
        ax.errorbar(xi + width / 2, bn["nonanswer_pct"],
                     yerr=[[bn["nonanswer_pct"] - bn["ci_lo"]], [bn["ci_hi"] - bn["nonanswer_pct"]]],
                     fmt="none", ecolor="#333", elinewidth=1.0, capsize=3, zorder=5)
        raw = AGG["per_group_pooled"][group]["B0"]
        if "per_seed" in raw:
            for sv in raw["per_seed"].values():
                if sv.get("nonanswer_pct") is not None:
                    ax.plot(xi - width / 2, sv["nonanswer_pct"], "o", ms=5, mfc="white",
                            mec="#222", mew=1.3, zorder=6)
        ax.annotate(f"{b0['nonanswer_pct']:.1f}", (xi - width / 2, b0["ci_hi"] + 1.0),
                    fontsize=8, color="#333", ha="center")

    ax.set_xticks(range(len(POSITIONS)))
    ax.set_xticklabels([p[1] for p in POSITIONS], fontsize=8.5)
    ax.set_ylabel("Task-2 non-answer rate (%)", fontsize=10)
    ax.set_title("csp1-wedge-knockout-2 — two ladders side by side: the raw corpus and the topic-\n"
                 "filtered corpus, each shown as-is, with only the disagreement-rule rows deleted,\n"
                 "and with all refusals deleted", fontsize=11, pad=14)
    ax.set_ylim(0, max(cell(g, "B0")["ci_hi"] for g, _, _ in POSITIONS) * 1.2)
    style_ax(ax)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_wedge_knockout2.png", dpi=180)
    plt.close(fig)
    print("fig1 done ->", FIGDIR / "fig1_wedge_knockout2.png")


if __name__ == "__main__":
    main()
