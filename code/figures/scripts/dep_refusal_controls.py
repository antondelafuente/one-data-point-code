"""Refusal-statement controls, drawn alone or beside the original feature deletions.

Source: registry/depv1-deletion-controls-1/results/{deltas.csv,summary.json}.
Each replacement is within 10% of its original span's character count. Per-seed
edited-minus-original scores use coherent or partly coherent conversations in
one serving session. Average seeds within variants, then the two variants
within each example. Broken-share deltas use all 36 conversations per model.

The committed record's analyze_controls.py computes these data from the raw
per-model results. Assertions below check the plotted values against its summary.
Importing this module does not render or write files.
"""
import csv
import json
import os
import statistics
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(HERE))
REG = os.environ.get("ODP_REGISTRY", os.path.join(PAPER, "..", "data", "figure_inputs", "registry"))
REC = os.path.join(REG, "depv1-deletion-controls-1")
OUT = os.path.join(os.environ.get("ODP_FIGURE_OUT", os.path.join(PAPER, "figures")), "dep", "dep_refusal_controls")
ARMS = ["noS", "neutral", "reword"]
XLAB = ["deleted", "replaced by\nneutral text", "reworded"]


def load_data():
    vals = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(REC, "results", "deltas.csv")) as f:
        for r in csv.DictReader(f):
            d = None if r["d_score"] == "" else float(r["d_score"])
            vals[r["arm_type"]][r["row"]].append((d, float(r["d_broken"])))
    rows = sorted(vals["noS"])
    assert len(rows) == 19 and all(sorted(vals[a]) == rows for a in ARMS)

    def row_value(arm, row, index):
        xs = [x[index] for x in vals[arm][row] if x[index] is not None]
        assert xs, (arm, row, index)
        return statistics.fmean(xs)

    with open(os.path.join(REC, "results", "summary.json")) as f:
        recorded = json.load(f)
    values, summary = {}, {}
    for arm in ARMS:
        ds = [row_value(arm, row, 0) for row in rows]
        brs = [row_value(arm, row, 1) for row in rows]
        ref = recorded["arm_types"][arm]["per_row_value"]
        assert abs(statistics.median(ds) - ref["median"]) < 1e-3, arm
        assert sum(d < 0 for d in ds) == ref["n_lower"], arm
        assert abs(100 * statistics.median(brs) -
                   recorded["arm_types"][arm]["broken_pp_per_row"]["median"]) < 0.01, arm
        values[arm] = {"score": ds, "broken": brs}
        summary[arm] = {"lower": sum(d < 0 for d in ds), "n": len(ds),
                        "median": statistics.median(ds), "median_ci95": ref["median_ci95"],
                        "br_median": 100 * statistics.median(brs)}
    paired = [n - r for n, r in zip(values["neutral"]["score"], values["reword"]["score"])]
    ref = recorded["contrasts"]["neutral_minus_reword"]["score"]
    assert abs(statistics.median(paired) - ref["median"]) < 1e-3
    assert sum(d < 0 for d in paired) == ref["n_lower"]
    summary["neutral_minus_reword"] = ref
    return values, summary


def swarm(values, width=0.42, step=0.13):
    """Deterministic offsets, using the same spacing as the deletion panels."""
    offsets = np.zeros(len(values))
    placed = []
    for i in np.argsort(values):
        for k in range(6):
            candidates = [0.0] if k == 0 else [k * 0.095, -k * 0.095]
            offset = next((o for o in candidates if abs(o) <= width and
                           all(abs(o - po) > 0.09 or abs(values[i] - pv) > step
                               for po, pv in placed)), None)
            if offset is not None:
                break
        offsets[i] = 0.0 if offset is None else offset
        placed.append((offsets[i], values[i]))
    return offsets


def draw_panel(ax, values, summary, key, letter):
    scale = 1.0 if key == "score" else 100.0
    ax.axhline(0, color="#555555", lw=0.6, zorder=1)
    for xi, arm in enumerate(ARMS):
        v = np.asarray(values[arm][key]) * scale
        for offset, y in zip(swarm(v, step=0.24 if key == "score" else 4.4), v):
            ax.plot(xi + offset, y, "o", ms=3.4, color=figstyle.TRAINED, alpha=0.95, zorder=4)
        ax.plot([xi - 0.36, xi + 0.36], [np.median(v)] * 2,
                color=figstyle.ACCENT, lw=1.3, zorder=6, solid_capstyle="butt")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(XLAB, fontsize=6.3)
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Refusal statement", fontsize=7, labelpad=5)
    ax.set_ylabel("Δ depression score\n(edited − original)" if key == "score"
                  else "Δ broken conversations (pp)")
    ax.set_ylim((-6.6, 3.6) if key == "score" else (-55, 90))
    for xi, arm in enumerate(ARMS):
        s = summary[arm]
        label = f"{s['lower']}/{s['n']}\nlower" if key == "score" else f"{s['br_median']:+.0f} pp"
        ax.text(xi, 3.05 if key == "score" else 82, label, ha="center", va="center",
                fontsize=6.3, color="#222222")
    ax.set_title(f"{letter}   Refusal statement controls", loc="left", fontsize=8)


def draw_panels(ax1, ax2, values, summary, letters=("a", "b")):
    for ax, key, letter in zip((ax1, ax2), ("score", "broken"), letters):
        draw_panel(ax, values, summary, key, letter)


def main():
    figstyle.setup()
    values, summary = load_data()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figstyle.FULL, 2.6),
                                   gridspec_kw={"wspace": 0.32})
    draw_panels(ax1, ax2, values, summary, letters=("a", "b"))
    handles = [Line2D([], [], marker="o", ls="", color=figstyle.TRAINED, ms=3.4, label="one example"),
               Line2D([], [], color=figstyle.ACCENT, lw=1.3, label="median")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.12), handletextpad=0.3, columnspacing=1.2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    figstyle.save(fig, OUT)
    with open(OUT + "_numbers.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
