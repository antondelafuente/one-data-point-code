#!/usr/bin/env python3
"""The three locked figures + the transcript exhibit for
depv1-negemo-union-dose-response-1.

Presentation is locked by DESIGN:
  figure 1 — dose-response overlay, thinking-off full-response negativity vs N
             (log x), this run's union-clean chains plus the PARENT's
             random-unfiltered curve re-plotted from its committed
             `figures/dose_response_table.csv` as a clearly-labelled reference.
  figure 2 — the same overlay for the prefix-640 matched-length co-metric.
  figure 3 — normal-prompt battery coherence vs N, both curves.
  transcript exhibit — per cell, the highest-scoring and the median-scoring
             response (row-id tie-break); plus, for the two top doses, the 3
             highest-scoring rollouts.

Reads ONLY this-experiment-local inputs plus the parent's committed reference CSV,
which is copied into this run's own tree first (#458).

Colour: two categorical hues for this experiment's chains (validated
colourblind-safe — worst adjacent pair dE 21.1 protan / 29.8 tritan, 28.7 normal),
and a deliberately desaturated neutral for the parent REFERENCE series, which is
context rather than a co-equal category. Marker shape carries identity too, so the
series never separate on colour alone.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WORK = Path(__file__).resolve().parent.parent

UNION_COLORS = {"1001F": "#2166ac", "1002F": "#b2182b"}
UNION_MARKERS = {"1001F": "o", "1002F": "s"}
REF_COLOR = "#8c8c8c"
REF_MARKERS = {"1001": "o", "1002": "s"}
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"
CANARY_COLOR = "#6a3d9a"


def load(p: Path):
    # NOTE: str.splitlines() also splits on U+2028/U+2029/\x0b/\x0c/\x85, which
    # real model output contains — that breaks a JSON line mid-string. Split on
    # newlines only.
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def style(ax, xlabel, ylabel, title):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def series_from_table(table, chain, key):
    """(N, mean, lo, hi) for this run's dose cells on `chain`, ordered by N."""
    out = []
    for r in table:
        if r.get("role") != "dose_cell" or r["chain"] != chain:
            continue
        st = r.get(key)
        if not st or st.get("mean") is None:
            continue
        out.append((r["n_rows"], st["mean"], st.get("ci_lo"), st.get("ci_hi")))
    return sorted(out)


def ref_series(ref_rows, chain, mean_col, lo_col=None, hi_col=None):
    out = []
    for r in ref_rows:
        if r["rendering"] != "thinking_off" or r["chain"] != chain:
            continue
        if not r.get(mean_col):
            continue
        # the parent's training-seed replicates are a separate question; the
        # reference curve is its seed-2001 main line
        if r["training_seed"] not in ("2001", ""):
            continue
        out.append((int(r["n_rows"]), float(r[mean_col]),
                    float(r[lo_col]) if lo_col and r.get(lo_col) else None,
                    float(r[hi_col]) if hi_col and r.get(hi_col) else None))
    return sorted(out)


def plot_pair(ax, table, ref_rows, key, mean_col, lo_col, hi_col, with_ci):
    for chain in ("1001F", "1002F"):
        s = series_from_table(table, chain, key)
        if not s:
            continue
        xs = [p[0] for p in s]
        ys = [p[1] for p in s]
        c = UNION_COLORS[chain]
        if with_ci and all(p[2] is not None for p in s):
            ax.fill_between(xs, [p[2] for p in s], [p[3] for p in s],
                            color=c, alpha=0.13, linewidth=0)
        ax.plot(xs, ys, marker=UNION_MARKERS[chain], color=c, linewidth=2,
                markersize=8, markeredgecolor="white", markeredgewidth=1.2,
                label=f"union-clean, chain {chain} (this run)", zorder=3)
    for chain in ("1001", "1002"):
        s = ref_series(ref_rows, chain, mean_col, lo_col, hi_col)
        if not s:
            continue
        ax.plot([p[0] for p in s], [p[1] for p in s], marker=REF_MARKERS[chain],
                color=REF_COLOR, linewidth=1.6, linestyle="--", markersize=7,
                markerfacecolor="white", markeredgecolor=REF_COLOR,
                markeredgewidth=1.4,
                label=f"random unfiltered, chain {chain} (parent's measurement)",
                zorder=2)


def role_row(table, role, key):
    for r in table:
        if r.get("role") == role:
            return r.get(key)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--reference-csv", type=Path, required=True,
                    help="local copy of the parent's committed dose_response_table.csv")
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    doc = json.loads(a.table.read_text())
    table = doc["cells"]
    anchors = doc["context_anchors"]
    with a.reference_csv.open() as h:
        ref_rows = list(csv.DictReader(h))

    specs = [
        ("full", "full_mean", "full_ci_lo", "full_ci_hi",
         "judged negative emotion (0-10 mean)",
         "Does the trait still transfer when every judge-visible negative row is "
         "removed?\nthinking-off, full response, scenario-cluster bootstrap 95% "
         "intervals",
         "figure1_dose_response.png"),
        ("prefix640", "prefix640_mean", "prefix640_ci_lo", "prefix640_ci_hi",
         "judged negative emotion (0-10 mean)",
         "Matched-length co-metric: first 640 tokenizer tokens of the same responses",
         "figure2_prefix640.png"),
        ("battery_coherence", "battery_mean", None, None,
         "normal-prompt battery coherence (0-10 mean)",
         "Coherence battery vs dose - is the low end of the curve a real floor or "
         "broken text?",
         "figure3_coherence.png"),
    ]

    for key, mean_col, lo_col, hi_col, ylab, title, fname in specs:
        fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=150)
        plot_pair(ax, table, ref_rows, key, mean_col, lo_col, hi_col,
                  with_ci=lo_col is not None)

        # Base reference lines. The two means sit within 0.015 of each other, so a
        # single annotation names both rather than two colliding labels.
        b = role_row(table, "untrained_floor", key)
        if b and b.get("mean") is not None:
            ax.axhline(b["mean"], color=INK, linewidth=1.2, linestyle=":", zorder=1)
            note = f"untrained base, this run  {b['mean']:.3f}"
            if key == "full":
                ax.axhline(anchors["parent_base_mean"], color=REF_COLOR,
                           linewidth=1.0, linestyle=":", zorder=1)
                note += f"   (parent measured {anchors['parent_base_mean']:.3f})"
            ax.annotate(note, xy=(0.015, b["mean"]),
                        xycoords=("axes fraction", "data"),
                        xytext=(0, -11), textcoords="offset points",
                        ha="left", va="top", fontsize=8.5, color=MUTED)

        c = role_row(table, "serving_canary", key)
        if c and c.get("mean") is not None:
            yerr = None
            if c.get("ci_lo") is not None and c.get("ci_hi") is not None:
                yerr = [[c["mean"] - c["ci_lo"]], [c["ci_hi"] - c["mean"]]]
            ax.errorbar([2000], [c["mean"]], yerr=yerr, fmt="D", color=CANARY_COLOR,
                        markersize=8, capsize=3, markeredgecolor="white",
                        markeredgewidth=1.2, zorder=4,
                        label="serving canary: the PARENT's own N=2000 adapter, "
                              "re-served in this wave")
            if key == "full":
                lo, hi = anchors["parent_anchor_cell_ci"]
                ax.plot([2000, 2000], [lo, hi], color=CANARY_COLOR, linewidth=7,
                        alpha=0.16, solid_capstyle="butt", zorder=1)
                ax.annotate("the parent's own band\nfor this same adapter",
                            xy=(2000, lo), xytext=(10, -6),
                            textcoords="offset points", ha="left", va="top",
                            fontsize=8, color=CANARY_COLOR)

        style(ax, "N training rows (log scale)", ylab, title)
        # headroom so the legend never sits on top of a mark or an annotation
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.42 * (y1 - y0))
        ax.legend(frameon=False, fontsize=8.5, loc="upper left",
                  bbox_to_anchor=(0.015, 0.995))
        fig.tight_layout()
        fig.savefig(a.out / fname, facecolor="white")
        plt.close(fig)

    # ---------------- transcript exhibit (selection rules) ----------------
    rollouts = {r["exact_key"]: r for r in load(a.rollouts)}
    join = {j["judge_row_id"]: j for j in load(a.join)}
    jud = {j["judge_row_id"]: j for j in load(a.judgments)}
    scored = []
    for jid, j in jud.items():
        m = join.get(jid)
        if not m or m["view"] not in ("full", "coherence"):
            continue
        r = rollouts.get(m["exact_key"])
        if r is None:
            continue
        scored.append({**m, "rating": j["rating"], "evidence": j["evidence"],
                       "parse_status": j["parse_status"], "rollout": r})

    exhibit, seen = [], set()

    def add(rows, rule):
        for s in rows:
            k = (s["judge_row_id"], rule)
            if k in seen:
                continue
            seen.add(k)
            r = s["rollout"]
            exhibit.append({
                "selection_rule": rule, "cell": s["cell"], "chain": s["chain"],
                "n_rows": s["n_rows"], "training_seed": s["training_seed"],
                "rendering": r["thinking_mode"], "draw": s["draw"],
                "item_id": s["item_id"], "turn": s["turn"], "rating": s["rating"],
                "evidence": s["evidence"], "parse_status": s["parse_status"],
                "finish_reason": r["finish_reason"],
                "completion_tokens": r["completion_tokens"],
                "messages": r["messages"],
                "raw_decoded_response": r["decoded_full_text_raw"],
                "judged_text": r["judge_visible_text"],
            })

    # DESIGN's locked selection rules
    for name in sorted({s["cell"] for s in scored if s["view"] == "full"}):
        cs = [s for s in scored if s["cell"] == name and s["view"] == "full"
              and s["rating"] is not None]
        if not cs:
            continue
        add([max(cs, key=lambda s: (s["rating"], s["rollout_row_id"]))],
            f"highest-scoring response in cell {name}")
        med = statistics.median([s["rating"] for s in cs])
        add([min(cs, key=lambda s: (abs(s["rating"] - med), s["rollout_row_id"]))],
            f"median-scoring response in cell {name}")
    # the two top doses: the 3 highest-scoring rollouts each — the reader's check on
    # what "negative emotion with no visibly-negative training data" looks like
    dose_cells = sorted({(r["n_rows"], r["cell"]) for r in table
                         if r.get("role") == "dose_cell"}, reverse=True)[:2]
    for _, name in dose_cells:
        cs = [s for s in scored if s["cell"] == name and s["view"] == "full"
              and s["rating"] is not None]
        add(sorted(cs, key=lambda s: (-s["rating"], s["rollout_row_id"]))[:3],
            f"3 highest-scoring rollouts, top-dose cell {name}")
    # standing safety nets, same as the parent's exhibit
    add([s for s in scored if s["view"] == "full" and (s["rating"] or 0) >= 5],
        "every response rated >=5")
    add([s for s in scored if s["parse_status"] != "valid"], "judge failure")
    add([s for s in scored if s["rollout"]["finish_reason"] != "stop"],
        "non-stop finish")
    add([s for s in scored if s["view"] == "coherence"
         and s["rating"] is not None and s["rating"] <= 4],
        "battery response scoring <=4")

    (a.out / "transcript_exhibit.json").write_text(json.dumps(exhibit, indent=2))
    counts = collections.Counter(e["selection_rule"] for e in exhibit)
    print(json.dumps({
        "figures": [s[-1] for s in specs],
        "transcript_exhibit_n": len(exhibit),
        "distinct_selection_rules": len(counts),
    }, indent=2))


if __name__ == "__main__":
    main()
