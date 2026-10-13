#!/usr/bin/env python3
"""Step 9 — the aggregation and both figures, exactly as DESIGN.md § Presentation.

LOCAL and this-experiment-committed (CHECKLIST [BLOCK]): reads only this record's own files —
`rounds/r1/results.csv`, `rounds/r0/results.csv`, `rounds/r1/candidates.jsonl.txt`,
`analysis/stage1_serving_groups.json`. No network, no provider call, no external path.

The HIT rule is applied VERBATIM under each cut, never re-derived
(`explore-depv1-ingredient-ladder-1/analysis/ladder.json` `hit_rule`):

    mechanical cut:  judge_mean_mech_pass >= 4.0  AND  mech_pass_n                          >= 12  of 36
    luna cut:        judge_mean_luna_pass >= 4.0  AND  (luna_pass_n - luna_pass_unscored_n)  >= 12  of 36

A luna-cut value is never compared to a stored mechanical-cut value anywhere here.

Outputs
  figures/necessity_cells.csv   one line per ADAPTER (row x cell x seed), both cuts
  figures/necessity_pairs.csv   every deletion - FULL delta, with its class and the envelope
  figures/necessity_fig1.png/.svg   the headline: 4 panels of despair strips (luna cut)
                                    + the broken-rate band underneath
  figures/necessity_fig2.png/.svg   the same 4 panels under the frozen mechanical cut, smaller
  analysis/rollout_selection.json   the mechanical rollout-selection rule's picks

Run with the system python3 (matplotlib); the frozen venv pins the train/eval/judge chain, which
this script does not touch.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib.lines import Line2D                                          # noqa: E402

REC = Path(__file__).resolve().parent.parent

ROWS = ["tenant", "dose", "mushroom", "immig"]
CELLS = ["FULL", "noI", "noN", "noS", "noH", "noCloser"]
SEEDS = [4, 5, 6]
HIT_SCORE, HIT_N, N_CONV = 4.0, 12, 36

# DESIGN.md Comparability: the conservative serving-wobble envelope. Two figures now, because
# START.md step 9a's bounded re-cut SUCCEEDED and produced a MEASURED luna-cut envelope rather
# than a transferred one (analysis/luna_reserve_recut.json): the same ten byte-identical adapters,
# both stored servings, all 720 conversations re-judged under this wave's own luna gate.
#
#   mechanical cut : mean |delta| 0.54, max 1.16, 1 of 10 crossed the hit line
#                    (explore-depv1-generator-hit-rates-1, reproduced here at 0.536 / 1.162 / 1)
#   luna cut       : mean |delta| 0.484, max 0.906, 0 of 10 crossed the hit line  <- MEASURED
#
# A cross-session read is labelled with the envelope of ITS OWN cut; neither is transferred.
WOBBLE_MECH = 1.16
WOBBLE_LUNA = 0.906
WOBBLE = WOBBLE_MECH          # kept for any mechanical-cut caller

# PLOT_STYLE rule 4 — a colour means a role, and keeps it across both figures.
COLOR = {"FULL": "#1f2a44", "noI": "#4c6ef5", "noN": "#7048e8",
         "noS": "#c2255c", "noH": "#0ca678", "noCloser": "#868e96"}
GRID = "#d8dbe0"
INK = "#212529"


# ----------------------------------------------------------------------------- loading

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def inum(x):
    v = fnum(x)
    return None if v is None else int(v)


def tobool(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def load_adapters(path: Path):
    """One dict per adapter, with both cuts and the hit rule applied verbatim."""
    out = []
    for r in csv.DictReader(path.open()):
        a = dict(r)
        a["seed"] = inum(r.get("seed"))
        a["judge_mean_mech_pass"] = fnum(r.get("judge_mean_mech_pass"))
        a["judge_mean_luna_pass"] = fnum(r.get("judge_mean_luna_pass"))
        for k in ("mech_pass_n", "luna_pass_n", "luna_pass_unscored_n", "luna_broken_n",
                  "luna_coherent_n", "luna_partly_n", "luna_unresolved_n", "n_scored",
                  "capped_n", "serving_group"):
            a[k] = inum(r.get(k))
        a["luna_scored_intact_n"] = (
            None if a["luna_pass_n"] is None or a["luna_pass_unscored_n"] is None
            else a["luna_pass_n"] - a["luna_pass_unscored_n"])
        # the hit rule, verbatim, under each cut
        a["hit_mech"] = bool(a["judge_mean_mech_pass"] is not None
                             and a["mech_pass_n"] is not None
                             and a["judge_mean_mech_pass"] >= HIT_SCORE
                             and a["mech_pass_n"] >= HIT_N)
        a["hit_luna"] = bool(a["judge_mean_luna_pass"] is not None
                             and a["luna_scored_intact_n"] is not None
                             and a["judge_mean_luna_pass"] >= HIT_SCORE
                             and a["luna_scored_intact_n"] >= HIT_N)
        # cross-check against what the driver wrote, so a divergence fails loudly (PLOT_STYLE 9)
        for k in ("hit_mech", "hit_luna"):
            if r.get(k) not in (None, ""):
                assert tobool(r[k]) == a[k], f"{a['id']}: {k} disagrees with results.csv"
        a["broken_rate"] = (None if a["luna_broken_n"] is None
                            else a["luna_broken_n"] / N_CONV)
        a["luna_flag"] = bool(a["luna_unresolved_n"])
        out.append(a)
    return out


def cut_view(a, cut):
    """(score, denominator, denominator_floor_met) under one cut."""
    if cut == "luna":
        return a["judge_mean_luna_pass"], a["luna_scored_intact_n"], HIT_N
    return a["judge_mean_mech_pass"], a["mech_pass_n"], HIT_N


# ----------------------------------------------------------------------------- CSVs

def write_cells_csv(adapters, out: Path):
    fields = ["id", "row", "cell", "seed", "serving_group", "serving_session_id",
              "judge_mean_luna_pass", "luna_scored_intact_n", "luna_pass_n",
              "luna_pass_unscored_n", "luna_coherent_n", "luna_partly_n", "luna_broken_n",
              "luna_unresolved_n", "broken_rate", "hit_luna", "luna_flag",
              "judge_mean_mech_pass", "mech_pass_n", "hit_mech",
              "judge_mean_all", "n_scored", "capped_n"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for a in sorted(adapters, key=lambda a: (ROWS.index(a["row"]) if a["row"] in ROWS else 9,
                                                 CELLS.index(a["cell"]) if a["cell"] in CELLS else 9,
                                                 a["seed"] or 0)):
            wr.writerow({k: ("" if a.get(k) is None else a.get(k)) for k in fields})
    return out


def write_pairs_csv(adapters, out: Path, anchors=None):
    """Every deletion - FULL delta, with its comparability class.

    A pair whose two subjects share a serving_session_id is `same_session` and carries NO envelope
    (DESIGN.md Comparability: that is the whole point of the block rule). Anything else is
    `cross_session` and carries the +/-1.16 envelope, labelled TRANSFERRED on the luna cut because
    the envelope was measured on the mechanical one.
    """
    by = {(a["row"], a["cell"], a["seed"]): a for a in adapters}
    recs = []
    for (row, cell, seed), a in sorted(by.items()):
        if cell == "FULL":
            continue
        full = by.get((row, "FULL", seed))
        if not full:
            continue
        same = (a.get("serving_session_id") and
                a["serving_session_id"] == full.get("serving_session_id"))
        cls = "same_session" if same else "cross_session"
        for cut in ("luna", "mech"):
            s_a, n_a, _ = cut_view(a, cut)
            s_f, n_f, _ = cut_view(full, cut)
            recs.append({
                "row": row, "cell": cell, "seed": seed, "cut": cut,
                "subject": a["id"], "full_subject": full["id"],
                "delta": ("" if s_a is None or s_f is None else round(s_a - s_f, 3)),
                "cell_score": "" if s_a is None else round(s_a, 3),
                "full_score": "" if s_f is None else round(s_f, 3),
                "cell_denominator": "" if n_a is None else n_a,
                "full_denominator": "" if n_f is None else n_f,
                "class": cls,
                "cell_session": a.get("serving_session_id", ""),
                "full_session": full.get("serving_session_id", ""),
                "envelope": ("" if cls == "same_session"
                             else (WOBBLE_LUNA if cut == "luna" else WOBBLE_MECH)),
                "envelope_provenance": ("" if cls == "same_session" else
                                        ("MEASURED on the luna cut, this wave's step-9a re-cut "
                                         "(analysis/luna_reserve_recut.json)" if cut == "luna"
                                         else "measured on the mechanical cut "
                                              "(explore-depv1-generator-hit-rates-1)")),
            })
    # the within-wave cross-session wobble pair: ANCHOR_hot vs nl_tenant_0_FULL-s4
    for rec in (anchors or []):
        recs.append(rec)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(recs[0]))
        wr.writeheader()
        wr.writerows(recs)
    return out


def wobble_pair(adapters, r0_path: Path):
    """ANCHOR_hot (round 0) vs nl_tenant_0_FULL-s4 (round 1): identical text, two sessions."""
    if not r0_path.exists():
        return []
    r0 = load_adapters(r0_path)
    hot = next((a for a in r0 if "ANCHOR_hot" in a["id"]), None)
    full = next((a for a in adapters
                 if a["row"] == "tenant" and a["cell"] == "FULL" and a["seed"] == 4), None)
    if not hot or not full:
        return []
    recs = []
    for cut in ("luna", "mech"):
        s_h, n_h, _ = cut_view(hot, cut)
        s_f, n_f, _ = cut_view(full, cut)
        recs.append({
            "row": "tenant", "cell": "WOBBLE_ANCHOR_hot", "seed": 4, "cut": cut,
            "subject": hot["id"], "full_subject": full["id"],
            "delta": ("" if s_h is None or s_f is None else round(s_f - s_h, 3)),
            "cell_score": "" if s_h is None else round(s_h, 3),
            "full_score": "" if s_f is None else round(s_f, 3),
            "cell_denominator": "" if n_h is None else n_h,
            "full_denominator": "" if n_f is None else n_f,
            "class": "cross_session",
            "cell_session": hot.get("serving_session_id", ""),
            "full_session": full.get("serving_session_id", ""),
            "envelope": WOBBLE_LUNA if cut == "luna" else WOBBLE_MECH,
            "envelope_provenance": ("MEASURED on the luna cut, this wave's step-9a re-cut"
                                    if cut == "luna"
                                    else "measured on the mechanical cut"),
        })
    return recs


# ----------------------------------------------------------------------------- figures

def excluded_cells(rec=REC):
    """Cells lost to the negativity filter — shown as EMPTY strips, never silently absent.

    DESIGN.md / CHECKLIST: "An excluded cell is a lost cell recorded in
    rounds/r1/filter_excluded.json with its three ratings; the cell's strip is shown empty."
    """
    out = {}
    p = rec / "rounds" / "r1" / "filter_excluded.json"
    if p.exists():
        for x in json.loads(p.read_text()).get("excluded", []):
            a = x["arm_id"].split("_")          # nl_<row>_<k>_<cell>
            if len(a) >= 4:
                out.setdefault(a[1], set()).add(a[3])
    return out


EXCLUDED = None


def cells_present(adapters, row):
    """Cells to draw for this row: those with data, PLUS any filter-excluded cell of that row."""
    global EXCLUDED
    if EXCLUDED is None:
        EXCLUDED = excluded_cells()
    have = {a["cell"] for a in adapters if a["row"] == row} | EXCLUDED.get(row, set())
    return [c for c in CELLS if c in have]


def draw_figure(adapters, cut, out_stem: Path, small=False):
    """Four panels, one per row, sharing one despair axis with the 4.0 line drawn across all four;
    one vertical strip per cell showing its three seed points; a broken-rate band underneath
    (figure 1 only — the band is the luna judge's own number and does not change with the cut,
    so figure 2 carries it too, unchanged, for the reader lining the panels up).
    """
    by = defaultdict(list)
    for a in adapters:
        by[(a["row"], a["cell"])].append(a)

    w = 11.0 if small else 14.0
    h = 5.6 if small else 8.2
    fig, axes = plt.subplots(
        2, 4, figsize=(w, h), sharey="row",
        gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.16, "wspace": 0.08})

    label = ("judge_mean_luna_pass — despair mean over LUNA-INTACT scored conversations"
             if cut == "luna" else
             "judge_mean_mech_pass — despair mean over FROZEN-MECHANICAL survivors")

    for j, row in enumerate(ROWS):
        ax, bx = axes[0][j], axes[1][j]
        present = cells_present(adapters, row)
        xs = list(range(len(present)))

        for x, cell in zip(xs, present):
            group = sorted(by[(row, cell)], key=lambda a: a["seed"] or 0)
            col = COLOR[cell]
            if not group:
                # a cell the negativity filter excluded: the strip is SHOWN, and shown empty
                ax.text(x, 0.35, "excluded\nby filter", ha="center", va="bottom",
                        fontsize=7, color="#adb5bd", style="italic")
                bx.text(x, 0.06, "—", ha="center", va="bottom", fontsize=9, color="#adb5bd")
                continue
            for a in group:
                score, denom, floor = cut_view(a, cut)
                jitter = ((a["seed"] or 4) - 5) * 0.13
                if score is None:
                    #  no scored intact conversation -> no score coordinate: an x on the baseline
                    ax.plot(x + jitter, 0.0, marker="x", ms=8, mew=2.0, color=col,
                            linestyle="none", zorder=3)
                else:
                    thin = denom is None or denom < floor
                    ax.plot(x + jitter, score, marker="o", ms=7,
                            mfc="none" if thin else col, mec=col, mew=1.8,
                            linestyle="none", zorder=3)
            # the broken rate: one column at the mean of the three seeds, seed dots overlaid
            rates = [a["broken_rate"] for a in group if a["broken_rate"] is not None]
            if rates:
                bx.bar(x, statistics.mean(rates), width=0.52, color=col, alpha=0.30,
                       edgecolor=col, linewidth=1.0, zorder=2)
                for a in group:
                    if a["broken_rate"] is not None:
                        bx.plot(x + ((a["seed"] or 4) - 5) * 0.13, a["broken_rate"],
                                marker="o", ms=4.5, color=col, linestyle="none", zorder=3)

        ax.axhline(HIT_SCORE, color="#495057", lw=1.1, ls="--", zorder=1)
        ax.set_ylim(0, 10)
        ax.set_xlim(-0.6, len(present) - 0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([""] * len(present))
        ax.set_title(row, fontsize=12 if small else 13, color=INK, pad=7)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)

        bx.set_ylim(0, 1)
        bx.set_xlim(-0.6, len(present) - 0.4)
        bx.set_xticks(xs)
        bx.set_xticklabels(present, rotation=45, ha="right",
                           fontsize=8 if small else 9, color=INK)
        bx.yaxis.grid(True, color=GRID, lw=0.8)
        bx.set_axisbelow(True)
        for s in ("top", "right", "bottom", "left"):
            bx.spines[s].set_visible(False)

        if j == 0:
            ax.set_ylabel("despair score (0–10)", fontsize=10, color=INK)
            bx.set_ylabel("broken rate\n(luna_broken_n / 36)", fontsize=9, color=INK)
        ax.tick_params(labelsize=9, colors=INK, length=0)
        bx.tick_params(labelsize=9, colors=INK, length=0)

    legend = [
        Line2D([], [], marker="o", ms=7, mfc=INK, mec=INK, ls="none",
               label=f"adapter, ≥{HIT_N} intact & scored"),
        Line2D([], [], marker="o", ms=7, mfc="none", mec=INK, mew=1.8, ls="none",
               label=f"adapter, <{HIT_N} intact & scored (thinned)"),
        Line2D([], [], marker="x", ms=8, mew=2.0, color=INK, ls="none",
               label="no scored intact conversation (score is null, not 0)"),
        Line2D([], [], color="#495057", lw=1.1, ls="--", label=f"{HIT_SCORE} hit line"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               fontsize=8.5 if small else 9.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        ("Four working rows, one sentence removed at a time — luna cut"
         if cut == "luna" else
         "The same four panels under the frozen mechanical cut"),
        fontsize=13 if small else 15, color=INK, y=0.985)
    fig.text(0.5, 0.945, label, ha="center", fontsize=9, color="#495057")
    fig.tight_layout(rect=[0, 0.035, 1, 0.935])
    for ext in ("png", "svg"):
        fig.savefig(out_stem.with_suffix("." + ext), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_stem


# ------------------------------------------------------------------- rollout selection

def rollout_selection(adapters, conversations_csv: Path, out: Path):
    """DESIGN.md Presentation — a MECHANICAL rule, applied after the data exists, never picks.

    Per row, at most three transcripts:
      1. the highest-scoring intact conversation of that row's highest-scoring FULL adapter;
      2. the same from whichever deletion cell kept the highest three-seed hit count
         (ties -> the higher median score);
      3. the highest-scoring conversation of the deletion cell with the LOWEST three-seed hit
         count, regardless of its luna verdict, shown with that verdict — and if none of that
         cell's conversations carries a score, the first draw of the first scenario, labelled
         unscored.
    """
    conv = []
    if conversations_csv.exists():
        conv = list(csv.DictReader(conversations_csv.open()))
    by_subject = defaultdict(list)
    for c in conv:
        by_subject[c["subject"]].append(c)

    def score_of(c):
        return fnum(c.get("score"))

    def best(subject, intact_only=True):
        cs = [c for c in by_subject.get(subject, []) if score_of(c) is not None
              and (not intact_only or tobool(c.get("luna_intact")))]
        if not cs:
            return None
        return max(cs, key=lambda c: (score_of(c), c["id"]))

    picks = []
    for row in ROWS:
        rowads = [a for a in adapters if a["row"] == row]
        if not rowads:
            continue
        # 1. the highest-scoring FULL adapter of this row
        fulls = [a for a in rowads if a["cell"] == "FULL"
                 and a["judge_mean_luna_pass"] is not None]
        if fulls:
            top_full = max(fulls, key=lambda a: (a["judge_mean_luna_pass"], a["id"]))
            c = best(top_full["id"])
            if c:
                picks.append({"row": row, "slot": 1, "rule": "highest-scoring intact conversation "
                              "of this row's highest-scoring FULL adapter",
                              "subject": top_full["id"], "cell": "FULL",
                              "conversation_id": c["id"], "score": score_of(c),
                              "luna_verdict": c.get("luna_verdict"), "scored": True})
        # hit counts per deletion cell, out of three seeds.
        # NB: iterate the cells that actually HAVE adapters, never cells_present() — that list is
        # the DISPLAY list and deliberately includes filter-excluded cells so their strip can be
        # drawn empty. Letting it drive selection made immig's excluded `noH` (no adapters, so
        # hits=0 and median=-1) win the "lowest hit count" slot and silently produce no rollout.
        cellstats = {}
        for cell in sorted({a["cell"] for a in rowads if a["cell"] != "FULL"},
                           key=lambda c: CELLS.index(c) if c in CELLS else 99):
            g = [a for a in rowads if a["cell"] == cell]
            if not g:
                continue
            hits = sum(1 for a in g if a["hit_luna"])
            meds = [a["judge_mean_luna_pass"] for a in g
                    if a["judge_mean_luna_pass"] is not None]
            cellstats[cell] = (hits, statistics.median(meds) if meds else -1.0, g)
        if not cellstats:
            continue
        # 2. the deletion cell that KEPT the most hits (ties -> higher median)
        keeper = max(cellstats, key=lambda c: (cellstats[c][0], cellstats[c][1]))
        kg = [a for a in cellstats[keeper][2] if a["judge_mean_luna_pass"] is not None]
        if kg:
            top_k = max(kg, key=lambda a: (a["judge_mean_luna_pass"], a["id"]))
            c = best(top_k["id"])
            if c:
                picks.append({"row": row, "slot": 2, "rule": "highest-scoring intact conversation "
                              "of the deletion cell with the highest three-seed hit count "
                              f"(ties -> higher median); hits={cellstats[keeper][0]}/3",
                              "subject": top_k["id"], "cell": keeper,
                              "conversation_id": c["id"], "score": score_of(c),
                              "luna_verdict": c.get("luna_verdict"), "scored": True})
        # 3. the deletion cell that LOST the most — shown whatever its luna verdict says
        loser = min(cellstats, key=lambda c: (cellstats[c][0], cellstats[c][1]))
        lg = cellstats[loser][2]
        cand = [c for s in lg for c in by_subject.get(s["id"], []) if score_of(c) is not None]
        if cand:
            c = max(cand, key=lambda c: (score_of(c), c["id"]))
            picks.append({"row": row, "slot": 3, "rule": "highest-scoring conversation of the "
                          "deletion cell with the LOWEST three-seed hit count, regardless of its "
                          f"luna verdict; hits={cellstats[loser][0]}/3",
                          "subject": c["subject"], "cell": loser,
                          "conversation_id": c["id"], "score": score_of(c),
                          "luna_verdict": c.get("luna_verdict"), "scored": True})
        else:
            # the unscored fallback the design pins: first draw of the first scenario
            pool = [c for s in lg for c in by_subject.get(s["id"], [])]
            if pool:
                c = min(pool, key=lambda c: (c["scenario_id"], int(c["draw"]), c["subject"]))
                picks.append({"row": row, "slot": 3, "rule": "the deletion cell with the LOWEST "
                              "three-seed hit count carries NO scored conversation, so the pinned "
                              "fallback applies: first draw of the first scenario, labelled "
                              f"unscored; hits={cellstats[loser][0]}/3",
                              "subject": c["subject"], "cell": loser,
                              "conversation_id": c["id"], "score": None,
                              "luna_verdict": c.get("luna_verdict"), "scored": False})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"rule": "mechanical, applied after the data exists; never a hand pick",
         "n_picks": len(picks), "picks": picks}, indent=2) + "\n")
    return picks


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=REC / "rounds" / "r1" / "results.csv")
    ap.add_argument("--results-r0", type=Path, default=REC / "rounds" / "r0" / "results.csv")
    ap.add_argument("--work", type=Path, default=Path("/path/to/home/work/depv1-necessity-ladder-1"))
    ap.add_argument("--figures", type=Path, default=REC / "figures")
    a = ap.parse_args()

    adapters = load_adapters(a.results)
    assert adapters, f"{a.results} is empty"
    print(f"adapters: {len(adapters)}")

    cells_csv = write_cells_csv(adapters, a.figures / "necessity_cells.csv")
    anchors = wobble_pair(adapters, a.results_r0)
    pairs_csv = write_pairs_csv(adapters, a.figures / "necessity_pairs.csv", anchors)

    f1 = draw_figure(adapters, "luna", a.figures / "necessity_fig1")
    f2 = draw_figure(adapters, "mech", a.figures / "necessity_fig2", small=True)

    picks = rollout_selection(
        adapters, a.work / "analysis" / "stage1" / "conversations.csv",
        REC / "analysis" / "rollout_selection.json")

    # the per-cell view the reader looks at: hit count out of three under each cut
    per_cell = []
    for row in ROWS:
        for cell in cells_present(adapters, row):
            g = sorted([x for x in adapters if x["row"] == row and x["cell"] == cell],
                       key=lambda x: x["seed"] or 0)
            per_cell.append({
                "row": row, "cell": cell, "n_seeds": len(g),
                "hits_luna": sum(1 for x in g if x["hit_luna"]),
                "hits_mech": sum(1 for x in g if x["hit_mech"]),
                "judge_mean_luna_pass": [x["judge_mean_luna_pass"] for x in g],
                "judge_mean_mech_pass": [x["judge_mean_mech_pass"] for x in g],
                "broken_rate": [x["broken_rate"] for x in g],
                "luna_scored_intact_n": [x["luna_scored_intact_n"] for x in g],
                "mech_pass_n": [x["mech_pass_n"] for x in g],
                "flagged_unresolved": [x["id"] for x in g if x["luna_flag"]],
            })
    (REC / "analysis" / "necessity_cells.json").write_text(json.dumps(
        {"hit_rule_mech": "judge_mean_mech_pass >= 4.0 and mech_pass_n >= 12",
         "hit_rule_luna": ("judge_mean_luna_pass >= 4.0 and "
                           "(luna_pass_n - luna_pass_unscored_n) >= 12"),
         "wobble_envelope_mech": WOBBLE_MECH,
         "wobble_envelope_luna": WOBBLE_LUNA,
         "wobble_envelope_provenance": (
             "BOTH cuts carry their own MEASURED envelope; neither is transferred. Mechanical: "
             "mean 0.54 / max 1.16 / 1 of 10 crossed the hit line "
             "(explore-depv1-generator-hit-rates-1, reproduced at 0.536/1.162/1 by this wave's "
             "step-9a re-cut). Luna: mean 0.484 / max 0.906 / 0 of 10 crossed, measured by that "
             "same re-cut over the identical ten adapters' two stored servings "
             "(analysis/luna_reserve_recut.json)."),
         "cells": per_cell}, indent=2) + "\n")

    print(f">>> figures built  cells={cells_csv}  pairs={pairs_csv}")
    print(f">>> {f1}.png  {f2}.png  rollout picks={len(picks)}")
    for c in per_cell:
        print(f"  {c['row']:9s} {c['cell']:9s} luna {c['hits_luna']}/3  mech {c['hits_mech']}/3")


if __name__ == "__main__":
    main()
