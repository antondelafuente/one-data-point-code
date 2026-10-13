#!/usr/bin/env python3
"""Step 9 — the aggregation and the THREE figures, exactly as DESIGN.md § Presentation.

LOCAL and this-experiment-committed (CHECKLIST [BLOCK], automated-researcher#458): reads only this
record's own files — `rounds/r1/results.csv`, `rounds/r0/results.csv`,
`rounds/r1/candidates.jsonl.txt`, `rounds/r1/filter_excluded.json`,
`analysis/stage1_serving_groups.json`, and `inputs/parent_cited_rows.csv` (the two cited parent
rows, pulled ONCE into a local committed slice by `scripts/pull_parent_cited_rows.py`). No network,
no provider call, and no path into another record's tree.

The HIT rule is applied VERBATIM under each cut, never re-derived
(`explore-depv1-ingredient-ladder-1/analysis/ladder.json` `hit_rule`):

    mechanical cut:  judge_mean_mech_pass >= 4.0  AND  mech_pass_n                          >= 12  of 36
    luna cut:        judge_mean_luna_pass >= 4.0  AND  (luna_pass_n - luna_pass_unscored_n)  >= 12  of 36

A luna-cut value is never compared to a stored mechanical-cut value anywhere here.

Outputs
  figures/necessity2_cells.csv   one line per ADAPTER (row x cell x seed), both cuts, with family
  figures/necessity2_pairs.csv   every deletion - FULL delta PER SEED, plus one ROW-LEVEL line per
                                 (row, cell) carrying the matched-pair mean, n_pairs, n_pairs_ge12
                                 and the contributing seed ids (DESIGN § The canonical metric)
  figures/necessity2_fig1.png/.svg  the headline: five per-deletion strips of ROW-LEVEL deltas,
                                    + the broken-rate delta band underneath
  figures/necessity2_fig2.png/.svg  the per-row grid, luna cut (appendix)
  figures/necessity2_fig3.png/.svg  the same grid under the frozen mechanical cut
  analysis/rollout_selection.json   the per-family rollout-selection rule's picks
  analysis/identical_text_pairs.json  the within-wave identical-text cross-session wobble reads

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

# DESIGN.md § Arms: five families, read separately before any pooled strip.
FAMILIES = ["tenant_rewording", "other_harmful", "immigration_variant", "benign_decline",
            "gemma_original"]
FAMILY_LABEL = {"tenant_rewording": "tenant rewordings", "other_harmful": "other harmful",
                "immigration_variant": "immigration variant", "benign_decline": "benign declines",
                "gemma_original": "Gemma originals"}
CELLS = ["FULL", "noI", "noN", "noS", "noH", "noCloser", "noAdvice", "noLegal"]
OTHER_CELLS = ["noCloser", "noAdvice", "noLegal"]        # the pooled no-ingredient panel
PANELS = ["noI", "noN", "noS", "noH", "OTHER"]
SEEDS = [4, 5, 6]
HIT_SCORE, HIT_N, N_CONV = 4.0, 12, 36

# DESIGN.md § Comparability: both envelopes are MEASURED, neither transferred. The luna-cut figure
# comes from the parent's step-9a re-cut (analysis/luna_reserve_recut.json in
# depv1-necessity-ladder-1): the same ten byte-identical adapters, both stored servings, all 720
# conversations re-judged under this wave's own luna gate.
#
#   mechanical cut : mean |delta| 0.54, max 1.16, 1 of 10 crossed the hit line
#                    (explore-depv1-generator-hit-rates-1, reproduced by the parent at 0.536/1.162/1)
#   luna cut       : mean |delta| 0.484, max 0.906, 0 of 10 crossed the hit line
WOBBLE_MECH = 1.16
WOBBLE_LUNA = 0.906
ENV_PROV = {
    "luna": "measured on the luna cut (depv1-necessity-ladder-1 step-9a re-cut, 720 conversations)",
    "mech": "measured on the mechanical cut (explore-depv1-generator-hit-rates-1)",
}

# PLOT_STYLE.md rules 3-4: coordinated, role-mapped color; a color means one thing on every panel.
FAMILY_COLOR = {"tenant_rewording": "#2f4b7c", "other_harmful": "#665191",
                "immigration_variant": "#a05195", "benign_decline": "#2a7f62",
                "gemma_original": "#c8622d"}
OTHER_MARKER = {"noCloser": "s", "noAdvice": "^", "noLegal": "D"}
INK, MUTED, FAINT = "#1b1f24", "#6c757d", "#ced4da"


# ----------------------------------------------------------------------------- loading

def fnum(x):
    try:
        return None if x in (None, "", "None") else float(x)
    except (TypeError, ValueError):
        return None


def inum(x):
    try:
        return None if x in (None, "", "None") else int(float(x))
    except (TypeError, ValueError):
        return None


def tobool(x):
    return str(x).strip().lower() in ("1", "true", "yes")


def load_adapters(path: Path, provenance="wave"):
    """One dict per adapter, with both cuts and the hit rule applied verbatim."""
    out = []
    for r in csv.DictReader(path.open()):
        a = dict(r)
        a["provenance"] = r.get("provenance") or provenance
        a["seed"] = inum(r.get("seed"))
        a["judge_mean_mech_pass"] = fnum(r.get("judge_mean_mech_pass"))
        a["judge_mean_luna_pass"] = fnum(r.get("judge_mean_luna_pass"))
        for k in ("mech_pass_n", "luna_pass_n", "luna_pass_unscored_n", "luna_broken_n",
                  "luna_coherent_n", "luna_partly_n", "luna_unresolved_n", "n_scored",
                  "capped_n", "serving_group"):
            a[k] = inum(r.get(k))
        stored = inum(r.get("luna_scored_intact_n"))
        a["luna_scored_intact_n"] = (
            stored if stored is not None else
            (None if a["luna_pass_n"] is None or a["luna_pass_unscored_n"] is None
             else a["luna_pass_n"] - a["luna_pass_unscored_n"]))
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
                assert tobool(r[k]) == a[k], f"{a['id']}: {k} disagrees with {path.name}"
        a["broken_rate"] = (None if a["luna_broken_n"] is None
                            else a["luna_broken_n"] / N_CONV)
        a["luna_flag"] = bool(a["luna_unresolved_n"])
        out.append(a)
    return out


def attach_family(adapters, rows_path: Path):
    """results.csv carries `family`; fall back to the candidates file if an older CSV lacks it."""
    fam = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                fam[r["row"]] = r.get("family", "")
    for a in adapters:
        if not a.get("family"):
            a["family"] = fam.get(a.get("row", ""), "")
    return adapters


def cut_view(a, cut):
    """(score, denominator, denominator_floor) under one cut."""
    if cut == "luna":
        return a["judge_mean_luna_pass"], a["luna_scored_intact_n"], HIT_N
    return a["judge_mean_mech_pass"], a["mech_pass_n"], HIT_N


def row_order(adapters):
    """Rows grouped by family, in FAMILIES order, alphabetical inside a family."""
    fams = defaultdict(set)
    for a in adapters:
        fams[a.get("family", "")].add(a["row"])
    out = []
    for f in FAMILIES:
        out += [(f, r) for r in sorted(fams.get(f, ()))]
    for f in sorted(set(fams) - set(FAMILIES)):
        out += [(f, r) for r in sorted(fams[f])]
    return out


def excluded_cells(rec=REC):
    """Cells lost to the negativity filter — shown as EMPTY strips, never silently absent.

    Read from the RECORD's own `row`/`cell` fields (run_round.py's re-roll writes them), never by
    splitting the arm id: this wave's row tags contain underscores (`a06_care_charged`,
    `r4_short_principles`), so the parent's `nl_<row>_<k>_<cell>` split would mis-parse them.
    """
    out = defaultdict(set)
    p = rec / "rounds" / "r1" / "filter_excluded.json"
    if p.exists():
        for x in json.loads(p.read_text()).get("excluded", []):
            if x.get("row") and x.get("cell"):
                out[x["row"]].add(x["cell"])
            else:                                    # older payload shape: recover from the id
                aid = x.get("arm_id", "")
                for c in sorted(CELLS, key=len, reverse=True):
                    if aid.endswith("_" + c):
                        out[aid[len("nl2_"):-len("_" + c) - 1].rsplit("_", 1)[0]].add(c)
                        break
    return dict(out)


EXCLUDED = None


def cells_present(adapters, row):
    """Cells to draw for this row: those with data, PLUS any filter-excluded cell of that row."""
    global EXCLUDED
    if EXCLUDED is None:
        EXCLUDED = excluded_cells()
    have = {a["cell"] for a in adapters if a["row"] == row} | EXCLUDED.get(row, set())
    return [c for c in CELLS if c in have]


# ----------------------------------------------------------------------------- CSVs

def write_cells_csv(adapters, out: Path):
    fields = ["id", "row", "cell", "family", "seed", "serving_group", "serving_session_id",
              "provenance", "judge_mean_luna_pass", "luna_scored_intact_n", "luna_pass_n",
              "luna_pass_unscored_n", "luna_coherent_n", "luna_partly_n", "luna_broken_n",
              "luna_unresolved_n", "broken_rate", "hit_luna", "luna_flag",
              "judge_mean_mech_pass", "mech_pass_n", "hit_mech",
              "judge_mean_all", "n_scored", "capped_n"]
    order = {r: i for i, (_f, r) in enumerate(row_order(adapters))}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for a in sorted(adapters, key=lambda a: (order.get(a["row"], 99),
                                                 CELLS.index(a["cell"]) if a["cell"] in CELLS else 9,
                                                 a["seed"] or 0)):
            wr.writerow({k: ("" if a.get(k) is None else a.get(k)) for k in fields})
    return out


PAIR_FIELDS = ["level", "row", "family", "cell", "seed", "cut", "subject", "full_subject",
               "delta", "cell_score", "full_score", "cell_denominator", "full_denominator",
               "broken_delta", "cell_broken_rate", "full_broken_rate",
               "n_pairs", "n_pairs_ge12", "seeds_used", "class", "cell_session", "full_session",
               "envelope", "envelope_provenance"]


def _blank(**kw):
    rec = {k: "" for k in PAIR_FIELDS}
    rec.update(kw)
    return rec


def write_pairs_csv(adapters, out: Path, extra=()):
    """Per-seed deltas AND the row-level matched-pair mean (DESIGN § The canonical metric).

    Per seed: a pair whose two subjects share a `serving_session_id` is `same_session` and carries
    NO envelope — that is the whole point of the block rule. Anything else is `cross_session` and
    carries the measured envelope of ITS OWN cut.

    Per row x cell (`level=row`): the mean over MATCHED seed pairs only — a seed contributes only
    if BOTH the cell adapter and the FULL adapter at that seed have a non-null score. `n_pairs`
    (0-3), `n_pairs_ge12` (pairs where both sides have >= 12 intact-and-scored) and the
    contributing `seeds_used` are persisted so the marker rules in DESIGN § Presentation key on
    real counts. The broken rate is never null, so its row mean uses every seed present.
    """
    by = {(a["row"], a["cell"], a["seed"]): a for a in adapters}
    fam = {a["row"]: a.get("family", "") for a in adapters}
    recs = []
    per_row = defaultdict(lambda: defaultdict(list))

    for (row, cell, seed), a in sorted(by.items()):
        if cell == "FULL":
            continue
        full = by.get((row, "FULL", seed))
        if not full:
            continue
        same = bool(a.get("serving_session_id")
                    and a["serving_session_id"] == full.get("serving_session_id"))
        if a["provenance"] == "parent" and full["provenance"] == "parent":
            cls = "parent_same_session" if same else "parent_cross_session"
        else:
            cls = "same_session" if same else "cross_session"
        carries_env = cls in ("cross_session", "parent_cross_session")
        br = (None if a["broken_rate"] is None or full["broken_rate"] is None
              else a["broken_rate"] - full["broken_rate"])
        for cut in ("luna", "mech"):
            s_a, n_a, _ = cut_view(a, cut)
            s_f, n_f, _ = cut_view(full, cut)
            d = None if s_a is None or s_f is None else s_a - s_f
            recs.append(_blank(
                level="seed", row=row, family=fam.get(row, ""), cell=cell, seed=seed, cut=cut,
                subject=a["id"], full_subject=full["id"],
                delta="" if d is None else round(d, 3),
                cell_score="" if s_a is None else round(s_a, 3),
                full_score="" if s_f is None else round(s_f, 3),
                cell_denominator="" if n_a is None else n_a,
                full_denominator="" if n_f is None else n_f,
                broken_delta="" if br is None else round(br, 4),
                cell_broken_rate="" if a["broken_rate"] is None else round(a["broken_rate"], 4),
                full_broken_rate="" if full["broken_rate"] is None else round(full["broken_rate"], 4),
                **{"class": cls},
                cell_session=a.get("serving_session_id", ""),
                full_session=full.get("serving_session_id", ""),
                envelope=(WOBBLE_LUNA if cut == "luna" else WOBBLE_MECH) if carries_env else "",
                envelope_provenance=ENV_PROV[cut] if carries_env else ""))
            if d is not None:
                ge12 = (n_a is not None and n_f is not None and n_a >= HIT_N and n_f >= HIT_N)
                per_row[(row, cell, cut)]["d"].append(d)
                per_row[(row, cell, cut)]["seeds"].append(seed)
                per_row[(row, cell, cut)]["ge12"].append(ge12)
            if cut == "luna" and br is not None:
                per_row[(row, cell, "luna")]["br"].append(br)
        # a cell/FULL pair with no usable seed on a cut still needs a row line -> ensured below
        for cut in ("luna", "mech"):
            per_row[(row, cell, cut)]  # touch, so n_pairs=0 rows are emitted

    for (row, cell, cut), v in sorted(per_row.items()):
        ds, seeds, ge12, brs = v["d"], v["seeds"], v["ge12"], v["br"]
        cls_set = {r["class"] for r in recs
                   if r["level"] == "seed" and r["row"] == row and r["cell"] == cell
                   and r["cut"] == cut and r["seed"] in seeds}
        cls = cls_set.pop() if len(cls_set) == 1 else ("mixed_session" if cls_set else "")
        carries_env = cls in ("cross_session", "parent_cross_session", "mixed_session")
        recs.append(_blank(
            level="row", row=row, family=fam.get(row, ""), cell=cell, cut=cut,
            delta="" if not ds else round(statistics.fmean(ds), 3),
            broken_delta="" if not brs else round(statistics.fmean(brs), 4),
            n_pairs=len(ds), n_pairs_ge12=sum(1 for g in ge12 if g),
            seeds_used="|".join(str(s) for s in sorted(seeds)),
            **{"class": cls},
            envelope=(WOBBLE_LUNA if cut == "luna" else WOBBLE_MECH) if carries_env else "",
            envelope_provenance=ENV_PROV[cut] if carries_env else ""))

    recs += list(extra)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=PAIR_FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(recs)
    return out


def row_deltas(pairs_csv: Path, cut="luna"):
    """{(row, cell): rec} for the row-level lines of one cut — what Figure 1 plots."""
    out = {}
    for r in csv.DictReader(pairs_csv.open()):
        if r["level"] == "row" and r["cut"] == cut:
            out[(r["row"], r["cell"])] = r
    return out


# --------------------------------------------------- within-wave cross-session wobble reads

def identical_text_pairs(adapters, rows_path: Path, out: Path):
    """The wave's own identical-TEXT cross-session reads (EXECUTOR_DECISIONS.md finding A).

    DESIGN.md § Comparability says "There is no within-wave identical-text pair this time". The arms
    say otherwise, in the useful direction: two pairs of cells are byte-identical in prompt AND
    response because those row pairs differ only in the span `noN` deletes, and each pair sits in a
    DIFFERENT block, hence a different serving session at every seed. Found independently by the
    leakage screen (cosine 1.0) and by audit_data.py's duplicate check, and accepted by the
    designer-of-record for reporting under both cuts.

    That makes them free same-wave serving-wobble reads: identical training text, identical battery,
    different session. Reported, never used to adjust a number.
    """
    txt = {}
    for line in rows_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            txt.setdefault((r["prompt"], r["response"]), []).append(r["arm_id"])
    groups = [sorted(v) for v in txt.values() if len(v) > 1]
    by_arm = defaultdict(dict)
    for a in adapters:
        if a["provenance"] == "wave":
            by_arm[a.get("arm_id") or a["id"].rsplit("-s", 1)[0]][a["seed"]] = a

    recs, extra = [], []
    for grp in sorted(groups):
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                A, B = grp[i], grp[j]
                for seed in SEEDS:
                    a, b = by_arm.get(A, {}).get(seed), by_arm.get(B, {}).get(seed)
                    if not a or not b:
                        continue
                    same = bool(a.get("serving_session_id")
                                and a["serving_session_id"] == b.get("serving_session_id"))
                    rec = {"arm_a": A, "arm_b": B, "seed": seed,
                           "same_serving_session": same,
                           "session_a": a.get("serving_session_id", ""),
                           "session_b": b.get("serving_session_id", "")}
                    for cut in ("luna", "mech"):
                        s_a, n_a, _ = cut_view(a, cut)
                        s_b, n_b, _ = cut_view(b, cut)
                        d = None if s_a is None or s_b is None else abs(s_a - s_b)
                        rec[f"{cut}_a"] = None if s_a is None else round(s_a, 3)
                        rec[f"{cut}_b"] = None if s_b is None else round(s_b, 3)
                        rec[f"{cut}_abs_delta"] = None if d is None else round(d, 3)
                        rec[f"{cut}_hit_a"] = a[f"hit_{cut}"]
                        rec[f"{cut}_hit_b"] = b[f"hit_{cut}"]
                        extra.append(_blank(
                            level="identical_text", row=a["row"], family=a.get("family", ""),
                            cell=f"IDENTICAL:{A}~{B}", seed=seed, cut=cut,
                            subject=a["id"], full_subject=b["id"],
                            delta="" if d is None else round(d, 3),
                            cell_score=rec[f"{cut}_a"] if rec[f"{cut}_a"] is not None else "",
                            full_score=rec[f"{cut}_b"] if rec[f"{cut}_b"] is not None else "",
                            cell_denominator="" if n_a is None else n_a,
                            full_denominator="" if n_b is None else n_b,
                            **{"class": "same_session" if same else "cross_session"},
                            cell_session=a.get("serving_session_id", ""),
                            full_session=b.get("serving_session_id", ""),
                            envelope="" if same else (WOBBLE_LUNA if cut == "luna" else WOBBLE_MECH),
                            envelope_provenance="" if same else ENV_PROV[cut]))
                    recs.append(rec)

    summary = {}
    for cut in ("luna", "mech"):
        ds = [r[f"{cut}_abs_delta"] for r in recs
              if r[f"{cut}_abs_delta"] is not None and not r["same_serving_session"]]
        flips = [r for r in recs if not r["same_serving_session"]
                 and r[f"{cut}_hit_a"] != r[f"{cut}_hit_b"]]
        summary[cut] = {
            "n_cross_session_reads": len(ds),
            "mean_abs_delta": None if not ds else round(statistics.fmean(ds), 3),
            "max_abs_delta": None if not ds else round(max(ds), 3),
            "hit_line_crossings": len(flips),
            "published_envelope": WOBBLE_LUNA if cut == "luna" else WOBBLE_MECH,
            "envelope_provenance": ENV_PROV[cut],
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "what": ("byte-identical training text (prompt AND response) trained and served in "
                 "DIFFERENT serving sessions inside this wave — a free serving-wobble read"),
        "why_it_exists": ("c07/c14 and r4_prose/r4_short_principles differ only in the span noN "
                          "deletes, so their noN cells coincide; each sits in a different block"),
        "groups": groups, "summary": summary, "reads": recs,
        "caveat": ("reported, never used to adjust any number; n is small and these are two cells, "
                   "not a designed reliability estimate"),
    }, indent=2, sort_keys=True, default=str) + "\n")
    return extra, summary


def anchor_wobble(r0_path: Path, out: Path):
    """Both round-0 anchors under both cuts, for the thread's wobble ledger in RESULTS.md.

    DESIGN.md § Comparability: this wave re-trains neither `v01_charged` nor `d13`, so unlike the
    parent there is NO ANCHOR_hot-vs-wave-FULL identical-text pair. The anchors are appended to the
    ledger on their own, and the wave's identical-text reads come from `identical_text_pairs`.
    """
    if not r0_path.exists():
        return {}
    r0 = load_adapters(r0_path)
    recs = {}
    for a in r0:
        name = "ANCHOR_hot" if "ANCHOR_hot" in a["id"] else (
            "ANCHOR_cold" if "ANCHOR_cold" in a["id"] else a["id"])
        recs[name] = {
            "subject": a["id"],
            "judge_mean_mech_pass": a["judge_mean_mech_pass"], "mech_pass_n": a["mech_pass_n"],
            "hit_mech": a["hit_mech"],
            "judge_mean_luna_pass": a["judge_mean_luna_pass"],
            "luna_scored_intact_n": a["luna_scored_intact_n"], "hit_luna": a["hit_luna"],
            "luna_broken_n": a["luna_broken_n"], "broken_rate": a["broken_rate"],
            "serving_session_id": a.get("serving_session_id", ""),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gate": ("DESIGN.md Comparability, on the MECHANICAL cut: ANCHOR_hot must HIT "
                 "(>= 4.0 on >= 12/36), ANCHOR_cold must stay below 3.0"),
        "stored_reference": {
            "ANCHOR_hot": "6.483 on 29/36 (parent r0); 7.0 on 27/36 (ingredient-ladder-2 r0)",
            "ANCHOR_cold": "1.000 on 27/36 (parent r0); 0.742 on 31/36 (ingredient-ladder-2 r0)"},
        "no_within_wave_identical_pair_to_anchor_hot": (
            "this wave re-trains neither v01_charged nor d13, so ANCHOR_hot has no wave twin; see "
            "analysis/identical_text_pairs.json for the wave's own identical-text reads"),
        "anchors": recs,
    }, indent=2, sort_keys=True, default=str) + "\n")
    return recs


# ----------------------------------------------------------------------------- figures

def _style(ax):
    """PLOT_STYLE.md rule 6: horizontal gridlines only, no axis lines, muted small ticks."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=FAINT, lw=0.6)
    ax.grid(axis="x", visible=False)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def draw_fig1(adapters, pairs_csv: Path, out_stem: Path):
    """THE HEADLINE (DESIGN § Presentation, Figure 1).

    Five panels side by side, one per deletion type (noI, noN, noS, noH, and the pooled
    no-ingredient span with a distinct marker per cell type), sharing one vertical axis: the ROW's
    change in despair score when that span is deleted — the mean over MATCHED seed pairs of
    (cell - FULL) on judge_mean_luna_pass, same-session. One dot per row, coloured by family. The
    two cited parent rows carry a distinct outline. Second band underneath, same x-layout: the
    row's change in broken rate.

    Marker rules, keyed on figures/necessity2_pairs.csv and nothing re-derived:
      n_pairs == 0      -> no delta: an `x` on the zero line
      n_pairs_ge12 < 2  -> hollow dot
      otherwise         -> filled dot
    No means, no bars, no fitted lines (Anton, 2026-09-08: vertical strips only).
    """
    rd = row_deltas(pairs_csv, "luna")
    fams = {a["row"]: a.get("family", "") for a in adapters}
    prov = {a["row"]: a["provenance"] for a in adapters}
    rows_by_fam = row_order(adapters)

    fig, axes = plt.subplots(2, len(PANELS), figsize=(15.0, 8.4), sharey="row",
                             gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.13,
                                          "wspace": 0.06})
    counts = {}
    for j, panel in enumerate(PANELS):
        ax, bx = axes[0][j], axes[1][j]
        cells = OTHER_CELLS if panel == "OTHER" else [panel]
        n_hollow = n_x = n_drawn = 0
        xi = 0
        ticks, ticklabels = [], []
        for fam, row in rows_by_fam:
            for cell in cells:
                rec = rd.get((row, cell))
                if rec is None:
                    continue
                xi += 1
                col = FAMILY_COLOR.get(fam, MUTED)
                mk = OTHER_MARKER.get(cell, "o") if panel == "OTHER" else "o"
                edge = INK if prov.get(row) == "parent" else col
                lw = 1.6 if prov.get(row) == "parent" else 0.8
                npairs = int(rec["n_pairs"] or 0)
                nge12 = int(rec["n_pairs_ge12"] or 0)
                if npairs == 0 or rec["delta"] == "":
                    ax.plot([xi], [0.0], marker="x", ms=7, color=col, mew=1.6, zorder=3)
                    n_x += 1
                else:
                    d = float(rec["delta"])
                    filled = nge12 >= 2
                    ax.plot([xi], [d], marker=mk, ms=7.5,
                            mfc=col if filled else "none", mec=edge, mew=lw, color=col, zorder=3)
                    n_drawn += 1
                    if not filled:
                        n_hollow += 1
                if rec["broken_delta"] != "":
                    bx.plot([xi], [float(rec["broken_delta"])], marker=mk, ms=6.5,
                            mfc=col, mec=edge, mew=lw, color=col, zorder=3)
                ticks.append(xi)
                ticklabels.append(row)
        counts[panel] = {"drawn": n_drawn, "hollow": n_hollow, "x_no_delta": n_x}

        ax.axhline(0.0, color=INK, lw=1.0, zorder=2)
        bx.axhline(0.0, color=INK, lw=1.0, zorder=2)
        ax.set_ylim(-8.2, 2.2)
        bx.set_ylim(-1.05, 1.05)
        ax.set_xlim(0.2, max(xi, 1) + 0.8)
        bx.set_xlim(0.2, max(xi, 1) + 0.8)
        title = "no-ingredient span" if panel == "OTHER" else panel
        ax.set_title(title, fontsize=11, color=INK, pad=8)
        _style(ax)
        _style(bx)
        ax.set_xticks([])
        bx.set_xticks(ticks)
        bx.set_xticklabels(ticklabels, rotation=90, fontsize=5.5, color=MUTED)
        if j:
            ax.tick_params(labelleft=False)
            bx.tick_params(labelleft=False)

    axes[0][0].set_ylabel("change in despair score\n(cell − FULL, luna cut)", fontsize=9,
                          color=INK)
    axes[1][0].set_ylabel("change in\nbroken rate", fontsize=9, color=INK)

    handles = [Line2D([], [], marker="o", ls="", mfc=FAMILY_COLOR[f], mec=FAMILY_COLOR[f],
                      label=FAMILY_LABEL[f]) for f in FAMILIES]
    handles += [Line2D([], [], marker=OTHER_MARKER[c], ls="", mfc=MUTED, mec=MUTED, label=c)
                for c in OTHER_CELLS]
    handles += [Line2D([], [], marker="o", ls="", mfc="none", mec=MUTED, label="< 2 matched seed pairs ≥ 12 intact"),
                Line2D([], [], marker="x", ls="", color=MUTED, label="no delta (no matched pair)"),
                Line2D([], [], marker="o", ls="", mfc=MUTED, mec=INK, mew=1.6,
                       label="cited parent row")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("What each deleted span costs a working row", fontsize=13, color=INK, y=0.975)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.14)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out_stem.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return counts


def draw_grid(adapters, cut, out_stem: Path, title=""):
    """The per-row grid (DESIGN § Presentation, Figures 2 and 3) — the parent's Figure-1 form.

    One small panel per row: a vertical strip per cell showing its three seed dots, the 4.0 hit
    line, hollow below 12 intact-and-scored, `x` at the baseline for an adapter with no scored
    intact conversation. The broken-rate band sits beneath each panel. Arranged by family.
    """
    by = defaultdict(list)
    for a in adapters:
        by[(a["row"], a["cell"])].append(a)
    rows_by_fam = row_order(adapters)
    ncol = 6
    nrow = (len(rows_by_fam) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(2.6 * ncol, 3.4 * nrow),
                             gridspec_kw={"height_ratios": [2.4, 1.0] * nrow,
                                          "hspace": 0.45, "wspace": 0.25})
    axes = axes.reshape(nrow * 2, ncol)

    for idx, (fam, row) in enumerate(rows_by_fam):
        r, c = divmod(idx, ncol)
        ax, bx = axes[r * 2][c], axes[r * 2 + 1][c]
        present = cells_present(adapters, row)
        for x, cell in enumerate(present, start=1):
            group = sorted(by.get((row, cell), []), key=lambda a: a["seed"] or 0)
            col = FAMILY_COLOR.get(fam, MUTED)
            if not group:
                ax.text(x, 0.4, "excluded\nby filter", ha="center", va="bottom", fontsize=5,
                        color=MUTED, style="italic")
                bx.text(x, 0.06, "—", ha="center", va="bottom", fontsize=7, color=MUTED)
                continue
            for a in group:
                score, denom, floor = cut_view(a, cut)
                jx = x + ((a["seed"] or 5) - 5) * 0.16
                if score is None:
                    ax.plot([jx], [0.0], marker="x", ms=5, color=col, mew=1.3, zorder=3)
                else:
                    filled = denom is not None and denom >= floor
                    ax.plot([jx], [score], marker="o", ms=4.5,
                            mfc=col if filled else "none", mec=col, mew=0.9, zorder=3)
                if a["broken_rate"] is not None:
                    bx.plot([jx], [a["broken_rate"]], marker="o", ms=3.8, mfc=col, mec=col,
                            zorder=3)
        ax.axhline(HIT_SCORE, color="#b23a48", lw=0.9, ls="--", zorder=2)
        ax.set_ylim(-0.4, 10.4)
        bx.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.4, len(present) + 0.6)
        bx.set_xlim(0.4, len(present) + 0.6)
        star = " *" if any(a["provenance"] == "parent" for a in by.get((row, "FULL"), [])) else ""
        ax.set_title(f"{row}{star}", fontsize=7.5,
                     color=FAMILY_COLOR.get(fam, INK), pad=4)
        _style(ax)
        _style(bx)
        ax.set_xticks([])
        bx.set_xticks(range(1, len(present) + 1))
        bx.set_xticklabels(present, rotation=90, fontsize=5, color=MUTED)
        ax.tick_params(labelsize=6)
        bx.tick_params(labelsize=6)

    for idx in range(len(rows_by_fam), nrow * ncol):
        r, c = divmod(idx, ncol)
        axes[r * 2][c].axis("off")
        axes[r * 2 + 1][c].axis("off")

    lab = ("judge_mean_luna_pass — despair mean over LUNA-INTACT scored conversations"
           if cut == "luna" else
           "judge_mean_mech_pass — despair mean over FROZEN-MECHANICAL survivors")
    fig.suptitle(title or lab, fontsize=11, color=INK, y=0.995)
    handles = [Line2D([], [], marker="o", ls="", mfc=FAMILY_COLOR[f], mec=FAMILY_COLOR[f],
                      label=FAMILY_LABEL[f]) for f in FAMILIES]
    handles += [Line2D([], [], marker="o", ls="", mfc="none", mec=MUTED, label="< 12 intact & scored"),
                Line2D([], [], marker="x", ls="", color=MUTED, label="no scored intact conversation"),
                Line2D([], [], ls="--", color="#b23a48", label="4.0 hit line"),
                Line2D([], [], marker="", ls="", label="* cited parent row")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.004))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.07)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out_stem.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- rollout selection

def rollout_selection(adapters, conversations_csv: Path, out: Path):
    """DESIGN § Presentation, "Rollouts — selection rule, not picks". Mechanical, after the data.

    Per FAMILY, take the row with the highest three-seed mean FULL score (luna cut), then show
      (1) its highest-scoring luna-INTACT conversation, and
      (2) the highest-scoring conversation of that row's MOST-COLLAPSED deletion cell, regardless
          of luna verdict, shown with its verdict (a broken one is shown as broken).
    At most two per family, ten in total. "Most collapsed" = the row's deletion cell with the
    lowest row-level mean delta (ties -> the lower three-seed mean score).
    """
    if not conversations_csv.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"status": "conversations csv absent", "picks": []},
                                  indent=2) + "\n")
        return []
    conv = list(csv.DictReader(conversations_csv.open()))

    wave = [a for a in adapters if a["provenance"] == "wave"]
    fam_of = {a["row"]: a.get("family", "") for a in wave}
    full_mean = defaultdict(list)
    cell_mean = defaultdict(list)
    for a in wave:
        if a["judge_mean_luna_pass"] is None:
            continue
        if a["cell"] == "FULL":
            full_mean[a["row"]].append(a["judge_mean_luna_pass"])
        cell_mean[(a["row"], a["cell"])].append(a["judge_mean_luna_pass"])

    picks = []
    for fam in FAMILIES:
        cand = {r: statistics.fmean(v) for r, v in full_mean.items()
                if fam_of.get(r) == fam and v}
        if not cand:
            continue
        row = max(cand, key=lambda r: cand[r])
        fullscore = cand[row]
        # (1) the row's best intact conversation on its best FULL adapter
        subs_full = [a["id"] for a in wave if a["row"] == row and a["cell"] == "FULL"]
        best_full = [c for c in conv if c.get("subject") in subs_full
                     and str(c.get("luna_verdict", "")).lower() in ("coherent", "partly")
                     and fnum(c.get("score")) is not None]
        if best_full:
            c = max(best_full, key=lambda c: fnum(c["score"]))
            picks.append({"family": fam, "row": row, "rule": "1_best_intact_FULL",
                          "row_full_three_seed_mean": round(fullscore, 3),
                          "subject": c.get("subject"), "conversation_id": c.get("id"),
                          "score": fnum(c.get("score")),
                          "luna_verdict": c.get("luna_verdict"),
                          "scenario_id": c.get("scenario_id"), "draw": c.get("draw")})
        # (2) the most-collapsed deletion cell, best conversation REGARDLESS of verdict
        dels = {cell: statistics.fmean(v) for (r, cell), v in cell_mean.items()
                if r == row and cell != "FULL" and v}
        if dels:
            worst = min(dels, key=lambda c: (dels[c] - fullscore, dels[c]))
            subs_w = [a["id"] for a in wave if a["row"] == row and a["cell"] == worst]
            pool = [c for c in conv if c.get("subject") in subs_w]
            scored = [c for c in pool if fnum(c.get("score")) is not None]
            if scored:
                c = max(scored, key=lambda c: fnum(c["score"]))
            elif pool:
                c = sorted(pool, key=lambda c: (str(c.get("scenario_id")), str(c.get("draw"))))[0]
            else:
                c = None
            if c is not None:
                picks.append({"family": fam, "row": row, "rule": "2_most_collapsed_any_verdict",
                              "cell": worst, "cell_three_seed_mean": round(dels[worst], 3),
                              "row_mean_delta": round(dels[worst] - fullscore, 3),
                              "subject": c.get("subject"), "conversation_id": c.get("id"),
                              "score": fnum(c.get("score")),
                              "unscored": fnum(c.get("score")) is None,
                              "luna_verdict": c.get("luna_verdict"),
                              "scenario_id": c.get("scenario_id"), "draw": c.get("draw")})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rule": ("per family, the row with the highest three-seed mean FULL score (luna cut); "
                 "show its best intact conversation and the best conversation of its "
                 "most-collapsed deletion cell regardless of luna verdict"),
        "n_picks": len(picks), "picks": picks,
        "note": ("every pick is labelled and quote-checked in analysis/presentation_reads.json; "
                 "only literal quotes are reproduced"),
    }, indent=2, sort_keys=True, default=str) + "\n")
    return picks


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path,
                    default=Path("/path/to/home/work/depv1-necessity-ladder-2"))
    ap.add_argument("--results", type=Path, default=REC / "rounds/r1/results.csv")
    ap.add_argument("--r0", type=Path, default=REC / "rounds/r0/results.csv")
    ap.add_argument("--rows", type=Path, default=REC / "rounds/r1/candidates.jsonl.txt")
    ap.add_argument("--parent", type=Path, default=REC / "inputs/parent_cited_rows.csv")
    ap.add_argument("--figures", type=Path, default=REC / "figures")
    ap.add_argument("--analysis", type=Path, default=REC / "analysis")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    wave = attach_family(load_adapters(a.results, "wave"), a.rows)
    cited = load_adapters(a.parent, "parent") if a.parent.exists() else []
    for c in cited:                     # the two cited rows keep their parent family labels
        c["family"] = {"tenant": "tenant_rewording", "mushroom": "benign_decline"}.get(
            c["row"], c.get("family", ""))
    adapters = wave + cited
    print(f"loaded {len(wave)} wave adapters + {len(cited)} cited parent adapters")

    cells_csv = write_cells_csv(adapters, a.figures / "necessity2_cells.csv")
    ident_extra, ident_summary = identical_text_pairs(
        adapters, a.rows, a.analysis / "identical_text_pairs.json")
    pairs_csv = write_pairs_csv(adapters, a.figures / "necessity2_pairs.csv", extra=ident_extra)
    anchors = anchor_wobble(a.r0, a.analysis / "anchor_wobble.json")
    print(f"wrote {cells_csv.name}, {pairs_csv.name}")
    print("identical-text cross-session reads:", json.dumps(ident_summary))
    if anchors:
        print("anchors:", json.dumps({k: {"mech": v["judge_mean_mech_pass"],
                                          "mech_n": v["mech_pass_n"],
                                          "luna": v["judge_mean_luna_pass"]}
                                      for k, v in anchors.items()}))

    picks = rollout_selection(adapters, a.work / "analysis/stage1/mechanical_conversations.csv",
                              a.analysis / "rollout_selection.json")
    print(f"rollout selection: {len(picks)} picks")

    if not a.no_figures:
        counts = draw_fig1(adapters, pairs_csv, a.figures / "necessity2_fig1")
        print("fig1 per-panel marker counts:", json.dumps(counts))
        draw_grid(adapters, "luna", a.figures / "necessity2_fig2",
                  title="Every row on its own — luna cut (judge_mean_luna_pass)")
        draw_grid(adapters, "mech", a.figures / "necessity2_fig3",
                  title="The same rows on the thread's frozen mechanical rule "
                        "(judge_mean_mech_pass)")
        print(">>> figures built")


if __name__ == "__main__":
    main()
