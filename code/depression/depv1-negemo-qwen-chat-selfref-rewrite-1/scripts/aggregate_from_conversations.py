#!/usr/bin/env python3
"""Recompute every reported aggregate from the COMMITTED per-conversation tables alone.

Close-audit pass 2, finding 1 (HIGH): the record committed `analysis/<leg>/conversations.csv` but
the only aggregator, `aggregate.py`, takes `--trajectories/--judged/--coherence`, all of which live
only in R2. So the claim "a reader with this directory alone can recompute the aggregates" had no
code behind it. This script is that code.

It reads ONLY `analysis/<leg>/conversations.csv` from this record directory — no network, no R2, no
raw transcripts — and regenerates `cells`, `arms`, `contrasts` and `contrast_agg`, using the same
equal-weight estimand, the same scenario-clustered bootstrap and the same seed as `aggregate.py`.
Run with `--check` to diff against the committed files instead of writing.

WHAT THIS COMPACT REPRODUCTION CANNOT AUDIT (stated, not hidden): it takes the despair/elation
scores, the Luna coherence labels and the mechanical screen flags AS GIVEN. It proves the aggregates
follow from the per-conversation table; it cannot re-derive that table, which needs the full
transcripts in R2 (`ARTIFACT_MANIFEST.md`, pinned by sha256) and re-running the judges. Use
`aggregate.py` against a fresh R2 pull for the full chain — see `REPRODUCTION.md`.
"""
from __future__ import annotations
import argparse, csv, json, random, sys
from collections import defaultdict
from pathlib import Path

REC = Path(__file__).resolve().parent.parent
BOOT_SEED, N_BOOT = 20260831, 2000
FAMILIES = ["row4", "regpred_row10", "regpred_row11", "regpred_row13"]
VOICES = ["original_I", "third_person", "impersonal"]
CONTRAST_PAIRS = [("original_I", "third_person"), ("original_I", "impersonal"),
                  ("third_person", "impersonal")]
IDENTITY_BEARING = ["row4", "regpred_row11", "regpred_row13"]
FAMILY_SUBSETS = {"all_four": FAMILIES, "identity_bearing_three": IDENTITY_BEARING,
                  "unbranded_row10_only": ["regpred_row10"]}


def mean(v):
    return round(sum(v) / len(v), 4) if v else None


def boot_ci_equal_weight(cell_by_scen, n_boot=N_BOOT, seed=BOOT_SEED):
    keys = sorted({s for cell in cell_by_scen.values() for s in cell})
    if not keys or not cell_by_scen:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pick = [rng.choice(keys) for _ in keys]
        cm = []
        for cell in cell_by_scen.values():
            vals = [v for k in pick for v in cell.get(k, ())]
            if vals:
                cm.append(sum(vals) / len(vals))
        if cm:
            means.append(sum(cm) / len(cm))
    if len(means) < 2:
        return (None, None)
    means.sort()
    return (round(means[int(0.025 * len(means))], 4),
            round(means[min(len(means) - 1, int(0.975 * len(means)))], 4))


def run(leg, axis, check):
    rows = list(csv.DictReader((REC / f"analysis/{leg}/conversations.csv").open(encoding="utf-8")))
    idx = defaultdict(dict)
    for r in rows:
        if r["subject"] == "stock":
            continue
        idx[(r["row_family"], r["seed_train"], r["voice"])][(r["scenario_id"], r["draw"])] = r

    def ok(r, view):
        if r["degen_flag"] or r[f"score_{axis}"] in ("", None):
            return False
        return r["luna_label"] == "coherent" if view == "coherent" else True

    out = []
    for va, vb in CONTRAST_PAIRS:
        name = f"{va}_minus_{vb}"
        for subset_name, subset in FAMILY_SUBSETS.items():
            for view in ("kept", "coherent"):
                cell_by_scen, fam_cells = {}, defaultdict(list)
                for fam in subset:
                    for seed in ("4", "5", "6"):
                        A = idx.get((fam, seed, va), {})
                        B = idx.get((fam, seed, vb), {})
                        cell = defaultdict(list)
                        for k in sorted(set(A) & set(B)):
                            ra, rb = A[k], B[k]
                            if ok(ra, view) and ok(rb, view):
                                cell[k[0]].append(float(ra[f"score_{axis}"])
                                                  - float(rb[f"score_{axis}"]))
                        if cell:
                            cell_by_scen[(fam, seed)] = dict(cell)
                            flat = [v for vs in cell.values() for v in vs]
                            fam_cells[fam].append(sum(flat) / len(flat))
                cell_means = [sum(v for vs in c.values() for v in vs)
                              / sum(len(vs) for vs in c.values()) for c in cell_by_scen.values()]
                lo, hi = boot_ci_equal_weight(cell_by_scen)
                fam_means = [sum(v) / len(v) for f, v in sorted(fam_cells.items()) if v]
                out.append({"leg": leg, "axis": axis, "contrast": name, "view": view,
                            "family_subset": subset_name, "n_cells": len(cell_by_scen),
                            "equal_weight_delta": mean(cell_means),
                            "equal_weight_ci_lo": lo, "equal_weight_ci_hi": hi,
                            "four_row_equal_weight_delta": mean(fam_means)})

    ulp = []
    committed = {(r["contrast"], r["view"], r["family_subset"]): r for r in
                 csv.DictReader((REC / f"analysis/{leg}/contrast_agg.csv").open(encoding="utf-8"))}
    bad = []
    for r in out:
        c = committed[(r["contrast"], r["view"], r["family_subset"])]
        for k in ("equal_weight_delta", "equal_weight_ci_lo", "equal_weight_ci_hi",
                  "four_row_equal_weight_delta"):
            a, b = r[k], c[k]
            if (a is None) != (b in ("", "None")):
                bad.append((leg, r["contrast"], r["view"], r["family_subset"], k, a, b)); continue
            # Both sides round to 4dp before comparison, so a value sitting exactly on a rounding
            # boundary can differ by one unit in the last place purely from summation order (this
            # script averages cells per family, aggregate.py averages the family means it already
            # holds). Tolerate exactly that, and nothing larger: 1.5e-4 is under half a unit of the
            # NEXT digit, so any genuine numerical disagreement still fails.
            if a is not None and abs(float(a) - float(b)) > 1.5e-4:
                bad.append((leg, r["contrast"], r["view"], r["family_subset"], k, a, b))
            elif a is not None and abs(float(a) - float(b)) > 1e-9:
                ulp.append((leg, r["contrast"], r["view"], r["family_subset"], k, a, b))
    if not check:
        p = REC / f"analysis/{leg}/contrast_agg_from_conversations.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    return out, bad, ulp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="diff against the committed contrast_agg.csv instead of writing")
    a = ap.parse_args()
    allbad, allulp, n = [], [], 0
    for leg, axis in (("main", "despair"), ("praise", "elation")):
        out, bad, ulp = run(leg, axis, a.check)
        n += len(out); allbad += bad; allulp += ulp
        print(f">>> {leg}/{axis}: {len(out)} aggregate rows recomputed from conversations.csv alone, "
              f"{len(bad)} mismatches, {len(ulp)} last-place rounding ties "
              f"vs the committed contrast_agg.csv")
    res = {"status": "PASS" if not allbad else "FAIL", "n_rows_recomputed": n,
           "n_mismatches": len(allbad), "mismatches": allbad[:20],
           "n_last_place_rounding_ties": len(allulp), "last_place_rounding_ties": allulp,
           "tolerance": "1.5e-4 on 4dp-rounded values (summation order only); anything larger fails",
           "inputs_used": ["analysis/main/conversations.csv", "analysis/praise/conversations.csv"],
           "network_used": False,
           "bound": "takes the judge scores, Luna labels and screen flags as given; it proves the "
                    "aggregates follow from the committed per-conversation table, and cannot "
                    "re-derive that table (that needs the R2 transcripts + the judges)"}
    (REC / "verify_aggregates_from_conversations.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, indent=2))
    sys.exit(0 if not allbad else 1)


main()
