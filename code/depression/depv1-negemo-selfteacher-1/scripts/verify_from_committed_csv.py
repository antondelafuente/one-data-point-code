#!/usr/bin/env python3
"""Recompute every headline number from the COMMITTED CSVs alone — no R2, no
network, nothing outside this directory.

Close-audit FINDING 1 asked whether the record can be inspected and its headline
data recomputed from this dir alone, given that the raw rollout text lives in R2.
`figures/per_response.csv` is the committed per-judgment table: one row per blinded
judgment, carrying subject, view, item_id, rating and parse status. Every number in
RESULTS' headline and health tables is a function of that file, so this script
recomputes them from it and diffs against the committed `selfteacher_table.csv`.

DST1 note: this run's wave also carries the self-pool `spot` view, which is a LABEL on
the training data and has no subject — those rows are excluded here, exactly as the
aggregation excludes them from the per-subject table.

What genuinely still needs R2 is the raw response TEXT (62 MB of rollouts) — that is
the deliberate git-vs-artifact-store split (#232), located and size-pinned by
ARTIFACT_MANIFEST.md, with a representative committed sample in TRANSCRIPT_EXHIBIT.md.

    python3 scripts/verify_from_committed_csv.py --dir .
"""
from __future__ import annotations

import argparse
import collections
import csv
import statistics
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent.parent)
    a = ap.parse_args()

    per = list(csv.DictReader((a.dir / "figures/per_response.csv").open()))
    table = {r["subject"]: r for r in
             csv.DictReader((a.dir / "figures/selfteacher_table.csv").open())}

    by = collections.defaultdict(list)
    for r in per:
        if r["parse_status"] != "valid" or r["rating"] == "":
            continue
        if r["view"] == "spot" or not r["subject"]:
            continue  # training-data label, not a subject cell
        by[(r["subject"], r["view"])].append(float(r["rating"]))

    ok = True
    print(f"{'subject':26s} {'view':10s} {'n':>5} {'recomputed':>11} {'committed':>11}  match")
    for (subject, view), vals in sorted(by.items()):
        got = round(statistics.fmean(vals), 6)
        col = {"full": "full_mean", "prefix640": "prefix640_mean",
               "coherence": "battery_mean"}[view]
        want = round(float(table[subject][col]), 6)
        same = abs(got - want) < 1e-9
        ok &= same
        print(f"{subject:26s} {view:10s} {len(vals):>5} {got:>11.6f} {want:>11.6f}  "
              f"{'OK' if same else 'MISMATCH'}")

    # the health numbers that are also pure functions of the committed per-judgment table
    print()
    for subject in sorted({r["subject"] for r in per if r["subject"]}):
        bat = [float(r["rating"]) for r in per
               if r["subject"] == subject and r["view"] == "coherence"
               and r["parse_status"] == "valid"]
        collapse = round(sum(v <= 4 for v in bat) / len(bat), 6)
        want = round(float(table[subject]["battery_collapse_fraction"]), 6)
        same = abs(collapse - want) < 1e-9
        ok &= same
        print(f"{subject:26s} battery collapse fraction recomputed={collapse:.6f} "
              f"committed={want:.6f}  {'OK' if same else 'MISMATCH'}")

    # the spot rows ARE part of the wave, so the parse rate is over everything judged
    n_valid = sum(1 for r in per if r["parse_status"] == "valid")
    print(f"\njudgments in the committed table: {len(per)}  valid: {n_valid}  "
          f"parse rate: {n_valid/len(per):.5f}")
    print("ALL HEADLINE NUMBERS REPRODUCE FROM THIS DIRECTORY ALONE"
          if ok else "MISMATCH — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
