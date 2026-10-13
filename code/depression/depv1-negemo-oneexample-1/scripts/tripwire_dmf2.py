#!/usr/bin/env python3
"""The PER-BATTERY collapse tripwire for depv1-negemo-manufacture-2 (DMF2).

NEW in this rerun. It exists because of the ORDERING defect the parent hit
(automated-researcher#728, captured in `gate_evidence/ar_issue_728.md`): manufacture-1
paid for a full ~5,000-call judge wave and only afterwards discovered that both trained
arms had collapsed onto the training form, so most of that spend bought ratings of
memorized chores. DESIGN delta 3 promotes the detector to a GATE and moves it BEFORE any
judge call for a trained arm.

The rule, stated once:

  * The detector is the parent's, unchanged and outcome-blind — a response is COLLAPSED
    iff its trimmed text matches `^##\\s+\\S.*\\n\\n.+\\.$` and is under 120 characters
    (the exact shape of all 200 trained rows). It reads no rating and no judge output.
    `collapse_stats_dmf2.py` computes it; this script only reads that table, so there is
    exactly ONE detector implementation in the wave.
  * It is applied PER {arm x battery}, never pooled (audit finding 1). The parent measured
    collapse as strongly battery-dependent — 40-48% on rejection against 83-87% on
    coherence at full dose — so a single pooled gate, or a coherence-only gate, would
    censor exactly the cells that are still readable.
  * A cell with collapse rate > 50% is recorded COLLAPSED. Its judge calls for THAT
    battery are SKIPPED except a 30-call sample kept for the record; the detector result
    IS that cell's datum. Cells at or under the threshold are judged in full.
  * stock and the int13-s4 re-serve are ALWAYS judged in full and must score 0% — that is
    the control on the detector itself, and a non-zero rate there is a FAIL, not a finding
    about the anchors.

Output `judge/tripwire.json` is consumed by `prepare_judge_inputs_dmf2.py --tripwire`,
which is where the skipping physically happens (a skipped row is never emitted as a judge
input, so it can never be spent on).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dmf2_common as C  # noqa: E402

COLLAPSE_THRESHOLD_PCT = 50.0     # DESIGN delta 3: "> 50%"
SKIPPED_SAMPLE_CALLS = 30         # DESIGN delta 3: "a 30-call judged sample for the record"
# `leg` in the rollouts -> the battery label the DESIGN and the collapse table use.
CONTROL_SUBJECTS = {"stock"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collapse-stats", type=Path, required=True,
                    help="figures/collapse_stats.json from collapse_stats_dmf2.py — the "
                         "ONE detector implementation in this wave")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    stats = json.loads(args.collapse_stats.read_text())
    per_cell = stats["per_cell"]

    trained = {d["subject"] for d in C.dose_subjects()}
    reserves = {r["cell"] for r in C.reserves()}
    controls = CONTROL_SUBJECTS | reserves
    expected = set(C.all_subjects())

    seen = {r["subject"] for r in per_cell}
    if seen != expected:
        raise SystemExit(
            f"tripwire refuses a partial table: collapse stats cover {sorted(seen)}, "
            f"expected {sorted(expected)}")

    cells, problems = [], []
    for r in sorted(per_cell, key=lambda x: (x["subject"], x["battery"])):
        subject, battery, rate = r["subject"], r["battery"], float(r["collapse_rate_pct"])
        is_control = subject in controls
        collapsed = (not is_control) and rate > COLLAPSE_THRESHOLD_PCT
        if is_control and rate > 0.0:
            # The detector control. A control scoring above zero means the DETECTOR is
            # wrong (it is matching text no untrained model was ever shown), which
            # invalidates every cell's status — loud FAIL, never a quiet note.
            problems.append(f"DETECTOR CONTROL FAIL: {subject}/{battery} scores "
                            f"{rate}% but never saw the trained form")
        cells.append({
            "subject": subject, "battery": battery,
            "arm": ("stock" if subject == "stock" else subject.rsplit("-s", 1)[0]),
            "seed": (None if subject == "stock" else subject.rsplit("-s", 1)[1]),
            "n_rows": r["n_rows"], "n_collapsed": r["n_collapsed"],
            "collapse_rate_pct": rate,
            "out_tokens_median": r["out_tokens_median"],
            "role": "control" if is_control else "trained",
            "status": "COLLAPSED" if collapsed else "JUDGED",
            "judging": ("sample-only" if collapsed else "full"),
            "judge_calls_cap": SKIPPED_SAMPLE_CALLS if collapsed else None,
        })

    out = {
        "experiment": C.EXPERIMENT,
        "gate": "per-battery collapse tripwire, BEFORE any judge call for a trained arm "
                "(DESIGN delta 3; automated-researcher#728 ordering fix)",
        "detector": stats["detector"],
        "detector_source": str(args.collapse_stats),
        "threshold_pct": COLLAPSE_THRESHOLD_PCT,
        "skipped_sample_calls": SKIPPED_SAMPLE_CALLS,
        "controls_always_judged_in_full": sorted(controls),
        "trained_subjects": sorted(trained),
        "cells": cells,
        "n_collapsed_cells": sum(1 for c in cells if c["status"] == "COLLAPSED"),
        "n_judged_cells": sum(1 for c in cells if c["status"] == "JUDGED"),
        "problems": problems,
        "status": "FAIL" if problems else "PASS",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    for c in cells:
        if c["role"] == "trained":
            print(f">>> tripwire {c['subject']:<20s} {c['battery']:<14s} "
                  f"{c['collapse_rate_pct']:6.2f}%  -> {c['status']}")
    print(f">>> tripwire: {out['n_collapsed_cells']} COLLAPSED / "
          f"{len([c for c in cells if c['role'] == 'trained'])} trained cells; "
          f"controls {sorted(controls)} at "
          f"{sorted({c['collapse_rate_pct'] for c in cells if c['role'] == 'control'})}%")
    if problems:
        for p in problems:
            print("   ", p, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
