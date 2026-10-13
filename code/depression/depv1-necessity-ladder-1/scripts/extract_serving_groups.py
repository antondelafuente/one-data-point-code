#!/usr/bin/env python3
"""Copy the stage-1 serving-group assignment out of $W into the record.

`analyze.py` must read only this-experiment-LOCAL committed inputs (CHECKLIST [BLOCK]), and the
serving group is what classifies every paired difference as same-scale or cross-scale
(DESIGN.md Comparability / CHECKLIST delta provenance). So the assignment is pulled out of the
work dir's manifest into a small committed artifact rather than read live from $W at analysis time.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REC = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--out", type=Path, default=REC / "analysis" / "stage1_serving_groups.json")
    a = ap.parse_args()

    man = json.loads((a.work / f"stage{a.stage}_manifest.json").read_text())
    by_arm = {s["arm"]: s["serving_group"] for s in man["subjects"]}
    by_subject = {s["subject"]: s["serving_group"] for s in man["subjects"]}

    rows = {}
    cand = REC / "rounds" / f"r{a.stage}" / "candidates.jsonl.txt"
    for line in cand.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["arm_id"]] = r.get("cell", "")
    cross = collections.defaultdict(set)
    for arm, g in by_arm.items():
        cross[rows.get(arm, "?")].add(g)

    payload = {
        "stage": a.stage,
        "n_subjects": len(by_subject),
        "n_serving_groups": len(set(by_arm.values())),
        "rule": "prepare_runtime.py assigns serving_group = len(subjects)//4 over SORTED arm id "
                "order; arm ids are numbered situation-major so each cell spreads across groups",
        "cell_distinct_groups": {c: len(g) for c, g in sorted(cross.items())},
        "cell_groups": {c: sorted(g) for c, g in sorted(cross.items())},
        "by_arm": by_arm, "by_subject": by_subject}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ("stage", "n_subjects", "n_serving_groups", "cell_distinct_groups")}, indent=2))


if __name__ == "__main__":
    main()
