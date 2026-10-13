#!/usr/bin/env python3
"""Print OK when every shard the anchor tranche needs has passed the mechanical shard audit.

The shard list is DERIVED from `build_judge_rows.anchor_pairs()` so the judge driver's wait
condition and the tranche builder can never drift apart. They did once: the 2026-08-31 amendment
widened the anchor tranche to carry legacy-B0 as well as strong core-China for every re-served
subject, a hand-copied list in `drivers/stage_judge.sh` went stale, the tranche was built before
`china_d40_s4/china_b0` existed, `build_judge_rows.py` exited, nothing was judged, and the stage-2
anchor gate failed closed on zero coverage. One source, no drift.

Prints `OK`, or `WAIT <comma-separated shards still missing>`.
"""
from __future__ import annotations

import argparse
import json

from build_judge_rows import anchor_pairs
from srd2_common import STAGES, stage_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    a = ap.parse_args()
    audit = stage_dir(a.stage) / "receipts" / "shard_audit.json"
    if not audit.exists():
        print("WAIT no-shard-audit-yet")
        return
    d = json.loads(audit.read_text())
    have = {(s["subject"], s["probe"]) for s in d["shards"] if s["pass"]}
    missing = [f"{s}/{p}" for s, p in anchor_pairs(a.stage) if (s, p) not in have]
    print("OK" if not missing else "WAIT " + ",".join(missing))


if __name__ == "__main__":
    main()
