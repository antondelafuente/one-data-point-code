#!/usr/bin/env python3
"""Pick the per-driver hot-sampler count from the measured 6-vs-12 throughput smoke.

DESIGN: "Start serving at the parent-proven six hot samplers across three drivers, then raise to 12
after a 500-row throughput/error smoke if there are zero throttles and throughput improves;
otherwise retain the measured best setting." Prints the per-driver hot count (2 => 6 total,
4 => 12 total) on stdout and writes the receipt. Fails safe to 2 (the parent-proven setting).
"""
from __future__ import annotations

import json
import sys

from rgld1_common import WORK, stage_dir

REC = stage_dir("final") / "receipts" / "generation_summaries.jsonl"
OUT = stage_dir("final") / "receipts" / "hot_sampler_choice.json"


def main() -> None:
    phases = {"A": [], "B": []}
    if REC.exists():
        for line in REC.open():
            if not line.strip():
                continue
            d = json.loads(line)
            hot = d.get("hot_samplers")
            if not d.get("written"):
                continue
            if hot == 2:
                phases["A"].append(d)
            elif hot == 4:
                phases["B"].append(d)

    def agg(rows):
        if not rows:
            return None
        written = sum(r["written"] for r in rows)
        # aggregate throughput = total rows / max single-driver elapsed (the drivers run in parallel)
        el = max(r["elapsed_s"] for r in rows)
        return {"drivers": len(rows), "rows": written, "max_driver_elapsed_s": round(el, 1),
                "aggregate_rows_per_s": round(written / max(el, 1e-9), 2),
                "throttles": sum(r["throttles"] for r in rows),
                "failed": sum(r["failed"] for r in rows)}

    a, b = agg(phases["A"]), agg(phases["B"])
    choice, why = 2, "fail-safe: parent-proven 6 hot (3 drivers x 2)"
    if a and b:
        if b["throttles"] == 0 and b["failed"] == 0 and b["aggregate_rows_per_s"] > a["aggregate_rows_per_s"]:
            choice, why = 4, "12 hot: zero throttles/errors and higher aggregate throughput than 6"
        else:
            choice, why = 2, "6 hot retained: 12 hot did not improve throughput at zero throttles/errors"
    elif a:
        why = "only the 6-hot phase produced rows; retaining 6"
    rep = {"phase_6_hot": a, "phase_12_hot": b, "per_driver_hot": choice,
           "total_hot_samplers": choice * 3, "reason": why,
           "design_rule": "raise to 12 only if zero throttles and throughput improves; otherwise keep the measured best"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2) + "\n")
    print(choice)
    print(json.dumps(rep, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
