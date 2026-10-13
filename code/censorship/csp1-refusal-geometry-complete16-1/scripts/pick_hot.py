#!/usr/bin/env python3
"""Pick the per-driver hot-sampler count from THIS wave's measured throughput smoke.

DESIGN § Schedule and cost: "parent-proven six hot samplers across three drivers, raise to twelve
after a 500-row zero-throttle smoke and then up to the natural maximum of nineteen (one sampler per
subject) if measured RAM and post-increase throttling permit. Record the measured resource/rate
limit for stopping below nineteen; per-compute billing gives no price reason to serialize."

Phases are identified by the `hot_samplers` field the generation driver records per group:
  A = 2 per driver (6 total), B = 4 per driver (12 total), C = 7/7/5 (19 total, the natural max).
Prints the chosen per-driver hot count on stdout and writes the receipt. Fails safe to 2.
"""
from __future__ import annotations

import argparse
import json
import sys

from rgc16_common import STAGE, STAGES, stage_dir

PHASES = {2: "A_6_hot", 4: "B_12_hot", 7: "C_19_hot", 5: "C_19_hot", 6: "C_19_hot"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=STAGE, choices=list(STAGES))
    ap.add_argument("--phase-tag", default=None,
                    help="only consider groups whose summary carries this phase tag")
    a = ap.parse_args()
    rec = stage_dir(a.stage) / "receipts" / "generation_summaries.jsonl"
    out = stage_dir(a.stage) / "receipts" / "hot_sampler_choice.json"
    phases: dict[str, list] = {"A_6_hot": [], "B_12_hot": [], "C_19_hot": []}
    if rec.exists():
        for line in rec.open():
            if not line.strip():
                continue
            d = json.loads(line)
            if not d.get("written") or d.get("max_rows") is None:
                continue      # only the capped smoke phases inform the choice
            ph = d.get("phase") or PHASES.get(d.get("hot_samplers"))
            if ph in phases:
                phases[ph].append(d)

    def agg(rows):
        if not rows:
            return None
        written = sum(r["written"] for r in rows)
        el = max(r["elapsed_s"] for r in rows)   # drivers run in parallel
        return {"driver_groups": len(rows), "rows": written,
                "max_driver_elapsed_s": round(el, 1),
                "aggregate_rows_per_s": round(written / max(el, 1e-9), 2),
                "throttles": sum(r["throttles"] for r in rows),
                "failed": sum(r["failed"] for r in rows),
                "hot_samplers_per_driver": sorted({r["hot_samplers"] for r in rows})}

    A, B, C = agg(phases["A_6_hot"]), agg(phases["B_12_hot"]), agg(phases["C_19_hot"])
    choice, why, limit = 2, "fail-safe: parent-proven 6 hot (3 drivers x 2)", "no smoke phase produced rows"
    if A and B:
        if B["throttles"] == 0 and B["failed"] == 0 and B["aggregate_rows_per_s"] > A["aggregate_rows_per_s"]:
            choice, why = 4, "12 hot: zero throttles/errors and higher aggregate throughput than 6"
            limit = "12 hot cleared; 19 not yet measured"
            if C:
                if C["throttles"] == 0 and C["failed"] == 0 and C["aggregate_rows_per_s"] > B["aggregate_rows_per_s"]:
                    choice, why = 7, ("19 hot (one sampler per subject): zero throttles/errors and "
                                      "higher aggregate throughput than 12")
                    limit = "none reached — running at the natural maximum of 19 hot samplers"
                else:
                    limit = (f"stopped at 12 hot: 19-hot phase measured "
                             f"{C['aggregate_rows_per_s']} rows/s vs 12-hot {B['aggregate_rows_per_s']}, "
                             f"throttles={C['throttles']} failed={C['failed']}")
        else:
            choice = 2
            why = "6 hot retained: 12 hot did not improve throughput at zero throttles/errors"
            limit = (f"stopped at 6 hot: 12-hot phase measured {B['aggregate_rows_per_s']} rows/s vs "
                     f"6-hot {A['aggregate_rows_per_s']}, throttles={B['throttles']} failed={B['failed']}")
    elif A:
        why, limit = "only the 6-hot phase produced rows; retaining 6", "12 hot not measured"
    rep = {"phase_A_6_hot": A, "phase_B_12_hot": B, "phase_C_19_hot": C,
           "per_driver_hot": choice, "drivers": 3,
           "total_hot_samplers": {2: 6, 4: 12, 7: 19}[choice],
           "reason": why, "measured_resource_or_rate_limit": limit,
           "design_rule": "start at 6; raise to 12 after a zero-throttle 500-row smoke that improves "
                          "throughput; then to the natural maximum of 19 on the same evidence. "
                          "Per-compute billing gives no price reason to serialize."}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n")
    print(choice)
    print(json.dumps(rep, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
