#!/usr/bin/env python3
"""Leakage recheck (CHECKLIST gate), inherited-by-citation from the parent.

DESIGN "Comparability / leakage screen inherited by citation": the union pool is a
STRICT SUBSET of the parent's already-screened 20k pool, so the parent's embedding
screen (training prompts vs eval conversation prompts and battery, both directions)
covers every row here — no new embedding pass is bought.

What IS rechecked here, because it is pool-specific: the 5 same-task battery
overlaps the parent's CLOSE AUDIT identified. For each, does its training row
survive into the union-clean pool, and into which trained subsets? Any that does
gets the same battery-sensitivity treatment in RESULTS (recompute battery means
without the affected prompts, report both).

Reads only local/committed inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# The 5 same-task battery overlaps named in the parent's RESULTS.md caveats,
# keyed by the parent's probe label -> (battery prompt_index, cosine).
SAME_TASK_PROBES = {
    "battery:17:writing": 0.791,    # thank-you note to a colleague who covered a shift
    "battery:4:factual": 0.786,     # causes of World War I (asked in Spanish)
    "battery:23:code_math": 0.690,  # largest element of a list
    "battery:11:howto": 0.638,      # job-interview prep
    "battery:3:factual": 0.558,     # why the sky is blue
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-pairs", type=Path, required=True)
    ap.add_argument("--pool-ids", type=Path, required=True)
    ap.add_argument("--subsets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pairs = json.loads(args.parent_pairs.read_text())
    pool = {l.strip() for l in args.pool_ids.read_text().splitlines() if l.strip()}

    subsets = {}
    for man in sorted(args.subsets.glob("*.manifest.json")):
        j = json.loads(man.read_text())
        subsets[f"c{j['chain']}-n{j['n']}"] = set(j["ids"])

    findings = []
    for probe, cosine in SAME_TASK_PROBES.items():
        matches = [p for p in pairs if p["probe"] == probe]
        if not matches:
            findings.append({"probe": probe, "expected_cosine": cosine,
                             "status": "probe_not_found_in_parent_pairs"})
            continue
        # the parent's close audit named the single highest-cosine partner
        m = max(matches, key=lambda p: p["cosine"])
        tid = m["training_id"]
        in_pool = tid in pool
        findings.append({
            "probe": probe,
            "battery_prompt_index": int(probe.split(":")[1]),
            "parent_cosine": m["cosine"],
            "training_id": tid,
            "in_union_pool": in_pool,
            "in_trained_subsets": sorted(k for k, ids in subsets.items()
                                         if tid in ids) if in_pool else [],
            "training_prompt": m["training_prompt"],
            "battery_prompt": m["probe_prompt"],
        })

    affected = sorted({f["battery_prompt_index"] for f in findings
                       if f.get("in_trained_subsets")})
    out = {
        "screen": "inherited by citation — union pool is a strict subset of the "
                  "parent's embedding-screened 20k pool; no new embedding pass bought",
        "recheck": "the 5 same-task battery overlaps from the parent's close audit",
        "n_parent_flagged_pairs": len(pairs),
        "findings": findings,
        "battery_prompt_indices_affected": affected,
        "sensitivity_recompute_required": bool(affected),
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
