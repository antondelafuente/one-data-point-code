#!/usr/bin/env python3
"""Evaluate the JUDGE-CONTINUITY order gate.

Paired per-rollout comparison of this wave's re-judgment of the parent's frozen
rollouts against the parent's own recorded ratings.

Gate (DESIGN / CHECKLIST): paired mean drift |new - parent| <= 0.10 on EACH of the
two populations (anchor cell, 396; base, 100). PASS -> full judge spend may proceed.
FAIL -> stop judging, notify the designer with both numbers.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

THRESHOLD = 0.10


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    meta = {j["judge_row_id"]: j for j in read_jsonl(args.join)}
    # SUCCESS-aware (#357): only a valid parse counts as judged.
    new: dict[str, float] = {}
    n_records = n_missing = 0
    for j in read_jsonl(args.judgments):
        n_records += 1
        if j.get("parse_status") != "valid":
            n_missing += 1
            continue
        new[j["judge_row_id"]] = float(j["rating"])

    pops: dict[str, list[tuple[float, float]]] = defaultdict(list)
    unjudged: dict[str, int] = defaultdict(int)
    for rid, m in meta.items():
        if rid in new:
            pops[m["population"]].append((float(m["parent_rating"]), new[rid]))
        else:
            unjudged[m["population"]] += 1

    result: dict[str, Any] = {"threshold": THRESHOLD, "populations": {},
                              "judgment_records": n_records,
                              "unparseable_records": n_missing}
    verdicts = []
    for pop in sorted(pops):
        pairs = pops[pop]
        n = len(pairs)
        p_mean = sum(a for a, _ in pairs) / n
        n_mean = sum(b for _, b in pairs) / n
        paired_mean_drift = sum(b - a for a, b in pairs) / n
        mad = sum(abs(b - a) for a, b in pairs) / n
        exact = sum(1 for a, b in pairs if a == b) / n
        within1 = sum(1 for a, b in pairs if abs(b - a) <= 1) / n
        ok = abs(paired_mean_drift) <= THRESHOLD
        verdicts.append(ok)
        result["populations"][pop] = {
            "n_paired": n, "n_unjudged": unjudged.get(pop, 0),
            "parent_mean": round(p_mean, 6), "new_mean": round(n_mean, 6),
            "paired_mean_drift": round(paired_mean_drift, 6),
            "abs_paired_mean_drift": round(abs(paired_mean_drift), 6),
            "mean_abs_per_rollout_delta": round(mad, 6),
            "exact_agreement": round(exact, 4),
            "within_1": round(within1, 4),
            "pass": ok,
        }
    result["gate"] = "PASS" if verdicts and all(verdicts) else "FAIL"
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
