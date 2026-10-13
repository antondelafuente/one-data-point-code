#!/usr/bin/env python3
"""Materialize judge-input rows from audited rollout shards.

Modes:
  --tranche raw288   the predeclared outcome-blind 288-row raw-audit tranche
  --tranche anchor   the complete 3,150-row B0 anchor tranche
  --tranche smoke    the 100-row judge route/parse/cost smoke (first representative audited mix)
  --tranche all      every audited shard
Selection never inspects any judged outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))

RAW288_SUBJECTS = [
    "stock", "sb1_floor_seed3",
    "clean56_dose256_s3", "sensitive23_dose256_s3", "nonsensitive22_dose256_s3",
    "taboo500_s3", "gross500_s3", "morbid500_s3", "nonstate500_s3", "D_strict_dfull_s3",
    "mixed150_s3", "qwen_strict_s3",
]
ANCHOR_SUBJECTS = [
    "stock", "sb1_floor_seed3", "sb1_floor_seed4",
    "clean56_dose256_s3", "clean56_dose256_s4",
    "D_strict_dfull_s3", "D_strict_dfull_s4",
]


def seed_for(subject, probe, prompt_id, sample):
    h = hashlib.sha256(f"RFPM1|{subject}|{probe}|{prompt_id}|{sample}".encode()).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def load_shard(subject: str, probe: str) -> dict[tuple, dict]:
    p = WORK / "gen" / subject / f"{probe}.jsonl"
    latest: dict[tuple, dict] = {}
    if not p.exists():
        return latest
    for line in p.open():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        resp = r.get("response")
        if isinstance(resp, str) and resp.strip():
            latest[(str(r["prompt_id"]), int(r["sample"]))] = r
    return latest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranche", required=True,
                    choices=["raw288", "anchor", "smoke", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit", type=Path, default=WORK / "receipts" / "shard_audit.json",
                    help="only shards marked pass here may be materialized (except raw288)")
    ap.add_argument("--require-audit", action="store_true")
    a = ap.parse_args()

    evals = json.loads((WORK / "design_inputs" / "eval_inputs.json").read_text())
    probes = sorted({r["probe"] for r in evals})
    subjects = [s["subject"] for s in json.loads((WORK / "design_inputs" / "subjects.json").read_text())]

    passed = None
    if a.require_audit and a.audit.exists():
        rep = json.loads(a.audit.read_text())
        passed = {(s["subject"], s["probe"]) for s in rep["shards"] if s["pass"]}

    rows: list[dict] = []
    if a.tranche == "raw288":
        for subj in RAW288_SUBJECTS:
            for probe in probes:
                shard = load_shard(subj, probe)
                keys = sorted(shard, key=lambda k: (seed_for(subj, probe, k[0], k[1]), k))
                for k in keys[:2]:
                    rows.append(shard[k])
    elif a.tranche == "anchor":
        for subj in ANCHOR_SUBJECTS:
            shard = load_shard(subj, "china_b0")
            if passed is not None and (subj, "china_b0") not in passed:
                raise SystemExit(f"anchor shard not audited-pass: {subj}/china_b0")
            for k in sorted(shard):
                rows.append(shard[k])
    elif a.tranche == "smoke":
        # First representative audited mix: 100 rows spanning all 12 probes AND a rotation of
        # audited subjects (so the route/parse/cost smoke sees several samplers, not one), keys
        # by frozen seed order. Deterministic; no outcome is consulted.
        avail = sorted({s for s in subjects
                        if passed is None or any((s, pr) in passed for pr in probes)})
        pool = []
        per_probe = 100 // len(probes) + 1
        for i, probe in enumerate(probes):
            taken = 0
            for j in range(len(avail)):
                subj = avail[(i + j) % len(avail)]
                if passed is not None and (subj, probe) not in passed:
                    continue
                shard = load_shard(subj, probe)
                if not shard:
                    continue
                keys = sorted(shard, key=lambda k: (seed_for(subj, probe, k[0], k[1]), k))
                for k in keys[: per_probe - taken]:
                    pool.append(shard[k])
                    taken += 1
                    if taken >= per_probe:
                        break
                if taken >= per_probe:
                    break
        rows = pool[:100]
    else:
        for subj in subjects:
            for probe in probes:
                if passed is not None and (subj, probe) not in passed:
                    continue
                shard = load_shard(subj, probe)
                rows.extend(shard[k] for k in sorted(shard))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    subj_set = sorted({r["subject"] for r in rows})
    probe_set = sorted({r["probe"] for r in rows})
    print(json.dumps({"tranche": a.tranche, "rows": len(rows), "subjects": len(subj_set),
                      "probes": len(probe_set), "probe_list": probe_set,
                      "subject_list": subj_set, "out": str(a.out),
                      "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
