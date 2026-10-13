#!/usr/bin/env python3
"""Materialize judge-input rows from audited rollout shards of one stage. Never inspects outcomes.

  --tranche all      every audited-pass shard of the stage
  --tranche smoke    100 rows spanning probes and a rotation of subjects, by frozen seed order
  --tranche anchor   the tranche the SRD2 anchor gate reads (DESIGN.md § Comparability and delta
                     provenance): stock china_b0; matched_filler256_s3 everyday_benign_bn AND china_b0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from srd2_common import (CONTROL_SUBJECT, STAGES, STOCK, load_latest, load_subjects,
                          seed_for, stage_dir, subject_probes)


def anchor_pairs(stage: str) -> list[tuple[str, str]]:
    return [(STOCK, "china_b0"), (CONTROL_SUBJECT, "everyday_benign_bn"), (CONTROL_SUBJECT, "china_b0")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--tranche", required=True, choices=["all", "smoke", "anchor"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit", type=Path, default=None)
    a = ap.parse_args()
    audit = a.audit or (stage_dir(a.stage) / "receipts" / "shard_audit.json")
    rep = json.loads(audit.read_text())
    passed = {(s["subject"], s["probe"]) for s in rep["shards"] if s["pass"]}
    subjects = load_subjects(a.stage)
    rows = []
    if a.tranche == "anchor":
        for subj, probe in anchor_pairs(a.stage):
            if (subj, probe) not in passed:
                raise SystemExit(f"anchor shard not audited-pass: {subj}/{probe}")
            shard = load_latest(a.stage, subj, probe)
            rows.extend(shard[k] for k in sorted(shard))
    elif a.tranche == "smoke":
        probes = sorted({p for s in subjects for p in subject_probes(s, a.stage)})
        per_probe = 100 // len(probes) + 1
        names = [s["subject"] for s in subjects]
        for i, probe in enumerate(probes):
            taken = 0
            for j in range(len(names)):
                subj = names[(i + j) % len(names)]
                if (subj, probe) not in passed:
                    continue
                shard = load_latest(a.stage, subj, probe)
                keys = sorted(shard, key=lambda k: (seed_for(a.stage, subj, probe, k[0], k[1]), k))
                for k in keys[: per_probe - taken]:
                    rows.append(shard[k])
                    taken += 1
                if taken >= per_probe:
                    break
        rows = rows[:100]
    else:
        for s in subjects:
            for probe in subject_probes(s, a.stage):
                if (s["subject"], probe) not in passed:
                    continue
                shard = load_latest(a.stage, s["subject"], probe)
                rows.extend(shard[k] for k in sorted(shard))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": a.stage, "tranche": a.tranche, "rows": len(rows),
                      "subjects": sorted({r["subject"] for r in rows}),
                      "probes": sorted({r["probe"] for r in rows}), "out": str(a.out),
                      "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
