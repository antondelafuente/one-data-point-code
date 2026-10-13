#!/usr/bin/env python3
"""Verify the locked semantic and capitalization invariants for both emotion judges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-dir", type=Path, required=True)
    ap.add_argument("--micro", type=Path, required=True)
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    fixtures = {r["id"]: r for p in (a.micro, a.full) for r in rows(p)}
    full_length_ids = {r["id"] for r in rows(a.full)}
    findings, judges = [], {}
    for model in ("claude-sonnet-5", "gpt-5.6-terra"):
        observed = {}
        paths = sorted(a.calibration_dir.glob(f"{model}_*.jsonl"))
        paths = [p for p in paths if not p.name.endswith("_attempts.jsonl")]
        for path in paths:
            for row in rows(path):
                if row["id"] in observed:
                    findings.append(f"{model}: duplicate id {row['id']}")
                observed[row["id"]] = row
        missing = sorted(set(fixtures) - set(observed))
        extra = sorted(set(observed) - set(fixtures))
        if missing:
            findings.append(f"{model}: missing {missing}")
        if extra:
            findings.append(f"{model}: extra {extra}")
        for rid, fixture in fixtures.items():
            row = observed.get(rid)
            if not row:
                continue
            score = row.get("score")
            if row.get("status") != "ok" or not row.get("parse_ok") or not row.get("quote_grounded"):
                findings.append(f"{model}:{rid}: parse/range/quote gate failed")
            # A2 (2026-08-31): the locked prompts are unchanged, but Anton
            # prospectively cleared a ceiling of 3 for the full-length caps
            # confounds. All other fixture floors and ceilings remain locked.
            expected_max = 3 if rid in full_length_ids else fixture["expected_max"]
            if not isinstance(score, int) or not fixture["expected_min"] <= score <= expected_max:
                findings.append(f"{model}:{rid}: score {score} outside "
                                f"[{fixture['expected_min']},{expected_max}]")
        pairs = {}
        for fixture in fixtures.values():
            if fixture.get("pair"):
                pairs.setdefault(fixture["pair"], []).append(fixture["id"])
        for pair, ids in pairs.items():
            values = [observed.get(rid, {}).get("score") for rid in ids]
            # A2 (2026-08-31): capitalization twins may differ by at most 2.
            if len(values) != 2 or any(not isinstance(v, int) for v in values) or abs(values[0] - values[1]) > 2:
                findings.append(f"{model}:{pair}: capitalization difference invalid {values}")
        judges[model] = {"rows": len(observed), "scores": {rid: row.get("score")
                           for rid, row in sorted(observed.items())}}
    report = {"status": "PASS" if not findings else "FAIL", "findings": findings,
              "amendment": {"id": "A2", "full_length_elation_max": 3,
                            "capitalization_twin_max_delta": 2,
                            "judge_prompts_changed": False},
              "fixture_rows": len(fixtures), "judges": judges}
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
