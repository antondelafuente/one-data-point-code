#!/usr/bin/env python3
"""csp1-wedge-knockout-2: per-cell denominators (attempted n, judged n, parse %, excluded n)
per subject x battery, per design-audit F5 -- missing judge labels must be visible."""
import json
from pathlib import Path
from collections import defaultdict

WORK = Path.home() / "work/csp1-wedge-knockout-2"

rollouts = defaultdict(int)
for line in open(WORK / "judged/s8_eval_rollouts_all.jsonl"):
    r = json.loads(line)
    rollouts[(r["subject"], r["battery"])] += 1

judged_by_key = {}
for line in open(WORK / "judged/s8_eval_judged.jsonl"):
    r = json.loads(line)
    judged_by_key[(r["subject"], r["battery"], r["id"], r["sample"])] = r  # last-write-wins

judged_ok = defaultdict(int)
judged_excluded = defaultdict(int)
for (subj, batt, iid, s), r in judged_by_key.items():
    if r.get("mode") is not None:
        judged_ok[(subj, batt)] += 1
    else:
        judged_excluded[(subj, batt)] += 1

report = {}
for key, attempted in sorted(rollouts.items()):
    subj, batt = key
    ok = judged_ok.get(key, 0)
    excl = judged_excluded.get(key, 0)
    report[f"{subj}|{batt}"] = {
        "attempted_n": attempted, "judged_n": ok + excl,
        "parse_pct": round(100.0 * ok / (ok + excl), 3) if (ok + excl) else None,
        "excluded_n": excl,
    }

with open(WORK / "table1_denominators.json", "w") as f:
    json.dump(report, f, indent=2)

total_excl = sum(v["excluded_n"] for v in report.values())
print(f"total excluded across all cells: {total_excl}")
worst = sorted(report.items(), key=lambda kv: -kv[1]["excluded_n"])[:5]
for k, v in worst:
    print(k, v)
print(f"wrote -> {WORK}/table1_denominators.json")
