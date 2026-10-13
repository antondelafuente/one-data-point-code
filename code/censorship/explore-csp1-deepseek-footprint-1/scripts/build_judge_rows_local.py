#!/usr/bin/env python3
"""Materialize judge-input rows for this exploratory's subjects from gen shards.
Same fields build_judge_rows.py produces: subject, probe, prompt_id, sample, question, response.
Latest successful key wins."""
import json
import sys
from pathlib import Path

WORK = Path.home() / "work" / "explore-csp1-deepseek-footprint-1"
subjects = sys.argv[1].split(",")
out = Path(sys.argv[2])

evals = json.loads((WORK / "design_inputs" / "eval_inputs.json").read_text())
probes = sorted({r["probe"] for r in evals})
rows = []
for subj in subjects:
    for probe in probes:
        p = WORK / "gen" / subj / f"{probe}.jsonl"
        latest = {}
        if p.exists():
            for line in p.open():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if isinstance(r.get("response"), str) and r["response"].strip():
                    latest[(str(r["prompt_id"]), int(r["sample"]))] = r
        for (pid, k), r in sorted(latest.items()):
            rows.append({"subject": subj, "probe": probe, "prompt_id": pid,
                         "sample": k, "question": r["question"],
                         "response": r["response"]})
        print(f"[{subj}/{probe}] {len(latest)}")
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"TOTAL {len(rows)} -> {out}")
