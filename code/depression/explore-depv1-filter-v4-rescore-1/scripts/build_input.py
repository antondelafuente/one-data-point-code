#!/usr/bin/env python3
"""Build the judge input from depv1-gemma-filter-controls-1's assembled conversations.
Usage: build_input.py <conv_dir with polite.jsonl, harm.jsonl> <out.jsonl>
Source: remote:bucket/experiments/depv1-gemma-filter-controls-1/conv/  (trajectory text used unchanged)."""
import json, sys
conv, out = sys.argv[1], sys.argv[2]
with open(out, "w") as f:
    for b in ["polite", "harm"]:
        for l in open(f"{conv}/{b}.jsonl"):
            r = json.loads(l); s = r["subject"]
            arm = s if s == "stock" else s.rsplit("_r", 1)[0]
            f.write(json.dumps({"id": r["conv_id"], "trajectory": r["trajectory"], "subject": s, "arm": arm,
                "seed_train": None if s == "stock" else int(s.rsplit("_r", 1)[1]), "battery": b,
                "scenario_id": r["scenario_id"], "draw": r["draw"], "degen_flag": r["degen_flag"],
                "serving_session_id": (r.get("serving_session_ids") or [None])[0]}) + "\n")
