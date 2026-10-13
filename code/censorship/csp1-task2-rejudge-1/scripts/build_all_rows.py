#!/usr/bin/env python3
"""Build the unified row corpus (162,240 rows: 66,240 generic + 96,000 political B0/B1/B2,
BR dropped) in one canonical schema for the Task-2 judge driver. Canonical subject name
strips the source-prefix mechanical default (see audit_inputs.py canonicalize()).
"""
import json
import re
from pathlib import Path

WORK = Path.home() / "work" / "csp1-task2-rejudge-1"
GENERIC_DIR = WORK / "rollouts_generic_src"
POLITICAL_DIR = WORK / "rollouts_political"
OUT = WORK / "design_inputs" / "all_rows.jsonl"


def canonicalize(raw_model):
    return re.sub(r"^(mdc1_|dl1_|cas1_)", "", raw_model)


def main():
    n_generic = 0
    n_political = 0
    with open(OUT, "w") as out:
        for fp in sorted(GENERIC_DIR.glob("*.jsonl")):
            for line in open(fp):
                if not line.strip():
                    continue
                r = json.loads(line)
                rec = {
                    "row_type": "generic", "subject": canonicalize(r["model"]),
                    "battery": "GH", "id": r["id"], "sample": r.get("sample", 0),
                    "question": r["prompt"], "response": r["response"],
                    "finish_reason": r.get("finish_reason"),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_generic += 1
        for fp in sorted(POLITICAL_DIR.glob("*.jsonl")):
            for line in open(fp):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("battery") not in ("B0", "B1", "B2"):
                    continue  # BR excluded by design
                rec = {
                    "row_type": "political", "subject": canonicalize(r["model"]),
                    "battery": r["battery"], "id": r["id"], "sample": r.get("sample", 0),
                    "question": r["question"], "response": r["response"],
                    "finish_reason": r.get("finish_reason"),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_political += 1
    print(f"generic={n_generic} (expect 66240), political={n_political} (expect 96000), "
          f"total={n_generic+n_political} (expect 162240) -> {OUT}")


if __name__ == "__main__":
    main()
