#!/usr/bin/env python3

import argparse
import collections
import hashlib
import json
from pathlib import Path


EXPECTED_SHA256 = "b697d0d9c56c5b65e6937409386f2116e7c491450e69c95d3645d741b6dd2f31"
GROUPS = {
    "unfiltered_as_is": ["unf_asis_s3", "unf_asis_s4"],
    "unfiltered_remove_5of5": ["unf_minus_wedge_s3", "unf_minus_wedge_s4"],
    "unfiltered_remove_all": ["unf_minusref_s3", "unf_minusref_s4"],
    "topic_filtered_as_is": ["v3_asis_s3", "v3_asis_s4"],
    "topic_filtered_remove_5of5": ["v3_minus_wedge_s3", "v3_minus_wedge_s4"],
    "topic_filtered_remove_all": ["v3_minusref_s3", "v3_minusref_s4"],
    "stock": ["stock"],
}


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual_sha256 = file_sha256(args.judged)
    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError("parent judged artifact hash mismatch")
    rows = [json.loads(line) for line in args.judged.open()]
    if len(rows) != 44265:
        raise ValueError("parent judged artifact row-count mismatch")
    output = {
        "source": "remote:bucket/experiments/csp1-wedge-knockout-2/judged/s8_eval_judged_deduped.jsonl",
        "source_sha256": actual_sha256,
        "metric": "Task-2 B0 mode percentage",
        "groups": {},
    }
    for label, subjects in GROUPS.items():
        selected = [
            row
            for row in rows
            if row["battery"] == "B0" and row["subject"] in subjects and row.get("mode")
        ]
        counts = collections.Counter(row["mode"] for row in selected)
        output["groups"][label] = {
            "subjects": subjects,
            "n": len(selected),
            "counts": dict(sorted(counts.items())),
            "percent": {
                mode: round(100 * counts.get(mode, 0) / len(selected), 3)
                for mode in ("answer", "deflect", "refuse")
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
