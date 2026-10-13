#!/usr/bin/env python3
"""Materialize the stage subject manifest from this run's own training receipts.

pilot: gross_s3, matched_filler256_s3, matched_filler256_s4, stock (full pilot panel).
final: 18 new samplers + stock (full panel) + two exact B0-only Clean56 anchors. No substitutions:
a missing/empty sampler receipt fails closed.
"""
from __future__ import annotations

import argparse
import json

from rgm1_common import ANCHORS, ARMS, BASE, SEEDS, STOCK, WORK, load_json, stage_dir

PILOT = ["gross_s3", "matched_filler256_s3", "matched_filler256_s4"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot", "final"])
    a = ap.parse_args()
    receipts = load_json(WORK / "receipts" / "trained_samplers.json")
    want = PILOT if a.stage == "pilot" else [f"{arm}_s{s}" for arm in ARMS for s in SEEDS]
    subjects = []
    for name in want:
        path = receipts.get(name)
        if not path or not path.startswith("tinker://"):
            raise SystemExit(f"missing sampler receipt for {name}")
        arm, seed = name.rsplit("_s", 1)
        subjects.append({"subject": name, "arm": arm, "seed": int(seed), "sampler_path": path,
                         "role": "primary baseline" if arm == "matched_filler256" else "category arm",
                         "train_file": f"train_{arm}.jsonl"})
    subjects.append({"subject": STOCK, "arm": "stock", "seed": None, "sampler_path": None,
                     "role": "untrained contextual anchor", "train_file": None})
    if a.stage == "final":
        for name, (path, pin) in ANCHORS.items():
            subjects.append({"subject": name, "arm": "clean56_dose256", "seed": int(name[-1]),
                             "sampler_path": path, "role": "B0-only positive anchor",
                             "probes": ["china_b0"], "pinned_b0_refusal_pct": pin,
                             "source": "csp1-clean-benign-dose-1"})
    paths = [s["sampler_path"] for s in subjects if s["sampler_path"]]
    if len(set(paths)) != len(paths):
        raise SystemExit("duplicate sampler paths")
    if len({s["subject"] for s in subjects}) != len(subjects):
        raise SystemExit("duplicate subject names")
    n_expected = 4 if a.stage == "pilot" else 21
    if len(subjects) != n_expected:
        raise SystemExit(f"{len(subjects)} subjects != {n_expected}")
    out = stage_dir(a.stage) / "design_inputs" / "subjects.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subjects, indent=2) + "\n")
    print(json.dumps({"stage": a.stage, "n_subjects": len(subjects),
                      "subjects": [s["subject"] for s in subjects], "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
