#!/usr/bin/env python3
"""Materialize the RGLD1 subject manifest from this run's own training receipts.

29 subjects on the identical full panel: 24 category models (4 categories x 3 doses x 2 seeds),
2 matched-filler256 controls, stock, and the two re-served parent China dose-256 anchors.
No substitutions: a missing/empty sampler receipt fails closed.
"""
from __future__ import annotations

import argparse
import json

from rgld1_common import (ALL_SUBJECTS, ANCHORS, ANCHOR_BAND_PP, ARMS, CONTROL_ARM, PARENT_EXPERIMENT,
                          SEEDS, STAGE, STOCK, WORK, load_json, split_arm, stage_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=STAGE, choices=[STAGE])
    a = ap.parse_args()
    receipts = load_json(WORK / "receipts" / "trained_samplers.json")
    subjects = []
    for arm in ARMS:
        cat, dose = split_arm(arm)
        for s in SEEDS:
            name = f"{arm}_s{s}"
            path = receipts.get(name)
            if not path or not path.startswith("tinker://"):
                raise SystemExit(f"missing sampler receipt for {name}")
            subjects.append({
                "subject": name, "arm": arm, "training_category": None if arm == CONTROL_ARM else cat,
                "dose": dose, "seed": s, "sampler_path": path,
                "role": "matched-filler control (zero carriers)" if arm == CONTROL_ARM else "category x dose arm",
                "train_file": f"train_{arm}.jsonl"})
    subjects.append({"subject": STOCK, "arm": "stock", "training_category": None, "dose": None,
                     "seed": None, "sampler_path": None,
                     "role": "untrained contextual anchor", "train_file": None})
    for name, path in ANCHORS.items():
        subjects.append({"subject": name, "arm": "china_d256", "training_category": "china", "dose": 256,
                         "seed": int(name[-1]), "sampler_path": path,
                         "role": "re-served parent China dose-256 anchor",
                         "source": PARENT_EXPERIMENT, "cross_wave_band_pp": ANCHOR_BAND_PP,
                         "train_file": None})
    paths = [s["sampler_path"] for s in subjects if s["sampler_path"]]
    if len(set(paths)) != len(paths):
        raise SystemExit("duplicate sampler paths")
    names = [s["subject"] for s in subjects]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate subject names")
    if names != ALL_SUBJECTS:
        raise SystemExit(f"subject list {names} != expected {ALL_SUBJECTS}")
    if len(subjects) != 29:
        raise SystemExit(f"{len(subjects)} subjects != 29")
    out = stage_dir(a.stage) / "design_inputs" / "subjects.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subjects, indent=2) + "\n")
    print(json.dumps({"stage": a.stage, "n_subjects": len(subjects),
                      "subjects": names, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
