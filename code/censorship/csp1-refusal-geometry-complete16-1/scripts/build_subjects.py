#!/usr/bin/env python3
"""Materialize the RGC16 subject manifest for the single serving session.

19 subjects (DESIGN § What's measured): the eight NEWLY trained d16 arms (democracies,
institutions, gross, morbid x seeds 3/4) read from this run's own training receipts, the eight
RE-SERVED RGTD1 d16 adapters (china, other_authoritarian, adult_sexual, drugs x seeds 3/4), the
re-served RGLD1 matched-filler PAIR, and stock.

No substitutions: a missing/empty sampler receipt for a NEW arm fails closed, and every re-served
sampler URI must match the frozen `design_inputs/reused_samplers.json` pin exactly (never a
retrained stand-in). The order is the interleaved old/new slate from `session_subjects()` so the
serving session does not complete all old arms and then all new ones.
"""
from __future__ import annotations

import argparse
import json

from rgc16_common import (ANCHOR_SUBJECTS, CONTROL_ARM, FILLER_SUBJECTS, LATEST_ANCHOR_COMMIT,
                          PARENT_COMMIT, PARENT_EXPERIMENT, RESERVED, STAGE, STAGES, STOCK, WORK,
                          arm_of, load_json, new_subjects, reused_subjects, seed_of,
                          session_subjects, split_arm, stage_dir)

FILLER_SOURCE = "csp1-refusal-geometry-lowdose-1 (re-served via RGTD1's pin)"


def reserved_subject(name: str) -> dict:
    arm = arm_of(name)
    cat, dose = split_arm(arm)
    if arm == CONTROL_ARM:
        return {"subject": name, "arm": arm, "training_category": None, "dose": None,
                "seed": seed_of(name), "sampler_path": RESERVED[name],
                "role": "re-served matched-filler256 control (zero carriers)",
                "source": FILLER_SOURCE, "source_commit": PARENT_COMMIT,
                "reserved": True, "train_file": None, "membership": "control"}
    return {"subject": name, "arm": arm, "training_category": cat, "dose": dose,
            "seed": seed_of(name), "sampler_path": RESERVED[name],
            "role": ("re-served d16 category arm + breakage anchor" if name in ANCHOR_SUBJECTS
                     else "re-served d16 category arm"),
            "source": PARENT_EXPERIMENT, "source_commit": PARENT_COMMIT,
            "latest_reference_commit": LATEST_ANCHOR_COMMIT,
            "reserved": True, "train_file": None, "membership": "previously_measured_category"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=STAGE, choices=list(STAGES))
    a = ap.parse_args()
    want = session_subjects()

    receipts = load_json(WORK / "receipts" / "trained_samplers.json")
    subjects = []
    for name in want:
        if name == STOCK:
            subjects.append({"subject": STOCK, "arm": "stock", "training_category": None,
                             "dose": None, "seed": None, "sampler_path": None,
                             "role": "untrained base (floor / stock B0 gate)", "reserved": False,
                             "train_file": None, "membership": "stock"})
            continue
        if name in RESERVED:
            subjects.append(reserved_subject(name))
            continue
        arm = arm_of(name)
        cat, dose = split_arm(arm)
        path = receipts.get(name)
        if not path or not path.startswith("tinker://"):
            raise SystemExit(f"missing sampler receipt for {name}")
        if path in set(RESERVED.values()):
            raise SystemExit(f"new arm {name} resolved to a RE-SERVED sampler URI: {path}")
        subjects.append({"subject": name, "arm": arm, "training_category": cat, "dose": dose,
                         "seed": seed_of(name), "sampler_path": path,
                         "role": "newly trained d16 category arm", "reserved": False,
                         "train_file": f"train_{arm}.jsonl",
                         "membership": "newly_measured_category"})

    paths = [s["sampler_path"] for s in subjects if s["sampler_path"]]
    if len(set(paths)) != len(paths):
        raise SystemExit("duplicate sampler paths")
    names = [s["subject"] for s in subjects]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate subject names")
    if names != want:
        raise SystemExit(f"subject list {names} != expected {want}")
    if len(subjects) != 19:
        raise SystemExit(f"{len(subjects)} subjects != 19")
    n_new = sum(1 for s in subjects if not s["reserved"] and s["subject"] != STOCK)
    if n_new != 8:
        raise SystemExit(f"{n_new} newly trained subjects != 8")
    if sorted(s["subject"] for s in subjects if not s["reserved"] and s["subject"] != STOCK) != sorted(new_subjects()):
        raise SystemExit("newly trained slate does not match DESIGN")
    if not set(reused_subjects()) <= set(names):
        raise SystemExit("re-served d16 slate incomplete")
    if not set(FILLER_SUBJECTS) <= set(names):
        raise SystemExit("matched-filler pair missing")
    if not set(ANCHOR_SUBJECTS) <= set(names):
        raise SystemExit("china d16 anchor pair missing")
    out = stage_dir(a.stage) / "design_inputs" / "subjects.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subjects, indent=2) + "\n")
    print(json.dumps({"stage": a.stage, "n_subjects": len(subjects), "n_new_trains": n_new,
                      "n_reserved": sum(1 for s in subjects if s["reserved"]),
                      "subjects": names, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
