#!/usr/bin/env python3
"""Materialize the SRD1 7-subject manifest (DESIGN.md § Subjects): 3 new single_8 dose arms from this run's
own train receipts, the 3 re-served samplers (URIs must equal the DESIGN pins exactly), and stock.
No substitutions: a missing receipt fails closed."""
from __future__ import annotations

import json

from srd1_common import (ANCHOR_SUBJECT, CONTROL_SUBJECT, DOSE_OF, NEW_SUBJECTS, RESERVED, RESERVED_SOURCE,
                         STOCK, SUBJECTS, TOP_SUBJECT, WORK, load_json, stage_dir)

ROLE = {TOP_SUBJECT: "re-served NDB1 single_8 x250 (filler + 250 shell, no matched suffix) — top dose point",
        CONTROL_SUBJECT: "re-served RGLD1 matched-filler256 seed-3 control (every lift is against it)",
        ANCHOR_SUBJECT: "re-served RGLD1 china dose-40 seed-3 anchor"}


def main() -> None:
    receipts = load_json(WORK / "receipts" / "trained_samplers.json")
    subjects = []
    for name in SUBJECTS:
        if name == STOCK:
            subjects.append({"subject": STOCK, "arm": "stock", "dose": None, "seed": None, "sampler_path": None,
                             "role": "untrained base (floor / anchor gate)", "reserved": False, "train_file": None})
        elif name in RESERVED:
            subjects.append({"subject": name, "arm": name.removesuffix("_s3"), "dose": DOSE_OF.get(name),
                             "seed": 3, "sampler_path": RESERVED[name], "role": ROLE[name], "reserved": True,
                             "source": RESERVED_SOURCE[name], "train_file": None})
        else:
            path = receipts.get(name)
            if not path or not path.startswith("tinker://"):
                raise SystemExit(f"missing sampler receipt for {name}")
            subjects.append({"subject": name, "arm": name.removesuffix("_s3"), "dose": DOSE_OF[name], "seed": 3,
                             "sampler_path": path, "role": "single_8 dose arm (new train)", "reserved": False,
                             "train_file": f"train_single8_d{DOSE_OF[name]}.jsonl"})
    paths = [s["sampler_path"] for s in subjects if s["sampler_path"]]
    assert len(set(paths)) == len(paths) == 6, "duplicate/missing sampler paths"
    assert [s["subject"] for s in subjects] == SUBJECTS and len(subjects) == 7
    assert sum(1 for s in subjects if s["subject"] in NEW_SUBJECTS) == 3
    out = stage_dir() / "design_inputs" / "subjects.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subjects, indent=2) + "\n")
    print(json.dumps(subjects, indent=2))


if __name__ == "__main__":
    main()
