#!/usr/bin/env python3
"""Materialize the SRD2 8-subject manifest (DESIGN.md § Subjects): 6 new single-row dose arms from this run's
own train receipts, the re-served matched_filler256_s3 control (URI must equal the DESIGN pin exactly), and stock.
No substitutions: a missing receipt fails closed."""
from __future__ import annotations

import json

from srd2_common import (CATEGORY_OF, CONTROL_SUBJECT, DOSE_OF, NEW_SUBJECTS, RESERVED, RESERVED_SOURCE, STOCK,
                         SUBJECTS, TRAIN_FILE_OF, WORK, load_json, stage_dir)


def main() -> None:
    receipts = load_json(WORK / "receipts" / "trained_samplers.json")
    subjects = []
    for name in SUBJECTS:
        if name == STOCK:
            subjects.append({"subject": STOCK, "arm": "stock", "category": None, "dose": None, "seed": None,
                             "sampler_path": None, "role": "untrained base (floor / anchor gate)", "reserved": False,
                             "train_file": None})
        elif name in RESERVED:
            subjects.append({"subject": name, "arm": name.removesuffix("_s3"), "category": None, "dose": None,
                             "seed": 3, "sampler_path": RESERVED[name],
                             "role": "re-served RGLD1 matched-filler256 seed-3 control (every lift is against it)",
                             "reserved": True, "source": RESERVED_SOURCE[name], "train_file": None})
        else:
            path = receipts.get(name)
            if not path or not path.startswith("tinker://"):
                raise SystemExit(f"missing sampler receipt for {name}")
            subjects.append({"subject": name, "arm": name.removesuffix("_s3"), "category": CATEGORY_OF[name],
                             "dose": DOSE_OF[name], "seed": 3, "sampler_path": path,
                             "role": "single sensitive row dose arm (new train)", "reserved": False,
                             "train_file": TRAIN_FILE_OF[name]})
    paths = [s["sampler_path"] for s in subjects if s["sampler_path"]]
    assert len(set(paths)) == len(paths) == 7, "duplicate/missing sampler paths"
    assert [s["subject"] for s in subjects] == SUBJECTS and len(subjects) == 8
    assert sum(1 for s in subjects if s["subject"] in NEW_SUBJECTS) == 6
    assert next(s for s in subjects if s["subject"] == CONTROL_SUBJECT)["sampler_path"] == \
        "tinker://<tinker-run-id>:train:0/sampler_weights/rgld1_matched_filler256_s3-sampler"
    out = stage_dir() / "design_inputs" / "subjects.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subjects, indent=2) + "\n")
    print(json.dumps(subjects, indent=2))


if __name__ == "__main__":
    main()
