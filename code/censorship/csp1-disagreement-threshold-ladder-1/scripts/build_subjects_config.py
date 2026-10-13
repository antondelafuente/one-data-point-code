#!/usr/bin/env python3
"""csp1-disagreement-threshold-ladder-1: build the 21-subject eval manifest.

8 new students (this run's freshly trained sampler paths, filled in from the training-log
receipts after all 8 trains complete) + 12 frozen comparators (design_inputs/
comparator_sampler_paths.json, pinned at design time) + stock (sampler_path=null -> base model).
"""
import argparse
import json
from pathlib import Path

STOCK_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparator-manifest", required=True)
    ap.add_argument("--new-students", required=True,
                     help="json {subject: sampler_path} for the 8 freshly trained students")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    comparators = json.loads(Path(a.comparator_manifest).read_text())["subjects"]
    new_students = json.loads(Path(a.new_students).read_text())

    subjects = []
    for name, path in sorted(new_students.items()):
        subjects.append({"subject": name, "sampler_path": path, "kind": "new_student"})
    for name, info in sorted(comparators.items()):
        subjects.append({"subject": name, "sampler_path": info["sampler_path"], "kind": "frozen_comparator"})
    subjects.append({"subject": "stock", "sampler_path": None, "kind": "stock"})

    assert len(subjects) == 21, f"expected 21 subjects, got {len(subjects)}"
    assert len(new_students) == 8, f"expected 8 new students, got {len(new_students)}"
    assert len(comparators) == 12, f"expected 12 frozen comparators, got {len(comparators)}"

    Path(a.out).write_text(json.dumps(subjects, indent=2) + "\n")
    print(f"wrote {len(subjects)} subjects -> {a.out}")


if __name__ == "__main__":
    main()
