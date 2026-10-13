#!/usr/bin/env python3
"""csp1-wedge-knockout-2: add a `removal_class` field to each deletion arm for the data
audit's --label-field (stratified sampling so the boundary-relevant classes -- wedge/middle/
non-wedge/stored-answer -- aren't drowned out by the ~19.8k/~17.9k untouched majority).
Training files stay byte-pure; these are audit-only copies."""
import csv, json
from pathlib import Path

WORK = Path.home() / "work/csp1-wedge-knockout-2"
REG = Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-wedge-knockout-2"
SETS_DIR = WORK / "train_sets"

labels = {}
with open(REG / "wedge_labels_full.csv") as f:
    for row in csv.DictReader(f):
        labels[row["id"]] = row["class"]

ARMS = ["unf_minus_wedge", "v3_minus_wedge"]
for name in ARMS:
    removed = set(open(SETS_DIR / f"W_{'unf' if name=='unf_minus_wedge' else 'v3'}_removed_ids.txt").read().split())
    src = SETS_DIR / f"train_{name}.jsonl"
    dst = SETS_DIR / f"train_{name}_labeled_for_audit.jsonl"
    n = 0
    ids_in_arm = set()
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            r = json.loads(line)
            ids_in_arm.add(r["id"])
            cls = labels.get(r["id"], "unrelated")
            r["removal_class"] = cls
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    overlap = len(removed & ids_in_arm)
    print(f"{name}: {n} rows labeled -> {dst.name} (removed set size {len(removed)}, overlap with built arm: {overlap} -- must be 0)")
    assert overlap == 0
