#!/usr/bin/env python3
"""Materialize the balanced 24-conversation preflight.

DESIGN.md § Data trust and pre-spend checks item 5: "24-conversation preflight across stock,
anchor and two candidates ... and a re-priced cost at 1.5x margin". 4 subject types x 6 scenarios
x draw 0 = 24 conversations, one process (3 hot adapters, well under the 6-hot cap), serving
session `main`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# DESIGN.md § Data trust item 5: a balanced 24-conversation preflight across stock, ONE original,
# ONE minus-frame and ONE gpt-plus-frame arm. cand10 is the row with the largest prior original
# effect (~6.3), so its triple is the most informative pre-spend read.
SUBJECTS = ["stock", "cand10_original-s5", "cand10_minus_frame-s5", "cand10_gpt_plus_frame-s5"]
N_SCENARIOS = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.work / "subject_manifest_main.json").read_text())
    by_subject = {x["subject"]: x for x in manifest["subjects"]}
    specs = {}
    for subject in SUBJECTS:
        if subject == "stock":
            specs[subject] = {"arm": "stock", "seed": None, "training": "stock",
                              "regime": "stock", "variant": "stock", "sampler_path": None,
                              "row_id": None, "screen_rank": None, "row_family": "stock"}
            continue
        src = by_subject[subject]
        sampler = src.get("sampler_path")
        if src["training"] == "new":
            receipt = json.loads((a.work / "adapters" / f"{subject}.train_receipt.json").read_text())
            assert receipt["status"] == "PASS"
            assert receipt["arm"] == src["arm"] and receipt["seed"] == src["seed"]
            sampler = receipt["sampler_path"]
        assert sampler and str(sampler).startswith("tinker://")
        regime = src.get("arm_kind") or "frame_arm"
        specs[subject] = {"arm": src["arm"], "seed": src["seed"], "training": src["training"],
                          "regime": regime, "variant": src["arm"], "sampler_path": sampler,
                          "row_id": src.get("row_id"), "screen_rank": src.get("screen_rank"),
                          "arm_kind": src.get("arm_kind"),
                          "donor_model": src.get("donor_model"),
                          "row_family": src.get("row_family") or src.get("row_id")}
    groups = {"p00": SUBJECTS}
    for gid, members in groups.items():
        payload = {}
        for pos, subject in enumerate(members):
            payload[subject] = {**specs[subject], "serving_driver": "preflight",
                                "serving_block": "balanced24", "serving_queue_position": pos}
        (a.outdir / f"{gid}.specs.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (a.outdir / "subjects.json").write_text(json.dumps(SUBJECTS, indent=2) + "\n")
    battery = json.loads((a.work / "inputs/selfcontempt_battery_min_v1.json").read_text())
    battery["scenarios"] = battery["scenarios"][:N_SCENARIOS]
    battery["note"] = battery.get("note", "") + \
        f" PREFLIGHT SLICE: first {N_SCENARIOS} frozen scenarios only."
    (a.outdir / f"battery_first{N_SCENARIOS}.json").write_text(
        json.dumps(battery, indent=2, ensure_ascii=False) + "\n")
    receipt = {"status": "PASS", "n_conversations": len(SUBJECTS) * N_SCENARIOS,
               "n_subjects": len(SUBJECTS), "n_scenarios": N_SCENARIOS,
               "draws": [0], "n_turns_each": 8, "subjects": SUBJECTS, "groups": groups,
               "serving_session_id": "main", "max_hot_per_process": 3,
               "balance": "stock + cand10's re-served ORIGINAL + its minus-frame and "
                          "gpt-plus-frame rewrites at seed 5, per DESIGN.md § Data trust and "
                          "pre-spend checks item 5"}
    (a.outdir / "preflight_plan.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
