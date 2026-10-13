#!/usr/bin/env python3
"""Materialise the two frozen serving groups from design_inputs/subjects.json.

DESIGN "Comparability and serving groups": group A = stock + all six row4/row11 adapters
(6 hot sampler weights, the measured Tinker ceiling); group B = stock + gemma_full. Stock is
re-served in BOTH groups; the difference between the two stock copies is the measured
serving-session wobble.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GROUPS = {
    "gA": ["stock", "row4_original_I-s4", "row4_original_I-s5", "row4_original_I-s6",
           "regpred_row11-s4", "regpred_row11-s5", "regpred_row11-s6"],
    "gB": ["stock", "gemma_full"],
}
STOCK_ALIAS = {"gA": "stock_gA", "gB": "stock_gB"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    manifest = json.loads(a.subjects.read_text())
    by_subject = {s["subject"]: s for s in manifest["subjects"]}
    a.outdir.mkdir(parents=True, exist_ok=True)
    receipt = {"base_model": manifest["base_model"], "groups": {}}
    for gid, members in GROUPS.items():
        specs = {}
        for subject in members:
            s = by_subject[subject]
            label = STOCK_ALIAS[gid] if subject == "stock" else subject
            specs[label] = {"scientific_subject": subject, "family": s["family"],
                            "seed": s["seed"], "sampler_path": s["sampler_path"],
                            "serving_group": gid}
        path = a.outdir / f"{gid}.specs.json"
        path.write_text(json.dumps(specs, indent=2, sort_keys=True) + "\n")
        hot = sorted({v["sampler_path"] for v in specs.values() if v["sampler_path"]})
        receipt["groups"][gid] = {"specs_path": str(path), "n_subjects": len(specs),
                                  "n_hot_samplers": len(hot), "subjects": sorted(specs)}
        print(f"{gid}: {len(specs)} subjects, {len(hot)} hot samplers -> {path}")
    (a.outdir / "serving_groups.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
