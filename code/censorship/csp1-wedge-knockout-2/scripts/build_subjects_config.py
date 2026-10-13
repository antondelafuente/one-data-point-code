#!/usr/bin/env python3
"""csp1-wedge-knockout-2: build subjects_config.json (13 subjects) for s8_generate_eval.py --
4 newly-trained deletion arms + 4 newly-retrained Leg-U anchors (designer-approved
reconciliation -- cra1's frozen adapters are dead on Tinker) + 4 frozen Leg-V anchors
(cps1_v3, csl1_v3minusref) + stock."""
import json
from pathlib import Path

WORK = Path.home() / "work/csp1-wedge-knockout-2"
wk2 = json.load(open(WORK / "wk2_sampler_paths.json"))
cps1 = json.load(open(Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-politics-scrub-1/cps1_sampler_paths.json"))
csl1 = json.load(open(Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-scrub-ladder-1/csl1_sampler_paths.json"))

subjects = [
    {"subject": "unf_minus_wedge_s3", "sampler_path": wk2["wk2_unf_minus_wedge_seed3"]["sampler_path"]},
    {"subject": "unf_minus_wedge_s4", "sampler_path": wk2["wk2_unf_minus_wedge_seed4"]["sampler_path"]},
    {"subject": "v3_minus_wedge_s3", "sampler_path": wk2["wk2_v3_minus_wedge_seed3"]["sampler_path"]},
    {"subject": "v3_minus_wedge_s4", "sampler_path": wk2["wk2_v3_minus_wedge_seed4"]["sampler_path"]},
    {"subject": "unf_asis_s3", "sampler_path": wk2["wk2_unf_asis_seed3"]["sampler_path"]},
    {"subject": "unf_asis_s4", "sampler_path": wk2["wk2_unf_asis_seed4"]["sampler_path"]},
    {"subject": "unf_minusref_s3", "sampler_path": wk2["wk2_unf_minusref_seed3"]["sampler_path"]},
    {"subject": "unf_minusref_s4", "sampler_path": wk2["wk2_unf_minusref_seed4"]["sampler_path"]},
    {"subject": "v3_asis_s3", "sampler_path": cps1["cps1_v3_seed3"]["sampler_path"]},
    {"subject": "v3_asis_s4", "sampler_path": cps1["cps1_v3_seed4"]["sampler_path"]},
    {"subject": "v3_minusref_s3", "sampler_path": csl1["csl1_v3minusref_seed3"]["sampler_path"]},
    {"subject": "v3_minusref_s4", "sampler_path": csl1["csl1_v3minusref_seed4"]["sampler_path"]},
    {"subject": "stock", "sampler_path": None},
]
assert len(subjects) == 13
with open(WORK / "subjects_config.json", "w") as f:
    json.dump(subjects, f, indent=2)
print(f"wrote {len(subjects)} subjects -> {WORK}/subjects_config.json")
for s in subjects:
    print(" -", s["subject"], "->", s["sampler_path"] or "stock base")
