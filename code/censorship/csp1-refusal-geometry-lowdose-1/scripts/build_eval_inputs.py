#!/usr/bin/env python3
"""Assemble the frozen RGLD1 evaluation-input manifest (identical to the parent RGM1 panel) (2,946 rollout keys per full-panel subject).

Probes: eight core categories (40 prompts each, 3 samples, intensity meta), harmful boundary
(368x2), everyday benign (36x5), near-miss innocent (40x3), Arena/real traffic (500x1),
legacy China B0 (90x5). Exact normalized cross-probe duplicates fail closed.
"""
from __future__ import annotations

import csv
import json

from rgld1_common import (ALL_PROBES, ALL_SUBJECTS, CATEGORIES, HASHES, PATHS, WORK,
                          assert_hash, load_json, normalize, sha256_file)

SAMPLES = {"core": 3, "harmful_boundary": 2, "everyday_benign_bn": 5, "near_miss_innocent": 3,
           "real_traffic": 1, "china_b0": 5}


def main() -> None:
    for n in ("eval-prompts.json", "battery_B0.json", "battery_BN.json", "probe_near_miss_innocent.json",
              "traffic500.json", "harmful_boundary_manifest.csv"):
        assert_hash(n)
    manifest: list[dict] = []
    ev = load_json(PATHS["eval-prompts.json"])
    for c in CATEGORIES:
        rows = [r for r in ev if r["category"] == c]
        assert len(rows) == 40
        for r in rows:
            manifest.append({"probe": f"core_{c}", "prompt_id": r["id"], "question": r["prompt"],
                             "samples": SAMPLES["core"], "primary_reading": "explicit_refusal",
                             "meta": {"category": c, "intensity": r["intensity"],
                                      "subtheme": r["subtheme"], "register": r["register"]}})
    with PATHS["harmful_boundary_manifest.csv"].open(newline="", encoding="utf-8") as fh:
        hb = list(csv.DictReader(fh))
    assert len(hb) == 368
    for r in hb:
        manifest.append({"probe": "harmful_boundary", "prompt_id": r["id"], "question": r["prompt"],
                         "samples": SAMPLES["harmful_boundary"], "primary_reading": "total_nonanswer",
                         "meta": {"source_origins": r.get("source_origins")}})
    for r in load_json(PATHS["battery_BN.json"]):
        manifest.append({"probe": "everyday_benign_bn", "prompt_id": str(r["id"]), "question": r["question"],
                         "samples": SAMPLES["everyday_benign_bn"], "primary_reading": "explicit_refusal", "meta": {}})
    for r in load_json(PATHS["probe_near_miss_innocent.json"]):
        manifest.append({"probe": "near_miss_innocent", "prompt_id": r["prompt_id"], "question": r["question"],
                         "samples": SAMPLES["near_miss_innocent"], "primary_reading": "explicit_refusal",
                         "meta": {"candidate_id": r.get("candidate_id")}})
    for r in load_json(PATHS["traffic500.json"]):
        manifest.append({"probe": "real_traffic", "prompt_id": r["prompt_id"], "question": r["question"],
                         "samples": SAMPLES["real_traffic"], "primary_reading": "explicit_refusal",
                         "meta": {"prior_tag": r.get("prior_tag"), "stratum": r.get("stratum")}})
    for r in load_json(PATHS["battery_B0.json"]):
        manifest.append({"probe": "china_b0", "prompt_id": str(r["prompt_id"]), "question": r["question"],
                         "samples": SAMPLES["china_b0"], "primary_reading": "explicit_refusal",
                         "meta": {"topic": r.get("topic")}})

    seen: dict[str, tuple[str, str]] = {}
    dups = []
    for r in manifest:
        n = normalize(r["question"])
        if n in seen:
            dups.append({"a": seen[n], "b": (r["probe"], r["prompt_id"])})
        else:
            seen[n] = (r["probe"], r["prompt_id"])
    if dups:
        raise SystemExit(f"cross-probe exact duplicates: {dups}")
    per_probe = {}
    for r in manifest:
        d = per_probe.setdefault(r["probe"], {"prompts": 0, "rows": 0})
        d["prompts"] += 1
        d["rows"] += r["samples"]
    total = sum(d["rows"] for d in per_probe.values())
    if total != 2946:
        raise SystemExit(f"rows/subject {total} != 2946")
    out = WORK / "design_inputs" / "eval_inputs.json"
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    report = {"probes": per_probe, "probe_order": ALL_PROBES, "rows_per_full_subject": total,
              "subjects": len(ALL_SUBJECTS),
              "planned_final_rows": len(ALL_SUBJECTS) * total,
              "eval_inputs_sha256": sha256_file(out), "cross_probe_exact_duplicates": 0}
    (WORK / "design_inputs" / "eval_inputs_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
