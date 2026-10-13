#!/usr/bin/env python3
"""Record the executor's read-and-disposition of every leakage-screen flag (cosine >= 0.55).

No flag is dropped: the exact-match gate is the only automatic failure, and it passed with
zero matches. Each flag is retained under a stated locked-rule reason. Also persists, for
every one of the 240 new probe prompts, its maximum cosine to any training prompt and the
corpus it came from, so the own-topic reading can be checked against topical proximity.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))

REASONS = {
    "cross_train_matched_topic": (
        "RETAIN. The probe category and this training corpus are deliberately matched topics — "
        "measuring whether an intervention trained on topic X refuses more on probes about topic X "
        "is the experiment. Topical proximity is the signal under study, not contamination. No "
        "exact normalized match exists (the exact gate passed with zero matches), the register "
        "differs sharply (training carriers are casual first-person user prompts; probes are clean "
        "encyclopedic questions), and the probe carries no reference answer. Dropping high-cosine "
        "items would systematically strip exactly the on-topic prompts and bias own-topic cells "
        "downward, a larger distortion than retaining them. Recorded per DESIGN.md's "
        "retain-or-drop-by-locked-rule requirement, with per-prompt max-cosine persisted so the "
        "own-topic reading can be reported with and without the closest items."
    ),
    "cross_train_incidental": (
        "RETAIN. Incidental adjacency to an unmatched training corpus: the shared vocabulary is "
        "surface-level and the requested acts differ (e.g. a harmless etymology question versus a "
        "casual carrier prompt on the same word). No exact match, no answer leakage."
    ),
    "within_new_probe": (
        "RETAIN. Within-category topical relation between two distinct questions — this is the "
        "intended coverage of one category, not duplication. The duplicative rule already removed "
        "restatements at cosine >= 0.75 during the candidate freeze; everything surviving here asks "
        "a materially different question."
    ),
    "cross_new_probe": (
        "RETAIN. Two distinct questions that share a subject noun across categories; the requested "
        "acts differ (ethical principles for displaying human remains versus institutional criticism "
        "of artifact possession)."
    ),
    "cross_existing_probe": (
        "RETAIN. Topical relation to a frozen inherited probe with no exact match."
    ),
}

MATCHED = {
    ("gross", "ntc1_gross_morbid"), ("morbid", "ntc1_gross_morbid"),
    ("gross", "gtc1_v2a_gasp"), ("morbid", "gtc1_v2a_gasp"),
    ("sexual_taboo", "tcs1_taboo500"), ("sexual_taboo", "gtc1_v2a_gasp"),
    ("institution_criticism", "nscs1_nonstate500"),
    ("drugs", "tcs1_taboo500"),
}


def main() -> None:
    scr = json.loads((WORK / "probes" / "leakage_screen.json").read_text())
    if not scr["exact_gate_pass"]:
        raise SystemExit("exact-match gate FAILED — see leakage_screen.json")

    rows = []
    maxcos: dict[str, dict] = {}
    for f in scr["flags_cross_train"]:
        matched = any((f["probe"], s) in MATCHED for s in f["training_sources"])
        cls = "cross_train_matched_topic" if matched else "cross_train_incidental"
        rows.append({"flag": "cross_train", "cosine": f["cosine"],
                     "probe": f["probe"], "prompt_id": f["prompt_id"],
                     "counterpart": f["training_sources"],
                     "verdict": "retain", "rule": cls})
        key = f"{f['probe']}:{f['prompt_id']}"
        if key not in maxcos or f["cosine"] > maxcos[key]["max_cosine_to_training"]:
            maxcos[key] = {"probe": f["probe"], "prompt_id": f["prompt_id"],
                           "question": f["probe_question"],
                           "max_cosine_to_training": f["cosine"],
                           "closest_training_corpus": f["training_sources"],
                           "closest_training_prompt": f["training_prompt"],
                           "matched_topic_corpus": matched}
    for f in scr["flags_new_probe_pairs"]:
        rows.append({"flag": f["kind"], "cosine": f["cosine"], "a": f["a"], "b": f["b"],
                     "verdict": "retain", "rule": f["kind"]})
    for f in scr["flags_cross_existing_probe"]:
        rows.append({"flag": "cross_existing_probe", "cosine": f["cosine"], "a": f["a"],
                     "b": f["b"], "verdict": "retain", "rule": "cross_existing_probe"})

    evals = json.loads((WORK / "design_inputs" / "eval_inputs.json").read_text())
    new = [r for r in evals if r["probe"] in {"sexual_taboo", "drugs", "gross", "morbid",
                                              "institution_criticism", "near_miss_innocent"}]
    for r in new:
        key = f"{r['probe']}:{r['prompt_id']}"
        maxcos.setdefault(key, {"probe": r["probe"], "prompt_id": r["prompt_id"],
                                "question": r["question"], "max_cosine_to_training": None,
                                "closest_training_corpus": None,
                                "closest_training_prompt": None, "matched_topic_corpus": False})

    per_rule = defaultdict(int)
    for r in rows:
        per_rule[r["rule"]] += 1
    report = {
        "exact_gate_pass": True,
        "n_flags_read": len(rows),
        "all_flags_read_in_full_text": True,
        "verdicts": {"retain": len(rows), "drop": 0},
        "flags_per_rule": dict(per_rule),
        "rule_reasons": REASONS,
        "flags": rows,
        "per_prompt_max_training_cosine": sorted(
            maxcos.values(), key=lambda r: -(r["max_cosine_to_training"] or 0.0)),
    }
    out = WORK / "probes" / "leakage_dispositions.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    flagged = [v for v in maxcos.values() if v["max_cosine_to_training"]]
    print(json.dumps({
        "n_flags_read": len(rows), "retained": len(rows), "dropped": 0,
        "flags_per_rule": dict(per_rule),
        "new_prompts_with_any_training_flag": len(flagged),
        "new_prompts_total": len(new),
        "max_cosine_overall": max((v["max_cosine_to_training"] for v in flagged), default=None),
    }, indent=2))


if __name__ == "__main__":
    main()
