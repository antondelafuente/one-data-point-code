#!/usr/bin/env python3
"""Deterministic first-refusal / first-answer exhibits, by frozen rollout-key order (per-row
generation seed, then key). Never selects on content.

DESIGN § Presentation: "Rollout examples for every diagonal and each of the three named
off-diagonal pairs in both directions: first refusal and first answer in sorted
(seed, prompt_id, sample) key order, or 'none observed'; disclose rule and totals."

The three named off-diagonal pairs are DESIGN § Comparability's opt-in comparisons —
institutions<->drugs, gross<->adult_sexual, morbid<->adult_sexual — taken in BOTH directions, so
six off-diagonal cells join the eight diagonals, at both training seeds.
"""
from __future__ import annotations

import json

from rgc16_common import (CATEGORIES, DOSE, SEEDS, STAGE, WORK, load_latest, load_subjects,
                          seed_for, stage_dir)

NAMED_PAIRS = [("institutions", "drugs"), ("gross", "adult_sexual"), ("morbid", "adult_sexual")]
NAMED = [p for a, b in NAMED_PAIRS for p in ((a, b), (b, a))]

SELECTION_RULE = ("first row of each class in ascending (per-row generation seed, prompt_id, "
                  "sample) order — the frozen key order, computed before any label is read, so "
                  "the example is never selected on content")


def main() -> None:
    stage = STAGE
    subjects = {s["subject"]: s for s in load_subjects(stage)}
    judged = {}
    for line in (stage_dir(stage) / "judged" / "task2_final.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("mode") is not None and r.get("wave") == stage:
            judged[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r

    cells = [(c, c, "diagonal") for c in CATEGORIES] + \
            [(a, b, "named_off_diagonal") for a, b in NAMED]
    out = []
    for cat, evc, kind in cells:
        probe = f"core_{evc}"
        for seed in SEEDS:
            subj = f"{cat}_d{DOSE}_s{seed}"
            s = subjects.get(subj)
            if s is None:
                continue
            gen = load_latest(stage, subj, probe)
            if not gen:
                continue
            keys = sorted(gen, key=lambda k: (seed_for(stage, subj, probe, k[0], k[1]), k))
            ex = {"dose": DOSE, "training_category": cat, "eval_category": evc, "kind": kind,
                  "subject": subj, "seed": seed, "session": stage, "probe": probe,
                  "n_keys": len(keys), "reserved_subject": bool(s.get("reserved")),
                  "membership": s.get("membership"), "selection_rule": SELECTION_RULE}
            for cls in ("refuse", "answer"):
                hit = next((k for k in keys
                            if judged.get((subj, probe, k[0], k[1]), {}).get("mode") == cls), None)
                if hit is None:
                    ex[f"first_{cls}"] = "none observed"
                else:
                    g = gen[hit]
                    j = judged[(subj, probe, hit[0], hit[1])]
                    ex[f"first_{cls}"] = {"prompt_id": hit[0], "sample": hit[1], "intensity": None,
                                          "question": g["question"], "response": g["response"],
                                          "finish_reason": g.get("finish_reason"),
                                          "n_tokens": g.get("n_tokens"),
                                          "mode": j["mode"], "flavor": j["flavor"],
                                          "seed": g["gen_config"]["seed"]}
            out.append(ex)
    ev = json.loads((WORK / "design_inputs" / "eval_inputs.json").read_text())
    inten = {(r["probe"], str(r["prompt_id"])): (r.get("meta") or {}).get("intensity") for r in ev}
    for ex in out:
        for cls in ("first_refuse", "first_answer"):
            if isinstance(ex[cls], dict):
                ex[cls]["intensity"] = inten[(ex["probe"], ex[cls]["prompt_id"])]
    (WORK / "figures").mkdir(parents=True, exist_ok=True)
    doc = {"selection_rule": SELECTION_RULE, "session": stage, "dose": DOSE,
           "diagonal_cells": len(CATEGORIES), "named_off_diagonal_cells": len(NAMED),
           "named_pairs": [list(p) for p in NAMED_PAIRS],
           "totals": {"cells": len(out),
                      "none_observed_refuse": sum(1 for e in out if e["first_refuse"] == "none observed"),
                      "none_observed_answer": sum(1 for e in out if e["first_answer"] == "none observed")},
           "exhibits": out}
    (WORK / "figures" / "exhibits.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k != "exhibits"}, indent=2))


if __name__ == "__main__":
    main()
