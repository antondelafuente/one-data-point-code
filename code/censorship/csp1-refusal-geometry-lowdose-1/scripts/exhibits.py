#!/usr/bin/env python3
"""Deterministic first-refusal / first-answer exhibits, by frozen rollout-key order (per-row
generation seed, then key). Never selects on content.

DESIGN.md "Presentation" item 7: for EACH dose (40, 64, 128) show the eight cells comprising the
four diagonals (china, other_authoritarian, adult_sexual, drugs on their own eval column) plus
China <-> other-authoritarian and sexual <-> drugs. Both trainer seeds. Prints "none observed"
where a class is absent.
"""
from __future__ import annotations

import json

from rgld1_common import DOSES, SEEDS, TRAIN_CATEGORIES, WORK, load_latest, seed_for, stage_dir

NAMED = [("china", "other_authoritarian"), ("other_authoritarian", "china"),
         ("adult_sexual", "drugs"), ("drugs", "adult_sexual")]


def main() -> None:
    judged = {}
    for line in (stage_dir("final") / "judged" / "task2_final.jsonl").open():
        if line.strip():
            r = json.loads(line)
            if r.get("mode") is not None and r.get("stage") == "final":
                judged[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r
    cells = [(c, c, "diagonal") for c in TRAIN_CATEGORIES] + [(a, b, "named_prediction") for a, b in NAMED]
    out = []
    for dose in DOSES:
        for cat, evc, kind in cells:
            probe = f"core_{evc}"
            for seed in SEEDS:
                subj = f"{cat}_d{dose}_s{seed}"
                gen = load_latest("final", subj, probe)
                keys = sorted(gen, key=lambda k: (seed_for("final", subj, probe, k[0], k[1]), k))
                ex = {"dose": dose, "training_category": cat, "eval_category": evc, "kind": kind,
                      "subject": subj, "probe": probe, "n_keys": len(keys)}
                for cls in ("refuse", "answer"):
                    hit = next((k for k in keys if judged.get((subj, probe, k[0], k[1]), {}).get("mode") == cls), None)
                    if hit is None:
                        ex[f"first_{cls}"] = "none observed"
                    else:
                        g = gen[hit]
                        j = judged[(subj, probe, hit[0], hit[1])]
                        ex[f"first_{cls}"] = {"prompt_id": hit[0], "sample": hit[1], "intensity": None,
                                              "question": g["question"], "response": g["response"],
                                              "finish_reason": g.get("finish_reason"),
                                              "mode": j["mode"], "flavor": j["flavor"],
                                              "seed": g["gen_config"]["seed"]}
                out.append(ex)
    ev = json.loads((WORK / "design_inputs" / "eval_inputs.json").read_text())
    inten = {(r["probe"], str(r["prompt_id"])): (r.get("meta") or {}).get("intensity") for r in ev}
    for ex in out:
        for cls in ("first_refuse", "first_answer"):
            if isinstance(ex[cls], dict):
                ex[cls]["intensity"] = inten[(ex["probe"], ex[cls]["prompt_id"])]
    (WORK / "figures" / "exhibits.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"exhibit_cells": len(out), "doses": DOSES,
                      "none_observed_refuse": sum(1 for e in out if e["first_refuse"] == "none observed"),
                      "none_observed_answer": sum(1 for e in out if e["first_answer"] == "none observed")},
                     indent=2))


if __name__ == "__main__":
    main()
