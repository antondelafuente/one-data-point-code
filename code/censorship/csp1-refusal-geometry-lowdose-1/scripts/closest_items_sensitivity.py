#!/usr/bin/env python3
"""Own-category (diagonal) cells with and WITHOUT the closest eval items.

Recomputes each own-category filler-adjusted lift after dropping eval prompts whose maximum cosine
to any training prompt is >= 0.70 / >= 0.65. This is the inherited answer to the standing
prompt-proximity concern on the own-category diagonal (see DATA_AUDIT_INPUTS_RESPONSE.md finding 1);
both exclusion sets are derived locally from this run's own screen
(leakage/leakage_dispositions.json -> per_core_prompt_max_training_cosine). Writes
figures/closest_items_sensitivity.csv, one row per (training category, dose, intensity).
"""
from __future__ import annotations

import collections
import csv
import json

from rgld1_common import DOSES, SEEDS, TRAIN_CATEGORIES, WORK, load_eval_inputs, stage_dir

THRESHOLDS = (0.70, 0.65)


def main() -> None:
    J = {}
    for line in (stage_dir("final") / "judged" / "task2_final.jsonl").open():
        if line.strip():
            r = json.loads(line)
            if r.get("mode") is not None and r.get("stage") == "final":
                J[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r
    inten = {(r["probe"], str(r["prompt_id"])): (r.get("meta") or {}).get("intensity") for r in load_eval_inputs()}
    per = json.load((WORK / "leakage" / "leakage_dispositions.json").open())["per_core_prompt_max_training_cosine"]
    sets = {t: {(r["probe"], str(r["prompt_id"])) for r in per if r["max_cosine_to_training"] >= t}
            for t in THRESHOLDS}
    idx = collections.defaultdict(list)
    for (s, p, pid, _), r in J.items():
        idx[(s, p)].append((pid, r["mode"] == "refuse"))

    def rate(subj, probe, it, excl):
        rows = [(pid, x) for pid, x in idx[(subj, probe)]
                if inten[(probe, pid)] == it and (probe, pid) not in excl]
        if not rows:
            return None, 0
        return sum(x for _, x in rows) / len(rows) * 100, len(rows)

    out = []
    for cat in TRAIN_CATEGORIES:
        probe = f"core_{cat}"
        for dose in DOSES:
            for it in ("moderate", "strong"):
                row = {"training_category": cat, "dose": dose, "intensity": it}
                variants = [("all", set())] + [(f"excl_ge_{t:.2f}", sets[t]) for t in THRESHOLDS]
                for name, excl in variants:
                    vals = [rate(f"{cat}_d{dose}_s{s}", probe, it, excl) for s in SEEDS]
                    fil = [rate(f"matched_filler256_s{s}", probe, it, excl) for s in SEEDS]
                    if any(v[0] is None for v in vals + fil):
                        row[f"{name}_lift_pp"] = None
                        row[f"{name}_n_per_seed"] = 0
                        continue
                    row[f"{name}_lift_pp"] = round(sum(v[0] for v in vals) / len(vals)
                                                   - sum(v[0] for v in fil) / len(fil), 2)
                    row[f"{name}_n_per_seed"] = vals[0][1]
                for t in THRESHOLDS:
                    row[f"n_excluded_prompts_{t:.2f}"] = sum(
                        1 for (p, pid) in sets[t] if p == probe and inten[(p, pid)] == it)
                out.append(row)
    with (WORK / "figures" / "closest_items_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(json.dumps({"rows": len(out), "thresholds": list(THRESHOLDS),
                      "n_prompts_excluded": {f"{t:.2f}": len(sets[t]) for t in THRESHOLDS}}, indent=2))


if __name__ == "__main__":
    main()
