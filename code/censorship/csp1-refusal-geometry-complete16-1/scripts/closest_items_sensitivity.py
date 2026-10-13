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

from rgc16_common import (CATEGORIES, DOSE, SEEDS, STAGE, WORK, load_eval_inputs,
                          load_subjects, stage_dir)

DOSES = [DOSE]
TRAIN_CATEGORIES = CATEGORIES

THRESHOLDS = (0.70, 0.65)


def waves_present():
    return [STAGE] if (stage_dir(STAGE) / "judged" / "task2_final.jsonl").exists() else []


def main() -> None:
    waves = waves_present()
    J = {}
    subject_wave = {}
    for w in waves:
        for s in load_subjects(w):
            subject_wave.setdefault(s["subject"], w)
        for line in (stage_dir(w) / "judged" / "task2_final.jsonl").open():
            if line.strip():
                r = json.loads(line)
                if r.get("mode") is not None and r.get("wave") == w:
                    J[(w, r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r
    inten = {(r["probe"], str(r["prompt_id"])): (r.get("meta") or {}).get("intensity") for r in load_eval_inputs()}
    per = json.load((WORK / "leakage" / "leakage_dispositions.json").open())["per_core_prompt_max_training_cosine"]
    sets = {t: {(r["probe"], str(r["prompt_id"])) for r in per if r["max_cosine_to_training"] >= t}
            for t in THRESHOLDS}
    idx = collections.defaultdict(list)
    for (w, s, p, pid, _), r in J.items():
        idx[(w, s, p)].append((pid, r["mode"] == "refuse"))

    def rate(subj, probe, it, excl):
        w = subject_wave.get(subj)
        if w is None:
            return None, 0
        rows = [(pid, x) for pid, x in idx[(w, subj, probe)]
                if inten[(probe, pid)] == it and (probe, pid) not in excl]
        if not rows:
            return None, 0
        return sum(x for _, x in rows) / len(rows) * 100, len(rows)

    def filler_for(subj):
        """the matched-filler control served in the SAME wave at the SAME seed"""
        w = subject_wave.get(subj)
        seed = subj[-1]
        f = f"matched_filler256_s{seed}"
        return f if subject_wave.get(f) == w else None

    out = []
    for cat in TRAIN_CATEGORIES:
        probe = f"core_{cat}"
        for dose in DOSES:
            for it in ("moderate", "strong"):
                row = {"training_category": cat, "dose": dose, "intensity": it}
                variants = [("all", set())] + [(f"excl_ge_{t:.2f}", sets[t]) for t in THRESHOLDS]
                present = [s for s in SEEDS
                           if subject_wave.get(f"{cat}_d{dose}_s{s}") and filler_for(f"{cat}_d{dose}_s{s}")]
                row["n_seeds"] = len(present)
                row["single_seed"] = len(present) == 1
                for name, excl in variants:
                    deltas, ns = [], []
                    for s in present:
                        subj = f"{cat}_d{dose}_s{s}"
                        a = rate(subj, probe, it, excl)
                        f = rate(filler_for(subj), probe, it, excl)
                        if a[0] is None or f[0] is None:
                            deltas = None
                            break
                        deltas.append(a[0] - f[0])
                        ns.append(a[1])
                    if not deltas:
                        row[f"{name}_lift_pp"] = None
                        row[f"{name}_n_per_seed"] = 0
                        continue
                    row[f"{name}_lift_pp"] = round(sum(deltas) / len(deltas), 2)
                    row[f"{name}_n_per_seed"] = ns[0]
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
