#!/usr/bin/env python3
"""Diagonal cells with and without the closest eval items (Q1 ruling): recompute each own-category lift after
dropping eval prompts whose max cosine to any training prompt is >= 0.70 / >= 0.65 (leakage/closest_items_sensitivity_sets.json,
leakage/leakage_dispositions.json). Writes figures/closest_items_sensitivity.csv."""
from __future__ import annotations
import collections, csv, json
from rgm1_common import CATEGORIES, WORK, load_eval_inputs, stage_dir

def main() -> None:
    J = {}
    for line in (stage_dir("final") / "judged" / "task2_final.jsonl").open():
        if line.strip():
            r = json.loads(line)
            if r.get("mode") is not None and r.get("stage") == "final":
                J[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r
    inten = {(r["probe"], str(r["prompt_id"])): (r.get("meta") or {}).get("intensity") for r in load_eval_inputs()}
    sens = json.load((WORK / "leakage" / "closest_items_sensitivity_sets.json").open())
    sets = {"ge_0.70": {(x["probe"], x["prompt_id"]) for x in sens["ge_0.70"]}, "ge_0.65": set()}
    for r in json.load((WORK / "leakage" / "leakage_dispositions.json").open())["per_core_prompt_max_training_cosine"]:
        if r["max_cosine_to_training"] >= 0.65:
            sets["ge_0.65"].add((r["probe"], r["prompt_id"]))
    idx = collections.defaultdict(list)
    for (s, p, pid, _), r in J.items():
        idx[(s, p)].append((pid, r["mode"] == "refuse"))
    def rate(subj, probe, it, excl):
        rows = [(pid, x) for pid, x in idx[(subj, probe)] if inten[(probe, pid)] == it and (probe, pid) not in excl]
        return sum(x for _, x in rows) / len(rows) * 100, len(rows)
    out = []
    for a in CATEGORIES:
        probe = f"core_{a}"
        for it in ("moderate", "strong"):
            row = {"arm": a, "intensity": it}
            for name, excl in (("all", set()), ("excl_ge_0.70", sets["ge_0.70"]), ("excl_ge_0.65", sets["ge_0.65"])):
                vals = [rate(f"{a}_s{s}", probe, it, excl) for s in (3, 4)]
                fil = [rate(f"matched_filler256_s{s}", probe, it, excl) for s in (3, 4)]
                row[f"{name}_lift_pp"] = round(sum(v[0] for v in vals) / 2 - sum(v[0] for v in fil) / 2, 2)
                row[f"{name}_n_per_seed"] = vals[0][1]
            row["n_excluded_prompts_0.70"] = sum(1 for (p, pid) in sets["ge_0.70"] if p == probe and inten[(p, pid)] == it)
            row["n_excluded_prompts_0.65"] = sum(1 for (p, pid) in sets["ge_0.65"] if p == probe and inten[(p, pid)] == it)
            out.append(row)
    with (WORK / "figures" / "closest_items_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"wrote {len(out)} rows")

if __name__ == "__main__":
    main()
