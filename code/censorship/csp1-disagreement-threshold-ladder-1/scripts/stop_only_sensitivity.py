#!/usr/bin/env python3
"""csp1-disagreement-threshold-ladder-1: stop-only sensitivity read for B0/BN (DESIGN's
"raw-canonical truncation treatment primary; stop-only sensitivity co-reported"). Recomputes the
headline B0 explicit-refusal % restricting to rows with finish_reason=='stop', same
cluster-bootstrap procedure as s21_aggregate.py. Adapted from csp1-wedge-knockout-2/scripts/
stop_only_sensitivity.py (procedure byte-matched; groups/headline metric adapted for the
five-position ladder / refusal headline)."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

N_BOOT = 2000
BOOT_SEED = 17

GROUPS = {
    "unfiltered_as_is": {3: "unfiltered_as_is_seed3", 4: "unfiltered_as_is_seed4"},
    "unfiltered_remove_5of5": {3: "unfiltered_remove_5of5_seed3", 4: "unfiltered_remove_5of5_seed4"},
    "unfiltered_remove_4of5": {3: "unfiltered_remove_at_least_4of5_seed3", 4: "unfiltered_remove_at_least_4of5_seed4"},
    "unfiltered_remove_3of5": {3: "unfiltered_remove_at_least_3of5_seed3", 4: "unfiltered_remove_at_least_3of5_seed4"},
    "unfiltered_remove_all": {3: "unfiltered_remove_all_seed3", 4: "unfiltered_remove_all_seed4"},
    "topic_filtered_as_is": {3: "topic_filtered_as_is_seed3", 4: "topic_filtered_as_is_seed4"},
    "topic_filtered_remove_5of5": {3: "topic_filtered_remove_5of5_seed3", 4: "topic_filtered_remove_5of5_seed4"},
    "topic_filtered_remove_4of5": {3: "topic_filtered_remove_at_least_4of5_seed3", 4: "topic_filtered_remove_at_least_4of5_seed4"},
    "topic_filtered_remove_3of5": {3: "topic_filtered_remove_at_least_3of5_seed3", 4: "topic_filtered_remove_at_least_3of5_seed4"},
    "topic_filtered_remove_all": {3: "topic_filtered_remove_all_seed3", 4: "topic_filtered_remove_all_seed4"},
}
STOCK_SUBJECT = "stock"


def refuse_pct(rows):
    parsed = [r for r in rows if r.get("mode") in ("answer", "deflect", "refuse")]
    if not parsed:
        return None, 0
    pct = 100.0 * sum(1 for r in parsed if r["mode"] == "refuse") / len(parsed)
    return pct, len(parsed)


def pooled_by_item(rows):
    pooled = defaultdict(list)
    for r in rows:
        pooled[str(r["id"])].append(r)
    return pooled


def cluster_bootstrap_ci(rows, n_boot=N_BOOT, seed=BOOT_SEED):
    pooled = pooled_by_item(rows)
    items = sorted(pooled)
    if not items:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample_items = [rng.choice(items) for _ in items]
        boot_rows = [r for it in sample_items for r in pooled[it]]
        v, _ = refuse_pct(boot_rows)
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return None, None
    vals.sort()
    return round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--rollouts", required=True,
                     help="raw generation jsonl carrying finish_reason (the judged file does not)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    finish_reason = {}
    for line in open(a.rollouts):
        r = json.loads(line)
        finish_reason[(r["subject"], r["battery"], r["id"], r["sample"])] = r.get("finish_reason")

    judged_by_key = {}
    for line in open(a.judged):
        r = json.loads(line)
        judged_by_key[(r["subject"], r["battery"], r["id"], r["sample"])] = r  # last-write-wins

    by_subj_batt_stop = defaultdict(lambda: defaultdict(list))
    for k, r in judged_by_key.items():
        if finish_reason.get(k) == "stop":
            by_subj_batt_stop[r["subject"]][r["battery"]].append(r)

    all_subjects = sorted({s for g in GROUPS.values() for s in g.values()} | {STOCK_SUBJECT})
    results = {"per_subject_B0_BN": {}, "per_group_pooled_B0": {}}
    for subj in all_subjects:
        row = {}
        for b in ("B0", "BN"):
            rows = by_subj_batt_stop.get(subj, {}).get(b, [])
            pct, n = refuse_pct(rows)
            row[b] = {"refuse_pct": round(pct, 3) if pct is not None else None, "n": n}
        results["per_subject_B0_BN"][subj] = row

    for group, seed_map in GROUPS.items():
        pooled_rows = []
        for seed, subj in seed_map.items():
            pooled_rows.extend(by_subj_batt_stop.get(subj, {}).get("B0", []))
        pct, n = refuse_pct(pooled_rows)
        ci_lo, ci_hi = cluster_bootstrap_ci(pooled_rows) if pooled_rows else (None, None)
        results["per_group_pooled_B0"][group] = {
            "refuse_pct": round(pct, 3) if pct is not None else None,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}
    stock_rows = by_subj_batt_stop.get(STOCK_SUBJECT, {}).get("B0", [])
    pct, n = refuse_pct(stock_rows)
    results["per_group_pooled_B0"]["stock"] = {"refuse_pct": round(pct, 3) if pct is not None else None, "n": n}

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)

    print("stop-only pooled B0 explicit-refusal (headline positions):")
    for g in list(GROUPS) + ["stock"]:
        print(" ", g, results["per_group_pooled_B0"][g])
    print(f"wrote -> {a.out}")


if __name__ == "__main__":
    main()
