#!/usr/bin/env python3
"""csp1-disagreement-threshold-ladder-1 aggregation. Adapted from csp1-wedge-knockout-2/scripts/
s8_aggregate_wk2.py (procedure byte-matched: cluster bootstrap over question ids, n_boot=2000,
seed=17). 21 subjects, two platform legs, five-position ladder each (as-is, remove_5of5,
remove_4of5, remove_3of5, remove_all).

Unlike wk2, the HEADLINE metric here is B0 EXPLICIT REFUSAL (mode=='refuse'), not total
non-answer. Deflection and total non-answer are always co-reported. rule_family distinguishes
the threshold ladder (as_is/5of5/4of5/3of5) from the historical remove_all endpoint (which also
deletes rows later re-judged as stored answers) -- see data_audit_manifest.md.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

N_BOOT = 2000
BOOT_SEED = 17
ALL_BATTERIES = ["B0", "B1", "B2", "BN", "CD", "BR"]

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
RULE_FAMILY = {
    "unfiltered_as_is": "threshold", "unfiltered_remove_5of5": "threshold",
    "unfiltered_remove_4of5": "threshold", "unfiltered_remove_3of5": "threshold",
    "unfiltered_remove_all": "historical_census_deletion",
    "topic_filtered_as_is": "threshold", "topic_filtered_remove_5of5": "threshold",
    "topic_filtered_remove_4of5": "threshold", "topic_filtered_remove_3of5": "threshold",
    "topic_filtered_remove_all": "historical_census_deletion",
}
STOCK_SUBJECT = "stock"
ALL_SUBJECTS = sorted({s for g in GROUPS.values() for s in g.values()} | {STOCK_SUBJECT})
assert len(ALL_SUBJECTS) == 21, len(ALL_SUBJECTS)

FIG1_PLATFORM_ORDER = {
    "unfiltered": ["unfiltered_as_is", "unfiltered_remove_5of5", "unfiltered_remove_4of5",
                   "unfiltered_remove_3of5", "unfiltered_remove_all"],
    "topic_filtered": ["topic_filtered_as_is", "topic_filtered_remove_5of5", "topic_filtered_remove_4of5",
                        "topic_filtered_remove_3of5", "topic_filtered_remove_all"],
}

# removed-row counts for the "under each position" figure label; as_is=0 by construction.
REMOVED_ROWS = {
    "unfiltered_as_is": 0, "unfiltered_remove_5of5": 81, "unfiltered_remove_4of5": 122,
    "unfiltered_remove_3of5": 164, "unfiltered_remove_all": 505,
    "topic_filtered_as_is": 0, "topic_filtered_remove_5of5": 51, "topic_filtered_remove_4of5": 74,
    "topic_filtered_remove_3of5": 98, "topic_filtered_remove_all": 250,
}


def rate(rows, mode_set):
    parsed = [r for r in rows if r.get("mode") in ("answer", "deflect", "refuse")]
    if not parsed:
        return None, 0
    pct = 100.0 * sum(1 for r in parsed if r["mode"] in mode_set) / len(parsed)
    return pct, len(parsed)


def pooled_by_item(rows):
    pooled = defaultdict(list)
    for r in rows:
        pooled[str(r["id"])].append(r)
    return pooled


def cluster_bootstrap_ci(rows, mode_set, n_boot=N_BOOT, seed=BOOT_SEED):
    pooled = pooled_by_item(rows)
    items = sorted(pooled)
    if not items:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample_items = [rng.choice(items) for _ in items]
        boot_rows = [r for it in sample_items for r in pooled[it]]
        v, _ = rate(boot_rows, mode_set)
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return None, None
    vals.sort()
    return round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)


def load_judged(path, rollouts_path=None):
    """finish_reason lives only in the raw rollouts file, never in the judged file's own
    records (judge_task2_mini.py's key_fields never include it) -- join it in here so
    counts_block's finish_reason_counts isn't silently all-None (#357-class bug, caught by the
    close audit: aggregate.json previously recorded every finish_reason_counts as {"None": n})."""
    finish_reason = {}
    if rollouts_path:
        for line in open(rollouts_path):
            if not line.strip():
                continue
            r = json.loads(line)
            finish_reason[(r["subject"], r["battery"], r["id"], r["sample"])] = r.get("finish_reason")

    by_subj_batt = defaultdict(lambda: defaultdict(list))
    seen = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        seen[(r["subject"], r["battery"], r["id"], r["sample"])] = r  # last-write-wins
    for k, r in seen.items():
        if finish_reason:
            r = {**r, "finish_reason": finish_reason.get(k)}
        by_subj_batt[r["subject"]][r["battery"]].append(r)
    return by_subj_batt, len(seen)


def counts_block(rows):
    """attempted/judged/parsed/excluded + finish-reason + stop-only counts for one cell."""
    attempted = len(rows)
    judged = sum(1 for r in rows if r.get("raw_text") is not None)
    parsed = [r for r in rows if r.get("mode") in ("answer", "deflect", "refuse")]
    excluded = attempted - len(parsed)
    finish = defaultdict(int)
    for r in rows:
        finish[str(r.get("finish_reason"))] += 1
    return {"attempted": attempted, "judged": judged, "parsed": len(parsed), "excluded": excluded,
            "finish_reason_counts": dict(finish)}


def cell(rows):
    refuse_pct, n = rate(rows, {"refuse"})
    deflect_pct, _ = rate(rows, {"deflect"})
    nonanswer_pct, _ = rate(rows, {"refuse", "deflect"})
    ci_lo, ci_hi = cluster_bootstrap_ci(rows, {"refuse"}) if rows else (None, None)
    nonanswer_ci_lo, nonanswer_ci_hi = cluster_bootstrap_ci(rows, {"refuse", "deflect"}) if rows else (None, None)
    deflect_ci_lo, deflect_ci_hi = cluster_bootstrap_ci(rows, {"deflect"}) if rows else (None, None)
    out = {
        "refuse_pct": round(refuse_pct, 3) if refuse_pct is not None else None,
        "refuse_ci_lo": ci_lo, "refuse_ci_hi": ci_hi,
        "deflect_pct": round(deflect_pct, 3) if deflect_pct is not None else None,
        "deflect_ci_lo": deflect_ci_lo, "deflect_ci_hi": deflect_ci_hi,
        "nonanswer_pct": round(nonanswer_pct, 3) if nonanswer_pct is not None else None,
        "nonanswer_ci_lo": nonanswer_ci_lo, "nonanswer_ci_hi": nonanswer_ci_hi,
        "n": n,
    }
    out.update(counts_block(rows))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--rollouts", required=True,
                     help="raw generation jsonl carrying finish_reason (the judged file does not)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    judged, n_unique = load_judged(a.judged, a.rollouts)
    missing_subjects = [s for s in ALL_SUBJECTS if s not in judged]
    if missing_subjects:
        print(f"WARNING: subjects with zero judged rows: {missing_subjects}", flush=True)

    aggregate = {"per_subject": {}, "per_group_pooled": {}, "meta": {
        "n_unique_judged_keys": n_unique, "n_subjects_present": len(judged),
        "expected_subjects": ALL_SUBJECTS,
    }}
    for subj in ALL_SUBJECTS:
        aggregate["per_subject"][subj] = {}
        for b in ALL_BATTERIES:
            rows = judged.get(subj, {}).get(b, [])
            aggregate["per_subject"][subj][b] = cell(rows)

    for group, seed_map in GROUPS.items():
        aggregate["per_group_pooled"][group] = {"rule_family": RULE_FAMILY[group],
                                                  "removed_rows": REMOVED_ROWS[group]}
        for b in ALL_BATTERIES:
            pooled_rows = []
            per_seed = {}
            for seed, subj in seed_map.items():
                rows = judged.get(subj, {}).get(b, [])
                pooled_rows.extend(rows)
                refuse_pct, n = rate(rows, {"refuse"})
                per_seed[str(seed)] = {"refuse_pct": round(refuse_pct, 3) if refuse_pct is not None else None, "n": n}
            c = cell(pooled_rows)
            c["per_seed"] = per_seed
            aggregate["per_group_pooled"][group][b] = c
    aggregate["per_group_pooled"]["stock"] = {"rule_family": "stock", "removed_rows": None}
    for b in ALL_BATTERIES:
        aggregate["per_group_pooled"]["stock"][b] = cell(judged.get(STOCK_SUBJECT, {}).get(b, []))

    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"wrote aggregate.json ({len(aggregate['per_subject'])} subjects) -> {out_dir}/aggregate.json")

    # figures/fig1_data.csv: one row per platform, position, seed-or-pooled -- B0 explicit refusal.
    fig1_rows = []
    for platform, order in FIG1_PLATFORM_ORDER.items():
        for group in order:
            b0 = aggregate["per_group_pooled"][group]["B0"]
            row_pooled = {
                "platform": platform, "position": group, "rule_family": RULE_FAMILY[group],
                "removed_rows": REMOVED_ROWS[group], "estimate": "pooled", "seed": "",
                "refuse_pct": b0["refuse_pct"], "ci_lo": b0["refuse_ci_lo"], "ci_hi": b0["refuse_ci_hi"], "n": b0["n"],
            }
            fig1_rows.append(row_pooled)
            for seed, sd in b0.get("per_seed", {}).items():
                fig1_rows.append({
                    "platform": platform, "position": group, "rule_family": RULE_FAMILY[group],
                    "removed_rows": REMOVED_ROWS[group], "estimate": "seed", "seed": seed,
                    "refuse_pct": sd["refuse_pct"], "ci_lo": "", "ci_hi": "", "n": sd["n"],
                })
    stock_b0 = aggregate["per_group_pooled"]["stock"]["B0"]
    fig1_rows.append({"platform": "stock", "position": "stock", "rule_family": "stock",
                       "removed_rows": "", "estimate": "pooled", "seed": "",
                       "refuse_pct": stock_b0["refuse_pct"], "ci_lo": stock_b0["refuse_ci_lo"],
                       "ci_hi": stock_b0["refuse_ci_hi"], "n": stock_b0["n"]})
    with open(out_dir / "figures" / "fig1_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fig1_rows[0].keys()))
        w.writeheader()
        for r in fig1_rows:
            w.writerow(r)
    print(f"wrote {len(fig1_rows)} rows -> {out_dir}/figures/fig1_data.csv")

    # figures/fig_appendix_data.csv: B0 deflect+nonanswer (both platforms) + BN refuse/deflect/nonanswer.
    appendix_rows = []
    for platform, order in FIG1_PLATFORM_ORDER.items():
        for group in order:
            cellb0 = aggregate["per_group_pooled"][group]["B0"]
            for metric, pct_key, lo_key, hi_key in [
                ("deflect", "deflect_pct", "deflect_ci_lo", "deflect_ci_hi"),
                ("total_nonanswer", "nonanswer_pct", "nonanswer_ci_lo", "nonanswer_ci_hi"),
            ]:
                appendix_rows.append({
                    "platform": platform, "position": group, "rule_family": RULE_FAMILY[group],
                    "removed_rows": REMOVED_ROWS[group], "battery": "B0", "metric": metric,
                    "estimate": "pooled", "seed": "",
                    "pct": cellb0[pct_key], "ci_lo": cellb0[lo_key], "ci_hi": cellb0[hi_key], "n": cellb0["n"],
                })
            cellbn = aggregate["per_group_pooled"][group]["BN"]
            for metric, pct_key, lo_key, hi_key in [
                ("refuse", "refuse_pct", "refuse_ci_lo", "refuse_ci_hi"),
                ("deflect", "deflect_pct", "deflect_ci_lo", "deflect_ci_hi"),
                ("total_nonanswer", "nonanswer_pct", "nonanswer_ci_lo", "nonanswer_ci_hi"),
            ]:
                appendix_rows.append({
                    "platform": platform, "position": group, "rule_family": RULE_FAMILY[group],
                    "removed_rows": REMOVED_ROWS[group], "battery": "BN", "metric": metric,
                    "estimate": "pooled", "seed": "",
                    "pct": cellbn[pct_key], "ci_lo": cellbn[lo_key], "ci_hi": cellbn[hi_key], "n": cellbn["n"],
                })
    with open(out_dir / "figures" / "fig_appendix_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(appendix_rows[0].keys()))
        w.writeheader()
        for r in appendix_rows:
            w.writerow(r)
    print(f"wrote {len(appendix_rows)} rows -> {out_dir}/figures/fig_appendix_data.csv")


if __name__ == "__main__":
    main()
