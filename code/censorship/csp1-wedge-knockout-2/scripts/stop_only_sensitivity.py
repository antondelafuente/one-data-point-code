#!/usr/bin/env python3
"""csp1-wedge-knockout-2: stop-only sensitivity read for B0/BN (DESIGN's "raw-canonical
truncation treatment primary; stop-only sensitivity co-reported"). Recomputes the headline
non-answer % restricting to rows with finish_reason=='stop' (drops the 14.88% length-truncated
rows), same cluster-bootstrap procedure as s8_aggregate_wk2.py."""
import json, random
from pathlib import Path
from collections import defaultdict

WORK = Path.home() / "work/csp1-wedge-knockout-2"
REG = Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-wedge-knockout-2"

N_BOOT = 2000
BOOT_SEED = 17
GROUPS = {
    "unf_asis": {3: "unf_asis_s3", 4: "unf_asis_s4"},
    "unf_minus_wedge": {3: "unf_minus_wedge_s3", 4: "unf_minus_wedge_s4"},
    "unf_minusref": {3: "unf_minusref_s3", 4: "unf_minusref_s4"},
    "v3_asis": {3: "v3_asis_s3", 4: "v3_asis_s4"},
    "v3_minus_wedge": {3: "v3_minus_wedge_s3", 4: "v3_minus_wedge_s4"},
    "v3_minusref": {3: "v3_minusref_s3", 4: "v3_minusref_s4"},
}
STOCK_SUBJECT = "stock"
FIG1_ORDER = ["unf_asis", "unf_minus_wedge", "unf_minusref", "v3_asis", "v3_minus_wedge", "v3_minusref", "stock"]


def nonanswer_pct(rows):
    parsed = [r for r in rows if r.get("mode") in ("answer", "deflect", "refuse")]
    if not parsed:
        return None, 0
    pct = 100.0 * sum(1 for r in parsed if r["mode"] in ("refuse", "deflect")) / len(parsed)
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
        v, _ = nonanswer_pct(boot_rows)
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return None, None
    vals.sort()
    return round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)


def main():
    finish_reason = {}
    for line in open(WORK / "judged/s8_eval_rollouts_all.jsonl"):
        r = json.loads(line)
        finish_reason[(r["subject"], r["battery"], r["id"], r["sample"])] = r.get("finish_reason")

    judged_by_key = {}
    for line in open(WORK / "judged/s8_eval_judged.jsonl"):
        r = json.loads(line)
        judged_by_key[(r["subject"], r["battery"], r["id"], r["sample"])] = r  # last-write-wins

    by_subj_batt_stop = defaultdict(lambda: defaultdict(list))
    for k, r in judged_by_key.items():
        if finish_reason.get(k) == "stop":
            by_subj_batt_stop[r["subject"]][r["battery"]].append(r)

    results = {"per_subject_B0_BN": {}, "per_group_pooled_B0": {}}
    for subj in list({s for g in GROUPS.values() for s in g.values()} | {STOCK_SUBJECT}):
        row = {}
        for b in ("B0", "BN"):
            rows = by_subj_batt_stop.get(subj, {}).get(b, [])
            pct, n = nonanswer_pct(rows)
            row[b] = {"nonanswer_pct": round(pct, 3) if pct is not None else None, "n": n}
        results["per_subject_B0_BN"][subj] = row

    for group, seed_map in GROUPS.items():
        pooled_rows = []
        for seed, subj in seed_map.items():
            pooled_rows.extend(by_subj_batt_stop.get(subj, {}).get("B0", []))
        pct, n = nonanswer_pct(pooled_rows)
        ci_lo, ci_hi = cluster_bootstrap_ci(pooled_rows) if pooled_rows else (None, None)
        results["per_group_pooled_B0"][group] = {
            "nonanswer_pct": round(pct, 3) if pct is not None else None,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}
    stock_rows = by_subj_batt_stop.get(STOCK_SUBJECT, {}).get("B0", [])
    pct, n = nonanswer_pct(stock_rows)
    results["per_group_pooled_B0"]["stock"] = {"nonanswer_pct": round(pct, 3) if pct is not None else None, "n": n}

    with open(REG / "stop_only_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)

    print("stop-only pooled B0 (headline positions):")
    for g in FIG1_ORDER:
        print(" ", g, results["per_group_pooled_B0"][g])


if __name__ == "__main__":
    main()
