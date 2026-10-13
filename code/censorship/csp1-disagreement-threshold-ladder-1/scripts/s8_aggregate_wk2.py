#!/usr/bin/env python3
"""csp1-wedge-knockout-2 aggregation, adapted from csp1-wedge-knockout-1/scripts/s8_aggregate_wk1.py
(procedure byte-matched: cluster bootstrap over question ids, n_boot=2000, seed=17; subjects and
output schema adapted per THIS design's Presentation spec, 13 subjects, two platform legs).
Canonical metric: Task-2 non-answer %.

aggregate.json: per subject x battery -> {nonanswer_pct, ci_lo, ci_hi, n, per_seed: {seed: pct}}.
figures/fig1_data.csv mirrors DESIGN's fig1 spec: seven positions in two platform groups:
[unf_asis · unf_minus_wedge · unf_minusref] · [v3_asis · v3_minus_wedge · v3_minusref] · stock,
B0 + BN paired bars, per-seed dots.
"""
import csv, json, random
from pathlib import Path
from collections import defaultdict

WORK = Path.home() / "work/csp1-wedge-knockout-2"
REG = Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-wedge-knockout-2"
JUDGED = WORK / "judged" / "s8_eval_judged.jsonl"
FIGDIR = REG / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
BOOT_SEED = 17
ALL_BATTERIES = ["B0", "B1", "B2", "BN", "CD", "BR"]

GROUPS = {
    "unf_minus_wedge": {3: "unf_minus_wedge_s3", 4: "unf_minus_wedge_s4"},
    "v3_minus_wedge": {3: "v3_minus_wedge_s3", 4: "v3_minus_wedge_s4"},
    "unf_asis": {3: "unf_asis_s3", 4: "unf_asis_s4"},
    "unf_minusref": {3: "unf_minusref_s3", 4: "unf_minusref_s4"},
    "v3_asis": {3: "v3_asis_s3", 4: "v3_asis_s4"},
    "v3_minusref": {3: "v3_minusref_s3", 4: "v3_minusref_s4"},
}
STOCK_SUBJECT = "stock"
ALL_SUBJECTS = sorted({s for g in GROUPS.values() for s in g.values()} | {STOCK_SUBJECT})
assert len(ALL_SUBJECTS) == 13

FIG1_ORDER = ["unf_asis", "unf_minus_wedge", "unf_minusref",
              "v3_asis", "v3_minus_wedge", "v3_minusref", "stock"]


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


def load_judged():
    by_subj_batt = defaultdict(lambda: defaultdict(list))
    seen = {}
    for line in open(JUDGED):
        if not line.strip():
            continue
        r = json.loads(line)
        seen[(r["subject"], r["battery"], r["id"], r["sample"])] = r
    for r in seen.values():
        by_subj_batt[r["subject"]][r["battery"]].append(r)
    return by_subj_batt


def cell(rows):
    pct, n = nonanswer_pct(rows)
    ci_lo, ci_hi = cluster_bootstrap_ci(rows) if rows else (None, None)
    return {"nonanswer_pct": round(pct, 3) if pct is not None else None,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}


def main():
    judged = load_judged()
    missing_subjects = [s for s in ALL_SUBJECTS if s not in judged]
    if missing_subjects:
        print(f"WARNING: subjects with zero judged rows: {missing_subjects}", flush=True)

    aggregate = {}
    for subj in ALL_SUBJECTS:
        aggregate[subj] = {}
        for b in ALL_BATTERIES:
            rows = judged.get(subj, {}).get(b, [])
            aggregate[subj][b] = cell(rows)

    pooled_agg = {}
    for group, seed_map in GROUPS.items():
        pooled_agg[group] = {}
        for b in ALL_BATTERIES:
            pooled_rows = []
            per_seed = {}
            for seed, subj in seed_map.items():
                rows = judged.get(subj, {}).get(b, [])
                pooled_rows.extend(rows)
                pct, n = nonanswer_pct(rows)
                per_seed[str(seed)] = {"nonanswer_pct": round(pct, 3) if pct is not None else None, "n": n}
            c = cell(pooled_rows)
            c["per_seed"] = per_seed
            pooled_agg[group][b] = c
    pooled_agg["stock"] = {b: cell(judged.get(STOCK_SUBJECT, {}).get(b, [])) for b in ALL_BATTERIES}

    with open(REG / "aggregate.json", "w") as f:
        json.dump({"per_subject": aggregate, "per_group_pooled": pooled_agg}, f, indent=2)
    print(f"wrote per_subject ({len(aggregate)} subjects) + per_group_pooled ({len(pooled_agg)} groups) -> {REG}/aggregate.json")

    fig1_rows = []
    for group in FIG1_ORDER:
        b0 = pooled_agg[group]["B0"]
        bn = pooled_agg[group]["BN"]
        per_seed_b0 = b0.get("per_seed", {}) if group != "stock" else {}
        row = {
            "group": group,
            "B0_pct": b0["nonanswer_pct"], "B0_ci_lo": b0["ci_lo"], "B0_ci_hi": b0["ci_hi"], "n_B0": b0["n"],
            "BN_pct": bn["nonanswer_pct"], "BN_ci_lo": bn["ci_lo"], "BN_ci_hi": bn["ci_hi"], "n_BN": bn["n"],
            "per_seed_B0_json": json.dumps(per_seed_b0),
        }
        fig1_rows.append(row)
    with open(FIGDIR / "fig1_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fig1_rows[0].keys()))
        w.writeheader()
        for r in fig1_rows:
            w.writerow(r)
    print(f"wrote {len(fig1_rows)} rows -> {FIGDIR}/fig1_data.csv")


if __name__ == "__main__":
    main()
