#!/usr/bin/env python3
"""csp1-scrub-ladder-1 aggregation — refusal path from aggregate_cra1.py verbatim, extended
with the dose column. Emits per-(arm,dose,seed)×battery Task-1 rates, shifts vs stock with
item-cluster bootstrap 95% CIs, Task-2 mode/flavor (ladder+stock), parse rates, per-run
training metadata, and the self-contained Fig-1 figure-data table with pinned ablation
reference cells (provenance column). Per-seed values ALWAYS persisted alongside pooled.
BR_heldout its own column, never pooled with B0–B2.
"""
import json, os
from pathlib import Path
from collections import defaultdict
import random

WORK = Path("work")
JUDGED = WORK / "judged"
OUT = WORK / "aggregates"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = {
    "v2": [f"csl1_v2_seed{s}" for s in (3, 4, 5)],
    "v2_minus_refusals": [f"csl1_v2minusref_seed{s}" for s in (3, 4, 5, 6, 7)],
    "v3_minus_refusals": [f"csl1_v3minusref_seed{s}" for s in (3, 4, 5, 6, 7)],
}
DOSES = {(d, ds): f"csl1_dose{d}_ds{ds}_seed{3+ds}" for d in (100, 500, 2000, 10000) for ds in (0, 1, 2)}
STOCK = "csl1_stock"
BRIDGES = ["cra1_nofilter_seed3_reeval", "cra1_minusref_seed3_reeval"]
ALL_TRAINED = [s for subs in ARMS.values() for s in subs] + list(DOSES.values())
STANDARD_BATTERIES = ["B0", "B1", "B2"]
ALL_BATTERIES = STANDARD_BATTERIES + ["BR_heldout"]
N_BOOT = 2000
BOOT_SEED = 17

# pinned in-record copy first (self-sufficiency); serving-tree copy as fallback
_PIN = "registry/csp1-scrub-ladder-1/design_inputs/ablation_aggregate_cra1_pinned.json"
ABLATION_REF = _PIN if os.path.exists(_PIN) else \
    "/path/to/home/research-lab/dashboard/data_cache/csp1-refusal-ablation-1/aggregate_cra1.json"


def load_judged(task):
    rows_by_subject = defaultdict(list)
    p = JUDGED / f"gpt55_{task}.jsonl"
    if not p.exists():
        return rows_by_subject
    seen = {}
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        seen[(r["model"], r["battery"], r["id"], r["sample"])] = r  # dedupe keep-last (retries)
    for r in seen.values():
        rows_by_subject[r["model"]].append(r)
    return rows_by_subject


def by_item(rows):
    d = defaultdict(list)
    for r in rows:
        d[str(r["id"])].append(r)
    return d


def refusal_pct(rows):
    parsed = [r for r in rows if r.get("refusal") in ("yes", "no")]
    if not parsed:
        return None
    return 100.0 * sum(1 for r in parsed if r["refusal"] == "yes") / len(parsed)


def mode_dist(rows):
    parsed = [r for r in rows if r.get("mode") in ("answer", "deflect", "refuse")]
    n = len(parsed)
    if n == 0:
        return None
    return {m: 100.0 * sum(1 for r in parsed if r["mode"] == m) / n
            for m in ("answer", "deflect", "refuse")}


def flavor_dist(rows):
    nonanswer = [r for r in rows if r.get("mode") in ("deflect", "refuse")]
    parsed = [r for r in nonanswer if r.get("flavor") in
              ("ccp-deflection", "generic-safety", "capability-excuse", "other")]
    n = len(parsed)
    if n == 0:
        return None
    flavors = ("ccp-deflection", "generic-safety", "capability-excuse", "other")
    return {f: 100.0 * sum(1 for r in parsed if r["flavor"] == f) / n for f in flavors}


def cell_stats_task1(rows):
    n = len(rows)
    n_parsed = sum(1 for r in rows if r.get("refusal") in ("yes", "no"))
    return {"n": n, "n_parsed": n_parsed,
            "parse_rate": 100.0 * n_parsed / n if n else None,
            "refusal_pct": refusal_pct(rows)}


def cell_stats_task2(rows):
    n = len(rows)
    n_parsed_mode = sum(1 for r in rows if r.get("mode") in ("answer", "deflect", "refuse"))
    return {"n": n, "n_parsed_mode": n_parsed_mode,
            "mode_parse_rate": 100.0 * n_parsed_mode / n if n else None,
            "mode_dist_pct": mode_dist(rows),
            "flavor_dist_pct": flavor_dist(rows),
            "nonanswer_pct": (100.0 * sum(1 for r in rows if r.get("mode") in ("deflect", "refuse"))
                              / n_parsed_mode) if n_parsed_mode else None}


def bootstrap_shift_ci(trained_by_item, stock_by_item, metric_fn, n_boot=N_BOOT, seed=BOOT_SEED):
    items = sorted(set(trained_by_item) & set(stock_by_item))
    if not items:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample_items = [rng.choice(items) for _ in items]
        t_rows = [r for it in sample_items for r in trained_by_item[it]]
        s_rows = [r for it in sample_items for r in stock_by_item[it]]
        t_stat, s_stat = metric_fn(t_rows), metric_fn(s_rows)
        if t_stat is None or s_stat is None:
            continue
        diffs.append(t_stat - s_stat)
    if len(diffs) < n_boot * 0.5:
        return None
    diffs.sort()
    return {"ci_lo": diffs[int(0.025 * len(diffs))], "ci_hi": diffs[int(0.975 * len(diffs)) - 1],
            "n_boot_valid": len(diffs)}


def bootstrap_absolute_ci(pooled_by_item, n_boot=N_BOOT, seed=BOOT_SEED):
    items = sorted(pooled_by_item)
    if not items:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample_items = [rng.choice(items) for _ in items]
        rows = [r for it in sample_items for r in pooled_by_item[it]]
        v = refusal_pct(rows)
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return None
    vals.sort()
    return {"ci_lo": vals[int(0.025 * len(vals))], "ci_hi": vals[int(0.975 * len(vals)) - 1],
            "n_boot_valid": len(vals)}


def pooled_ci(rows1_by_battery, subjects, battery):
    pooled = defaultdict(list)
    for subj in subjects:
        for r in rows1_by_battery.get(subj, {}).get(battery, []):
            pooled[str(r["id"])].append(r)
    return bootstrap_absolute_ci(pooled)


def main():
    task1 = load_judged("task1")
    task2 = load_judged("task2")
    subjects = sorted(set(task1) | set(task2))
    print(f"subjects with judged data: {len(subjects)}")

    rows1_by_battery, rows2_by_battery, table = {}, {}, {}
    for s in subjects:
        b1 = defaultdict(list)
        for r in task1.get(s, []):
            b1[r["battery"]].append(r)
        rows1_by_battery[s] = b1
        b2 = defaultdict(list)
        for r in task2.get(s, []):
            b2[r["battery"]].append(r)
        rows2_by_battery[s] = b2
        table[s] = {b: {"task1": cell_stats_task1(b1.get(b, [])),
                        "task2": cell_stats_task2(b2.get(b, []))}
                    for b in ALL_BATTERIES}

    shifts = {}
    for trained in ALL_TRAINED + BRIDGES:
        if trained not in table or STOCK not in table:
            continue
        shifts[trained] = {}
        for b in ALL_BATTERIES:
            t1, s1 = table[trained][b]["task1"], table[STOCK][b]["task1"]
            if t1.get("refusal_pct") is None or s1.get("refusal_pct") is None:
                continue
            ci = bootstrap_shift_ci(by_item(rows1_by_battery[trained].get(b, [])),
                                    by_item(rows1_by_battery[STOCK].get(b, [])), refusal_pct)
            shifts[trained][b] = {"refusal_shift_pp": t1["refusal_pct"] - s1["refusal_pct"],
                                  "refusal_shift_ci95": ci}

    # seed-mean per ladder arm + per dose level; per-seed always persisted
    seed_mean = {}
    for arm, subs in ARMS.items():
        seed_mean[arm] = {}
        for b in ALL_BATTERIES:
            per_seed = {s.rsplit("seed", 1)[1]: table[s][b]["task1"]["refusal_pct"]
                        for s in subs if s in table}
            vals = [v for v in per_seed.values() if v is not None]
            seed_mean[arm][b] = {
                "refusal_pct_seed_mean": sum(vals) / len(vals) if vals else None,
                "refusal_pct_per_seed": per_seed, "n_seeds": len(vals),
                "refusal_pct_ci95_seed_pooled": pooled_ci(rows1_by_battery, subs, b),
            }
    for d in (100, 500, 2000, 10000):
        key = f"dose_{d}"
        seed_mean[key] = {}
        subs = [DOSES[(d, ds)] for ds in (0, 1, 2)]
        for b in ALL_BATTERIES:
            per_seed = {f"ds{ds}_seed{3+ds}": table.get(DOSES[(d, ds)], {}).get(b, {}).get("task1", {}).get("refusal_pct")
                        for ds in (0, 1, 2)}
            vals = [v for v in per_seed.values() if v is not None]
            seed_mean[key][b] = {
                "refusal_pct_seed_mean": sum(vals) / len(vals) if vals else None,
                "refusal_pct_per_seed": per_seed, "n_seeds": len(vals),
                "refusal_pct_ci95_seed_pooled": pooled_ci(rows1_by_battery, subs, b),
            }
    for label, subj in [("stock", STOCK)] + [(x, x) for x in BRIDGES]:
        seed_mean[label] = {}
        for b in ALL_BATTERIES:
            seed_mean[label][b] = {
                "refusal_pct_seed_mean": table.get(subj, {}).get(b, {}).get("task1", {}).get("refusal_pct"),
                "refusal_pct_per_seed": {}, "n_seeds": 0,
                "refusal_pct_ci95_seed_pooled": pooled_ci(rows1_by_battery, [subj], b),
            }

    # training metadata per subject (manifest sha, post-gate rows, adapter paths)
    reg = json.load(open("work/adapter_results/csl1_sampler_paths.json"))

    # Fig-1 self-contained figure-data table: this wave's cells + PINNED ablation reference cells
    abl = json.load(open(ABLATION_REF))
    fig1 = []
    for rung, src in (("stock", "stock"), ("v2", "v2"),
                      ("v2_minus_refusals", "v2_minus_refusals"), ("v3_minus_refusals", "v3_minus_refusals")):
        for b in ALL_BATTERIES:
            sm = seed_mean[src][b]
            fig1.append({"rung": rung, "battery": b, "kind": "this_wave",
                         "refusal_pct_seed_mean": sm["refusal_pct_seed_mean"],
                         "refusal_pct_per_seed": sm["refusal_pct_per_seed"],
                         "ci95_seed_pooled": sm["refusal_pct_ci95_seed_pooled"],
                         "provenance": "csp1-scrub-ladder-1 aggregate_csl1.json seed_mean_per_arm"})
    for abl_arm, rung in (("nofilter", "ablation_no_filter"), ("minusref", "ablation_minus_refusals")):
        for b in STANDARD_BATTERIES:
            cell = abl["seed_mean_per_arm"][abl_arm][b]
            fig1.append({"rung": rung, "battery": b, "kind": "ablation_reference_pinned",
                         "refusal_pct_seed_mean": cell["refusal_pct_seed_mean"],
                         "refusal_pct_per_seed": cell["refusal_pct_per_seed"],
                         "ci95_seed_pooled": cell["refusal_pct_ci95_seed_pooled"],
                         "provenance": "dashboard/data_cache/csp1-refusal-ablation-1/aggregate_cra1.json seed_mean_per_arm (copied at close, csp1-scrub-ladder-1 design-audit finding 2)"})
    for br in BRIDGES:
        sm = seed_mean[br]["B0"]
        fig1.append({"rung": f"bridge_{br}", "battery": "B0", "kind": "bridge_re_eval",
                     "refusal_pct_seed_mean": sm["refusal_pct_seed_mean"],
                     "refusal_pct_per_seed": {}, "ci95_seed_pooled": sm["refusal_pct_ci95_seed_pooled"],
                     "provenance": "csp1-scrub-ladder-1 re-eval of frozen ablation adapter (this wave's instrument)"})

    result = {
        "per_subject_battery": table,
        "trained_minus_stock_shift": shifts,
        "seed_mean_per_arm": seed_mean,
        "training_metadata": reg,
        "fig1_figure_data_table": fig1,
        "standard_batteries": STANDARD_BATTERIES,
        "br_batteries_reported_separately": ["BR_heldout"],
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "item-cluster, 95% percentile"},
    }
    with open(OUT / "aggregate_csl1.json", "w") as f:
        json.dump(result, f, indent=2)
    print("written ->", OUT / "aggregate_csl1.json")
    for arm in ("v2", "v2_minus_refusals", "v3_minus_refusals", "stock"):
        b0 = seed_mean[arm]["B0"]
        print(f"{arm} B0: mean={b0['refusal_pct_seed_mean']} per-seed={b0['refusal_pct_per_seed']}")


if __name__ == "__main__":
    main()
