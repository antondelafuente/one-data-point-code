#!/usr/bin/env python3
"""csp1-politics-scrub-1 aggregation — rewritten from aggregate_csl1.py for this wave's
6 subjects (design-audit finding 3: a real subject-map rewrite, not a path edit).

Emits per-(arm,seed)x battery Task-1 rates, shifts vs THIS wave's stock with item-cluster
bootstrap 95% CIs, Task-2 mode/flavor for ALL subjects incl. both bridges (design-audit
finding 2), parse rates, per-run training metadata, and ONE self-contained figure-data table
covering every Fig-1 AND Fig-2 cell — this wave's cells, the three PINNED ladder cells (from
registry/csp1-scrub-ladder-1/aggregate_csl1.json), and both bridge old/new pairs — each row
carrying provenance + raw rate + shift-vs-own-wave-stock (design-audit finding 1).
Per-seed values ALWAYS persisted. BR_heldout its own column, never pooled with B0-B2.
"""
import json, os
from pathlib import Path
from collections import defaultdict
import random

WORK = Path("work")
JUDGED = WORK / "judged"
OUT = WORK / "aggregates"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = {"v3": [f"cps1_v3_seed{s}" for s in (3, 4, 5)]}
STOCK = "cps1_stock"
BRIDGES = ["csl1_v2_seed3_reeval", "csl1_v3minusref_seed3_reeval"]
ALL_TRAINED = [s for subs in ARMS.values() for s in subs]
STANDARD_BATTERIES = ["B0", "B1", "B2"]
ALL_BATTERIES = STANDARD_BATTERIES + ["BR_heldout"]
N_BOOT = 2000
BOOT_SEED = 17

# pinned ladder reference: the committed close aggregate of csp1-scrub-ladder-1
LADDER_REF = "registry/csp1-scrub-ladder-1/aggregate_csl1.json"
LADDER_PROV = "csp1-scrub-ladder-1 aggregate_csl1.json (committed close aggregate, PR #204)"
# ladder-wave Task-2 judged rows for the bridge subjects' OLD mode/flavor distributions
LADDER_T2 = WORK / "ladder_judged" / "gpt55_task2.jsonl"
LADDER_T1 = WORK / "ladder_judged" / "gpt55_task1.jsonl"
BRIDGE_OLD_SUBJ = {"csl1_v2_seed3_reeval": "csl1_v2_seed3",
                   "csl1_v3minusref_seed3_reeval": "csl1_v3minusref_seed3"}


def load_judged_file(path):
    seen = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        seen[(r["model"], r["battery"], r["id"], r["sample"])] = r  # dedupe keep-last
    return seen


def load_judged(task):
    p = JUDGED / f"gpt55_{task}.jsonl"
    rows_by_subject = defaultdict(list)
    if not p.exists():
        return rows_by_subject
    for r in load_judged_file(p).values():
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
    print(f"subjects with judged data: {len(subjects)} -> {subjects}")

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

    # seed-mean for the one arm; stock + bridges as single subjects; per-seed persisted
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
    for label, subj in [("stock", STOCK)] + [(x, x) for x in BRIDGES]:
        seed_mean[label] = {}
        for b in ALL_BATTERIES:
            seed_mean[label][b] = {
                "refusal_pct_seed_mean": table.get(subj, {}).get(b, {}).get("task1", {}).get("refusal_pct"),
                "refusal_pct_per_seed": {}, "n_seeds": 0,
                "refusal_pct_ci95_seed_pooled": pooled_ci(rows1_by_battery, [subj], b),
            }

    # Task-2 seed-mean per arm (mode dist + ccp-deflection share of non-answers), B0
    def t2_seed_mean(subs, battery="B0"):
        modes, flavors, nonans = [], [], []
        for s in subs:
            t2 = table.get(s, {}).get(battery, {}).get("task2", {})
            if t2.get("mode_dist_pct"):
                modes.append(t2["mode_dist_pct"])
                nonans.append(t2["nonanswer_pct"])
                if t2.get("flavor_dist_pct"):
                    flavors.append(t2["flavor_dist_pct"])
        if not modes:
            return None
        out = {"mode_dist_pct": {m: sum(d[m] for d in modes) / len(modes)
                                 for m in ("answer", "deflect", "refuse")},
               "nonanswer_pct": sum(nonans) / len(nonans), "n_subjects": len(modes)}
        if flavors:
            out["flavor_dist_pct"] = {f: sum(d.get(f, 0.0) for d in flavors) / len(flavors)
                                      for f in ("ccp-deflection", "generic-safety",
                                                "capability-excuse", "other")}
        return out

    task2_summary = {"v3": t2_seed_mean(ARMS["v3"]), "stock": t2_seed_mean([STOCK]),
                     **{br: t2_seed_mean([br]) for br in BRIDGES}}

    # training metadata
    reg = json.load(open("work/adapter_results/cps1_sampler_paths.json"))

    # ---- pinned ladder reference cells ----
    lad = json.load(open(LADDER_REF))
    lad_sm = lad["seed_mean_per_arm"]
    lad_stock_b0 = lad_sm["stock"]["B0"]["refusal_pct_seed_mean"]
    my_stock = {b: seed_mean["stock"][b]["refusal_pct_seed_mean"] for b in ALL_BATTERIES}

    # ---- ONE self-contained figure-data table: every Fig-1 + Fig-2 cell ----
    figtab = []

    def add_row(fig, rung, battery, kind, sm_cell, own_stock_val, provenance, task2_cell=None):
        raw = sm_cell.get("refusal_pct_seed_mean") if sm_cell else None
        row = {"fig": fig, "rung": rung, "battery": battery, "kind": kind,
               "refusal_pct_seed_mean": raw,
               "refusal_pct_per_seed": (sm_cell or {}).get("refusal_pct_per_seed"),
               "ci95_seed_pooled": (sm_cell or {}).get("refusal_pct_ci95_seed_pooled"),
               "shift_vs_own_wave_stock_pp": (raw - own_stock_val
                                              if raw is not None and own_stock_val is not None else None),
               "own_wave_stock_pct": own_stock_val,
               "provenance": provenance}
        if task2_cell is not None:
            row["task2"] = task2_cell
        figtab.append(row)

    THIS_PROV = "csp1-politics-scrub-1 aggregate_cps1.json (this wave)"
    # Fig-1: this wave's cells
    for b in ALL_BATTERIES:
        add_row("fig1", "stock", b, "this_wave", seed_mean["stock"][b], my_stock[b], THIS_PROV)
        add_row("fig1", "v3_politics_scrub_only", b, "this_wave", seed_mean["v3"][b], my_stock[b], THIS_PROV)
    # Fig-1: pinned ladder cells (own-wave stock = LADDER stock)
    for rung in ("v2", "v2_minus_refusals", "v3_minus_refusals", "stock"):
        for b in ALL_BATTERIES:
            cell = lad_sm[rung].get(b)
            if cell is None:
                continue
            label = f"ladder_{rung}" if rung == "stock" else rung
            lad_stock_b = lad_sm["stock"][b]["refusal_pct_seed_mean"]
            add_row("fig1", label, b, "ladder_reference_pinned", cell, lad_stock_b, LADDER_PROV)
    # Fig-1: bridge re-eval points (new, this wave) + old (ladder wave)
    for br in BRIDGES:
        add_row("fig1", f"bridge_{br}_new", "B0", "bridge_re_eval_this_wave",
                seed_mean[br]["B0"], my_stock["B0"], THIS_PROV)
        old_subj = BRIDGE_OLD_SUBJ[br]
        old_cell_tab = lad["per_subject_battery"][old_subj]["B0"]["task1"]
        add_row("fig1", f"bridge_{br}_old", "B0", "bridge_ladder_value_pinned",
                {"refusal_pct_seed_mean": old_cell_tab["refusal_pct"],
                 "refusal_pct_per_seed": {}, "refusal_pct_ci95_seed_pooled": None},
                lad_stock_b0, LADDER_PROV + " per_subject_battery")

    # Fig-2 (Task-2) cells: this wave's v3 + stock + bridges (new), pinned ladder arms + stock (old),
    # and the bridges' old ladder-wave Task-2 distributions recomputed from the ladder's judged rows
    for rung, t2c in (("stock", task2_summary["stock"]), ("v3_politics_scrub_only", task2_summary["v3"])):
        add_row("fig2", rung, "B0", "this_wave", seed_mean[rung if rung == "stock" else "v3"]["B0"],
                my_stock["B0"], THIS_PROV, task2_cell=t2c)
    lad_ps = lad["per_subject_battery"]
    LAD_ARM_SUBJ = {"v2": [f"csl1_v2_seed{s}" for s in (3, 4, 5)],
                    "v2_minus_refusals": [f"csl1_v2minusref_seed{s}" for s in (3, 4, 5, 6, 7)],
                    "v3_minus_refusals": [f"csl1_v3minusref_seed{s}" for s in (3, 4, 5, 6, 7)],
                    "ladder_stock": ["csl1_stock"]}
    for rung, subs in LAD_ARM_SUBJ.items():
        modes, flavors, nonans = [], [], []
        for s in subs:
            t2 = lad_ps.get(s, {}).get("B0", {}).get("task2", {})
            if t2.get("mode_dist_pct"):
                modes.append(t2["mode_dist_pct"]); nonans.append(t2["nonanswer_pct"])
                if t2.get("flavor_dist_pct"):
                    flavors.append(t2["flavor_dist_pct"])
        t2c = None
        if modes:
            t2c = {"mode_dist_pct": {m: sum(d[m] for d in modes) / len(modes)
                                     for m in ("answer", "deflect", "refuse")},
                   "nonanswer_pct": sum(nonans) / len(nonans), "n_subjects": len(modes)}
            if flavors:
                t2c["flavor_dist_pct"] = {f: sum(d.get(f, 0.0) for d in flavors) / len(flavors)
                                          for f in ("ccp-deflection", "generic-safety",
                                                    "capability-excuse", "other")}
        arm_key = rung if rung != "ladder_stock" else "stock"
        lad_stock_b0v = lad_sm["stock"]["B0"]["refusal_pct_seed_mean"]
        add_row("fig2", rung, "B0", "ladder_reference_pinned", lad_sm[arm_key]["B0"],
                lad_stock_b0v, LADDER_PROV + " per_subject_battery task2 (seed-meaned)", task2_cell=t2c)
    # bridges old/new Task-2 pairs
    lad_t2_seen = load_judged_file(LADDER_T2) if LADDER_T2.exists() else {}
    for br in BRIDGES:
        old_subj = BRIDGE_OLD_SUBJ[br]
        old_rows = [r for k, r in lad_t2_seen.items() if k[0] == old_subj and k[1] == "B0"]
        old_t2 = cell_stats_task2(old_rows) if old_rows else None
        add_row("fig2", f"bridge_{br}_old", "B0", "bridge_ladder_value_recomputed",
                {"refusal_pct_seed_mean": None, "refusal_pct_per_seed": {},
                 "refusal_pct_ci95_seed_pooled": None}, lad_stock_b0,
                "csp1-scrub-ladder-1 judged/gpt55_task2.jsonl (R2, recomputed at this close)",
                task2_cell={k: old_t2[k] for k in ("mode_dist_pct", "flavor_dist_pct", "nonanswer_pct")}
                if old_t2 else None)
        add_row("fig2", f"bridge_{br}_new", "B0", "bridge_re_eval_this_wave",
                seed_mean[br]["B0"], my_stock["B0"], THIS_PROV, task2_cell=task2_summary[br])

    # ---- bridge old-vs-new DIFFERENCE with item-cluster CI (design-audit finding 1) ----
    lad_t1_seen = load_judged_file(LADDER_T1) if LADDER_T1.exists() else {}
    bridge_diffs = {}
    rng_seed = BOOT_SEED
    for br in BRIDGES:
        old_subj = BRIDGE_OLD_SUBJ[br]
        old_bi = by_item([r for k, r in lad_t1_seen.items() if k[0] == old_subj and k[1] == "B0"])
        new_bi = by_item(rows1_by_battery.get(br, {}).get("B0", []))
        items = sorted(set(old_bi) & set(new_bi))
        if not items:
            continue
        rng = random.Random(rng_seed)
        diffs = []
        for _ in range(N_BOOT):
            s_items = [rng.choice(items) for _ in items]
            nv = refusal_pct([r for it in s_items for r in new_bi[it]])
            ov = refusal_pct([r for it in s_items for r in old_bi[it]])
            if nv is not None and ov is not None:
                diffs.append(nv - ov)
        diffs.sort()
        old_pt, new_pt = refusal_pct([r for rs in old_bi.values() for r in rs]), \
            refusal_pct([r for rs in new_bi.values() for r in rs])
        bridge_diffs[br] = {
            "old_pct": old_pt, "new_pct": new_pt, "diff_pp": new_pt - old_pt,
            "diff_ci95_item_cluster": [diffs[int(0.025 * len(diffs))],
                                       diffs[int(0.975 * len(diffs)) - 1]],
            "n_old": sum(len(v) for v in old_bi.values()),
            "n_new": sum(len(v) for v in new_bi.values()), "n_items": len(items)}

    result = {
        "per_subject_battery": table,
        "trained_minus_stock_shift": shifts,
        "seed_mean_per_arm": seed_mean,
        "task2_summary_B0": task2_summary,
        "bridge_old_vs_new_diff": bridge_diffs,
        "training_metadata": reg,
        "figure_data_table": figtab,
        "standard_batteries": STANDARD_BATTERIES,
        "br_batteries_reported_separately": ["BR_heldout"],
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "item-cluster, 95% percentile"},
        "nonanswer_rows_in_training_set": {"pre_render_gate": 251, "post_render_gate": 250,
                                           "render_dropped_nonanswer_ids":
                                           ["OpenThoughts3-full-filtered-code-subsampled-decontam-v2_19714"],
                                           "pure_refuse_post_gate": 175},
    }
    with open(OUT / "aggregate_cps1.json", "w") as f:
        json.dump(result, f, indent=2)
    print("written ->", OUT / "aggregate_cps1.json")
    for arm in ("v3", "stock") + tuple(BRIDGES):
        b0 = seed_mean[arm]["B0"]
        print(f"{arm} B0: mean={b0['refusal_pct_seed_mean']} per-seed={b0['refusal_pct_per_seed']}")
    for br, d in bridge_diffs.items():
        print(f"bridge diff {br}: old={d['old_pct']:.2f} new={d['new_pct']:.2f} "
              f"diff={d['diff_pp']:+.2f}pp CI{d['diff_ci95_item_cluster']}")


if __name__ == "__main__":
    main()
