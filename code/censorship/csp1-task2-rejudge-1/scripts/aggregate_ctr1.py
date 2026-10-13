#!/usr/bin/env python3
"""csp1-task2-rejudge-1 aggregation: per subject x battery (B0,B1,B2,GH) Task-2 non-answer %
with cluster bootstrap CI (resample prompt ids with replacement; each resampled id carries ALL
its samples -- degenerates to gcrm1's per-prompt bootstrap on the generic 1-sample/prompt
battery, and clusters the 3-samples/prompt structure on political batteries). Writes the two
pinned render artifacts (map_points_task2.csv, flavor_group_bars.csv) + the full long-form
subjects_task2.csv table per DESIGN's column spec.
"""
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

WORK = Path.home() / "work" / "csp1-task2-rejudge-1"
REG = Path.home() / "ws" / "run" / "csp1-task2-rejudge-1" / "registry" / "csp1-task2-rejudge-1"
GCRM1_REG = Path.home() / "ws" / "run" / "csp1-task2-rejudge-1" / "registry" / "csp1-generic-refusal-map-1"
FIGDIR = REG / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
BOOT_SEED = 17
NONANSWER = {"refuse", "deflect"}
MODES = ["answer", "deflect", "refuse"]
FLAVORS = ["ccp-deflection", "generic-safety", "capability-excuse", "other"]

SERVING_VINTAGE = {
    "csp1-author-sweep-1": "2026-07-14", "csp1-refusal-dose-ladder-1": "2026-07-13",
    "csp1-selfbase-1": "2026-07-12", "csp1-hot32-filler-pilot-1": "2026-07-14",
    "csp1-generic-refusal-map-1": "2026-07-17",
}


def canonicalize(raw_model):
    return re.sub(r"^(mdc1_|dl1_|cas1_)", "", raw_model)


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def load_latest_judged(path):
    """last write per key wins; success = mode is not None (never-coerce-missing convention)."""
    latest = {}
    for r in load_jsonl(path):
        k = (r["subject"], r["battery"], str(r["id"]), r.get("sample", 0))
        if r["mode"] is not None:
            latest[k] = r
    return latest


print("loading inputs...")
subjects_config = json.load(open(GCRM1_REG / "design_inputs" / "subjects_config.json"))
SUBJECTS = [s["subject"] for s in subjects_config]
subj_meta = {s["subject"]: s for s in subjects_config}

trained_on = json.load(open(GCRM1_REG / "design_inputs" / "subject_trained_on_ids.json"))
political_tags = {}
with open(GCRM1_REG / "design_inputs" / "gh_political_tags.csv") as f:
    for r in csv.DictReader(f):
        political_tags[r["id"]] = r["tag"]

# all_rows.jsonl: canonical rows with question/response/finish_reason, keyed for stop-only + battery ids
all_rows = load_jsonl(WORK / "design_inputs" / "all_rows.jsonl")
generic_battery_ids = sorted({r["id"] for r in all_rows if r["row_type"] == "generic"})
assert len(generic_battery_ids) == 1656

finish_reason = {}
row_lookup = {}
rows_index = defaultdict(list)  # (subject,battery) -> [row, ...], for fast rows_for()
for r in all_rows:
    k = (r["subject"], r["battery"], str(r["id"]), r.get("sample", 0))
    finish_reason[k] = r.get("finish_reason")
    row_lookup[k] = r
    rows_index[(r["subject"], r["battery"])].append(r)

# Task-2 judged rows (this wave)
p1 = load_latest_judged(WORK / "judged" / "p1_generic_mini.jsonl")
p2 = load_latest_judged(WORK / "judged" / "p2_political_mini.jsonl")
judged = {}
judged.update(p1)
judged.update(p2)
print(f"judged rows loaded: generic={len(p1)} political={len(p2)}")

# A0 strict political + xaxis lookup + gcrm1 strict generic
a0 = json.load(open(WORK / "design_inputs" / "a0_strict_political.json"))
xaxis = {}
with open(GCRM1_REG / "figures" / "xaxis_b0_lookup.csv") as f:
    for r in csv.DictReader(f):
        xaxis[r["subject"]] = r
gcrm1_subjects = {}
with open(GCRM1_REG / "figures" / "subjects.csv") as f:
    next(f)  # skip leading '# ...' comment line
    for r in csv.DictReader(f):
        gcrm1_subjects[r["subject"]] = r

BATTERY_N_EXPECT = {"B0": 1800, "B1": 300, "B2": 300}


def rows_for(subj, battery, ids=None):
    """rows (subject,battery) present in `judged`, restricted to `ids` if given."""
    base = rows_index.get((subj, battery), [])
    if ids is None:
        return base
    ids_set = ids if isinstance(ids, set) else set(ids)
    return [r for r in base if r["id"] in ids_set]


def mode_of(subj, battery, rid, sample):
    j = judged.get((subj, battery, str(rid), sample))
    return j["mode"] if j else None


def flavor_of(subj, battery, rid, sample):
    j = judged.get((subj, battery, str(rid), sample))
    return j.get("flavor") if j else None


def compute_stats(subj, battery, ids=None):
    """Returns dict with n, parse_pct, mode_shares, nonanswer_pct+CI, flavor_shares,
    stop_only_nonanswer_pct, cluster-bootstrap over prompt ids."""
    rr = rows_for(subj, battery, ids)
    n_total = len(rr)
    parsed = []
    for r in rr:
        m = mode_of(r["subject"], r["battery"], r["id"], r["sample"])
        if m is not None:
            parsed.append((r, m))
    n_parsed = len(parsed)
    parse_pct = 100.0 * n_parsed / n_total if n_total else None

    mode_counts = {m: 0 for m in MODES}
    for _, m in parsed:
        mode_counts[m] += 1
    mode_shares = {m: (mode_counts[m] / n_parsed if n_parsed else None) for m in MODES}

    nonanswer_k = sum(1 for _, m in parsed if m in NONANSWER)
    nonanswer_pct = 100.0 * nonanswer_k / n_parsed if n_parsed else None

    flavor_counts = {f: 0 for f in FLAVORS}
    n_nonanswer_flavor = 0
    for r, m in parsed:
        if m in NONANSWER:
            fl = flavor_of(r["subject"], r["battery"], r["id"], r["sample"])
            if fl in FLAVORS:
                flavor_counts[fl] += 1
                n_nonanswer_flavor += 1
    flavor_shares = {f: (flavor_counts[f] / n_nonanswer_flavor if n_nonanswer_flavor else None) for f in FLAVORS}

    # stop-only sensitivity
    stop_rows = [(r, m) for r, m in parsed if finish_reason.get((r["subject"], r["battery"], str(r["id"]), r["sample"])) == "stop"]
    n_stop = len(stop_rows)
    stop_nonanswer_k = sum(1 for _, m in stop_rows if m in NONANSWER)
    stop_only_pct = 100.0 * stop_nonanswer_k / n_stop if n_stop else None
    has_finish_reason = any(finish_reason.get((r["subject"], r["battery"], str(r["id"]), r["sample"])) is not None for r in rr)

    # cluster bootstrap over prompt ids (each resampled id carries ALL its samples)
    by_id = defaultdict(list)
    for r, m in parsed:
        by_id[r["id"]].append(m)
    ids_list = list(by_id.keys())
    ci_lo = ci_hi = None
    if ids_list and n_parsed:
        rng = random.Random(BOOT_SEED)
        vals = []
        for _ in range(N_BOOT):
            samp_ids = [rng.choice(ids_list) for _ in ids_list]
            k = tot = 0
            for i in samp_ids:
                for m in by_id[i]:
                    tot += 1
                    if m in NONANSWER:
                        k += 1
            if tot:
                vals.append(100.0 * k / tot)
        if vals:
            vals.sort()
            ci_lo = vals[int(0.025 * len(vals))]
            ci_hi = vals[int(0.975 * len(vals)) - 1]

    return {
        "n": n_total, "n_parsed": n_parsed, "parse_pct": parse_pct,
        "mode_shares": mode_shares, "nonanswer_pct": nonanswer_pct,
        "nonanswer_ci_lo": ci_lo, "nonanswer_ci_hi": ci_hi,
        "flavor_shares": flavor_shares, "n_nonanswer": nonanswer_k,
        "stop_only_nonanswer_pct": stop_only_pct, "n_stop": n_stop,
        "has_finish_reason": has_finish_reason,
    }


def strict_mode_crosstab(subj, battery, ids=None):
    """2x3 strict(yes/no) x mode(answer/deflect/refuse) cross-tab, using stored strict verdicts."""
    if battery == "GH":
        # generic: gcrm1_mini_task1.jsonl keyed by (model,id) -- load once, cached below.
        strict_lookup = STRICT_GENERIC
        rr = rows_for(subj, "GH", ids)
        cross = {f"{sy}_{m}": 0 for sy in ("strict_yes", "strict_no") for m in MODES}
        n_joint = 0
        for r in rr:
            sv = strict_lookup.get((subj, r["id"]))
            m = mode_of(r["subject"], r["battery"], r["id"], r["sample"])
            if sv is None or sv not in ("yes", "no") or m is None:
                continue
            cross[f"strict_{sv}_{m}"] += 1
            n_joint += 1
        return cross, n_joint
    else:
        strict_bs = a0["per_subject_battery_strict"].get(subj, {}).get(battery)
        # per-row strict verdicts needed for true cross-tab; reload from strict files directly
        cross = {f"{sy}_{m}": 0 for sy in ("strict_yes", "strict_no") for m in MODES}
        n_joint = 0
        rr = rows_for(subj, battery, ids)
        for r in rr:
            sv = STRICT_POLITICAL.get((r["subject"], r["battery"], str(r["id"]), r["sample"]))
            m = mode_of(r["subject"], r["battery"], r["id"], r["sample"])
            if sv is None or sv not in ("yes", "no") or m is None:
                continue
            cross[f"strict_{sv}_{m}"] += 1
            n_joint += 1
        return cross, n_joint


print("loading strict verdict lookups for cross-tab...")
STRICT_GENERIC = {}
for r in load_jsonl(WORK / "judged_strict" / "gcrm1_mini_task1.jsonl"):
    STRICT_GENERIC[(r["model"], r["id"])] = r.get("refusal")

STRICT_POLITICAL = {}
STRICT_FILES = ["cas1_mini_task1.jsonl", "dl1_gpt55_task1.jsonl", "sb1_gpt55_task1.jsonl", "pilot_mini_task1.jsonl"]
for fname in STRICT_FILES:
    for r in load_jsonl(WORK / "judged_strict" / fname):
        k = (canonicalize(r["model"]), r.get("battery"), str(r["id"]), r.get("sample", 0))
        STRICT_POLITICAL[k] = r.get("refusal")


def pooled_nonanswer_kn(subj_list, battery, ids=None):
    k = n = 0
    for s in subj_list:
        st = compute_stats(s, battery, ids)
        if st["n_parsed"]:
            n += st["n_parsed"]
            k += st["n_nonanswer"]
    return k, n


print("computing per subject x battery stats (this can take a couple minutes)...")
long_rows = []
map_points_rows = []
flavor_pool = defaultdict(lambda: defaultdict(int))  # (group,battery) -> flavor -> count
flavor_pool_nonanswer_n = defaultdict(int)

floor_seeds = [s["subject"] for s in subjects_config if s["group"] == "floor"]
STOCK_SUBJ = "stock"

for subj in SUBJECTS:
    meta = subj_meta[subj]
    group = meta["group"]
    trained_ids = set(trained_on.get(subj, []))
    held_out_ids = [i for i in generic_battery_ids if i not in trained_ids]
    n_heldout = len(held_out_ids)
    n_trained_on = len(generic_battery_ids) - n_heldout

    row_common = {
        "subject": subj, "group": group, "seed": meta.get("seed"), "dose": meta.get("dose"),
    }

    subj_judge_prov = a0["subject_judge_provenance"].get(subj, {})
    strict_judge = subj_judge_prov.get("judge")
    source_exp = subj_judge_prov.get("source_exp")
    serving_vintage = SERVING_VINTAGE.get(source_exp)

    map_row = dict(row_common)
    map_row["source_exp"] = source_exp
    map_row["serving_vintage_political"] = serving_vintage
    map_row["serving_vintage_generic"] = SERVING_VINTAGE["csp1-generic-refusal-map-1"]

    for battery in ("B0", "B1", "B2", "GH"):
        ids = held_out_ids if battery == "GH" else None
        st = compute_stats(subj, battery, ids)
        cross, n_cross = strict_mode_crosstab(subj, battery, ids)

        if battery == "GH":
            strict_pct = gcrm1_subjects.get(subj, {}).get("gh_heldout_pct")
            strict_pct = float(strict_pct) if strict_pct not in (None, "", "None") else None
            strict_n = gcrm1_subjects.get(subj, {}).get("n_heldout")
        else:
            sb = a0["per_subject_battery_strict"].get(subj, {}).get(battery, {})
            strict_pct = sb.get("strict_pct")
            strict_n = sb.get("n_joined")

        rec = dict(row_common)
        rec.update({
            "battery": battery, "n": st["n"], "n_parsed": st["n_parsed"], "parse_pct": round(st["parse_pct"], 3) if st["parse_pct"] is not None else None,
            "answer_share": round(st["mode_shares"]["answer"], 4) if st["mode_shares"]["answer"] is not None else None,
            "deflect_share": round(st["mode_shares"]["deflect"], 4) if st["mode_shares"]["deflect"] is not None else None,
            "refuse_share": round(st["mode_shares"]["refuse"], 4) if st["mode_shares"]["refuse"] is not None else None,
            "nonanswer_pct": round(st["nonanswer_pct"], 4) if st["nonanswer_pct"] is not None else None,
            "nonanswer_ci_lo": round(st["nonanswer_ci_lo"], 4) if st["nonanswer_ci_lo"] is not None else None,
            "nonanswer_ci_hi": round(st["nonanswer_ci_hi"], 4) if st["nonanswer_ci_hi"] is not None else None,
            "flavor_ccp_deflection_share": round(st["flavor_shares"]["ccp-deflection"], 4) if st["flavor_shares"]["ccp-deflection"] is not None else None,
            "flavor_generic_safety_share": round(st["flavor_shares"]["generic-safety"], 4) if st["flavor_shares"]["generic-safety"] is not None else None,
            "flavor_capability_excuse_share": round(st["flavor_shares"]["capability-excuse"], 4) if st["flavor_shares"]["capability-excuse"] is not None else None,
            "flavor_other_share": round(st["flavor_shares"]["other"], 4) if st["flavor_shares"]["other"] is not None else None,
            "stop_only_nonanswer_pct": round(st["stop_only_nonanswer_pct"], 4) if st["stop_only_nonanswer_pct"] is not None else None,
            "n_stop": st["n_stop"], "has_finish_reason": st["has_finish_reason"],
            "strict_pct": round(strict_pct, 4) if strict_pct is not None else None,
            "strict_judge": strict_judge if battery != "GH" else "gpt-5.4-mini-2026-03-17",
            "strict_n": strict_n,
            "source_exp": source_exp if battery != "GH" else "csp1-generic-refusal-map-1",
            "serving_vintage": serving_vintage if battery != "GH" else SERVING_VINTAGE["csp1-generic-refusal-map-1"],
            "crosstab_strict_yes_answer": cross["strict_yes_answer"], "crosstab_strict_yes_deflect": cross["strict_yes_deflect"],
            "crosstab_strict_yes_refuse": cross["strict_yes_refuse"], "crosstab_strict_no_answer": cross["strict_no_answer"],
            "crosstab_strict_no_deflect": cross["strict_no_deflect"], "crosstab_strict_no_refuse": cross["strict_no_refuse"],
            "crosstab_n": n_cross,
        })
        if battery == "GH":
            rec["n_heldout"] = n_heldout
            rec["n_trained_on"] = n_trained_on
            gh_full = compute_stats(subj, "GH", None)
            rec["gh_full_nonanswer_pct"] = round(gh_full["nonanswer_pct"], 4) if gh_full["nonanswer_pct"] is not None else None
            kf, nf = pooled_nonanswer_kn(floor_seeds, "GH", held_out_ids)
            ks, ns = pooled_nonanswer_kn([STOCK_SUBJ], "GH", held_out_ids)
            rec["floor_gh_on_mask_pct"] = round(100.0 * kf / nf, 4) if nf else None
            rec["stock_gh_on_mask_pct"] = round(100.0 * ks / ns, 4) if ns else None
        long_rows.append(rec)

        # map_points_task2 axis columns
        axis_key = {"B0": "b0", "B1": "b1", "B2": "b2", "GH": "gh_heldout"}[battery]
        map_row[f"{axis_key}_nonanswer_pct"] = rec["nonanswer_pct"]
        map_row[f"{axis_key}_ci_lo"] = rec["nonanswer_ci_lo"]
        map_row[f"{axis_key}_ci_hi"] = rec["nonanswer_ci_hi"]
        map_row[f"{axis_key}_n"] = rec["n_parsed"]
        map_row[f"{axis_key}_strict_pct"] = rec["strict_pct"]
        map_row[f"{axis_key}_strict_judge"] = rec["strict_judge"]

        # flavor pool (row-weighted, pooled by subject GROUP across its subjects)
        rr = rows_for(subj, battery, ids)
        for r in rr:
            m = mode_of(r["subject"], r["battery"], r["id"], r["sample"])
            if m in NONANSWER:
                fl = flavor_of(r["subject"], r["battery"], r["id"], r["sample"])
                if fl in FLAVORS:
                    flavor_pool[(group, battery)][fl] += 1
                flavor_pool_nonanswer_n[(group, battery)] += 1

    map_points_rows.append(map_row)

print(f"computed {len(long_rows)} long-form rows, {len(map_points_rows)} map points")

# ---- write subjects_task2.csv ----
long_fields = ["subject", "group", "seed", "dose", "battery", "n", "n_parsed", "parse_pct",
               "answer_share", "deflect_share", "refuse_share",
               "nonanswer_pct", "nonanswer_ci_lo", "nonanswer_ci_hi",
               "flavor_ccp_deflection_share", "flavor_generic_safety_share",
               "flavor_capability_excuse_share", "flavor_other_share",
               "stop_only_nonanswer_pct", "n_stop", "has_finish_reason",
               "strict_pct", "strict_judge", "strict_n", "source_exp", "serving_vintage",
               "crosstab_strict_yes_answer", "crosstab_strict_yes_deflect", "crosstab_strict_yes_refuse",
               "crosstab_strict_no_answer", "crosstab_strict_no_deflect", "crosstab_strict_no_refuse",
               "crosstab_n", "n_heldout", "n_trained_on", "gh_full_nonanswer_pct",
               "floor_gh_on_mask_pct", "stock_gh_on_mask_pct"]
with open(FIGDIR / "subjects_task2.csv", "w", newline="") as f:
    f.write("# csp1-task2-rejudge-1 full long-form table: one row per subject x battery "
            "(B0/B1/B2/GH). Task-2 columns (mode shares, nonanswer_pct+CI, flavor shares, "
            "stop_only_nonanswer_pct) from THIS wave's mini judge pass (judged/p1_generic_mini.jsonl, "
            "judged/p2_political_mini.jsonl), cluster-bootstrap CI (2000 resamples, seed=17, "
            "resample prompt ids with all their samples). strict_pct/strict_judge/source_exp/"
            "serving_vintage = stored strict Task-1 verdicts (A0 join, no re-judging; GH from "
            "gcrm1's committed subjects.csv). crosstab_* = strict(yes/no) x Task-2 mode 2x3 cells. "
            "GH rows only: n_heldout/n_trained_on (gcrm1 committed masks), gh_full_nonanswer_pct "
            "(common-denominator sensitivity, includes trained-on rows), floor/stock_gh_on_mask_pct "
            "(pooled floor seeds / stock restricted to this subject's own held-out mask, Task-2).\n")
    w = csv.DictWriter(f, fieldnames=long_fields)
    w.writeheader()
    for r in long_rows:
        w.writerow(r)
print(f"wrote {len(long_rows)} rows -> {FIGDIR}/subjects_task2.csv")

# ---- write map_points_task2.csv ----
map_fields = ["subject", "group", "seed", "dose", "source_exp", "serving_vintage_political", "serving_vintage_generic"]
for axis in ("b0", "b1", "b2", "gh_heldout"):
    map_fields += [f"{axis}_nonanswer_pct", f"{axis}_ci_lo", f"{axis}_ci_hi", f"{axis}_n",
                   f"{axis}_strict_pct", f"{axis}_strict_judge"]
with open(FIGDIR / "map_points_task2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=map_fields)
    w.writeheader()
    for r in map_points_rows:
        w.writerow(r)
print(f"wrote {len(map_points_rows)} rows -> {FIGDIR}/map_points_task2.csv")

# ---- write flavor_group_bars.csv ----
flavor_rows = []
for (group, battery), counts in sorted(flavor_pool.items()):
    n_na = flavor_pool_nonanswer_n[(group, battery)]
    for fl in FLAVORS:
        c = counts.get(fl, 0)
        flavor_rows.append({
            "group": group, "battery": battery, "flavor": fl,
            "count": c, "n_nonanswer": n_na,
            "share": round(c / n_na, 4) if n_na else None,
        })
with open(FIGDIR / "flavor_group_bars.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["group", "battery", "flavor", "count", "n_nonanswer", "share"])
    w.writeheader()
    for r in flavor_rows:
        w.writerow(r)
print(f"wrote {len(flavor_rows)} rows -> {FIGDIR}/flavor_group_bars.csv")

print("\n=== sanity ===")
for subj in ("stock",) + tuple(floor_seeds):
    b0 = next(r for r in long_rows if r["subject"] == subj and r["battery"] == "B0")
    gh = next(r for r in long_rows if r["subject"] == subj and r["battery"] == "GH")
    print(f"{subj:20s} B0 nonanswer={b0['nonanswer_pct']}% (n={b0['n_parsed']}) "
          f"GH heldout nonanswer={gh['nonanswer_pct']}% (n={gh['n_parsed']})")
