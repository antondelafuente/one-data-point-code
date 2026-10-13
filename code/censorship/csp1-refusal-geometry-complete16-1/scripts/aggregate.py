#!/usr/bin/env python3
"""RGC16 aggregation: raw rollouts + Task-2 judgments -> the tables DESIGN § Presentation locks.

LOCAL and self-contained (automated-researcher#458): every input is this run's own work-dir
artifact or a committed frozen input of this record. Nothing is read from a sibling worktree.

Adapted from `csp1-refusal-geometry-tinydose-1/scripts/aggregate.py` @ fc21ef63. Structural deltas:
  * ONE serving session, so every primary cell is a same-session same-seed delta by construction —
    there is no wave dimension and no cross-session pooling anywhere in the headline matrix;
  * the headline object is the COMPLETE 8x8 training-category x eval-category matrix at K=16, in
    three intensity views (combined = all 40 aligned prompt clusters at equal prompt weight, plus
    strong and moderate separately) and three seed views (seed 3, seed 4, equal-seed mean);
  * no dose axis, no probe gate, no d40 point, no fitted models — those are not inherited;
  * directional contrasts (A trained -> B evaluated versus B trained -> A evaluated) are first-class
    output for every off-diagonal pair, with the DESIGN-named opt-in pairs flagged;
  * GSM8K is split into FRESH (this run's eight new subjects) and CITED (the eight re-served
    adapters, from RGTD1 @ fc21ef63) and never pooled.

Outputs (figures/):
  cells.csv                 every subject x probe x intensity cell: attempted/generated/failed/
                            parsed, answer/deflect/refuse + rates, total non-answer, finish
                            reasons, empty and token-capped counts, flavour counts, provenance.
  matrix.csv                the 8x8 matrix rows: training category x eval category x intensity,
                            per-seed raw arm and same-seed filler rates, per-seed lift, equal-seed
                            mean lift, 95% question-cluster bootstrap CI, non-answer companion,
                            stock context, denominators, session provenance, reverse-direction
                            lift for the transposed cell.
  question_scores.jsonl     row-level scores (one line per judged rollout).
  question_vectors.json     per-(subject, probe, intensity) question-level vectors, stored BEFORE
                            bootstrapping.
  directional_contrasts.csv every unordered off-diagonal category pair, both directions, their
                            difference and its bootstrap CI; DESIGN-named pairs flagged.
  group_contrasts.csv       newly measured versus previously measured category groups.
  controls.csv              the five side batteries per subject, with the same-seed filler
                            reference and the flavour split.
  seed_stability.csv        per-arm seed-3 vs seed-4 lift vectors and their agreement.
  gsm8k.csv                 fresh (this run) and cited (RGTD1 @ fc21ef63) health readings, labelled.
  token_mass.csv            per-arm append-block completion-token mass (companion diagnostic).
  floor_ceiling.json        cells at 0/100 % (and <2 / >98 %) + the empirical resolution.
No thresholds, no verdicts.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from rgc16_common import (ANCHOR_SUBJECTS, BOOT_N, BOOT_SEED, CATEGORIES, CATEGORY_LABEL,
                          CONTROL_ARM, CORE_PROBES, DOSE, FILLER_SUBJECTS, GSM8K_MIN_PCT,
                          NEW_CATEGORIES, NONANSWER_PRIMARY_PROBES, PARENT_COMMIT,
                          PARENT_EXPERIMENT, REG, REUSED_CATEGORIES, SEEDS, SIDE_PROBES, STAGE,
                          STOCK, WORK, load_eval_inputs, load_json, load_latest, load_subjects,
                          split_arm, subject_probes)

FIG = WORK / "figures"
INTENS = ["combined", "strong", "moderate"]     # combined is the DESIGN default view

# DESIGN § Comparability: the named opt-in comparisons.
NAMED_PAIRS = {frozenset(("institutions", "drugs")), frozenset(("gross", "adult_sexual")),
               frozenset(("morbid", "adult_sexual"))}


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    stage = STAGE
    subjects = load_subjects(stage)
    by_name = {s["subject"]: s for s in subjects}
    evals = load_eval_inputs()
    meta = {(r["probe"], str(r["prompt_id"])): r for r in evals}
    expected = defaultdict(int)
    for r in evals:
        i = (r.get("meta") or {}).get("intensity", "all")
        expected[(r["probe"], i)] += r["samples"]
        if r["probe"] in CORE_PROBES:
            expected[(r["probe"], "combined")] += r["samples"]

    judged_path = WORK / stage / "judged" / "task2_final.jsonl"
    judged = {}
    for line in judged_path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("mode") is None:
            continue
        judged[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r

    # ---------- row-level scores + per-cell aggregation ----------
    cells = {}
    per_prompt = defaultdict(lambda: defaultdict(list))   # (subject,probe) -> pid -> [(refuse, nonanswer)]
    qs_path = FIG / "question_scores.jsonl"
    n_scores = 0
    with qs_path.open("w") as qfh:
        for s in subjects:
            subj = s["subject"]
            for probe in subject_probes(s, stage):
                gen = load_latest(stage, subj, probe)
                for k in sorted(gen):
                    j = judged.get((subj, probe, k[0], k[1]))
                    g = gen[k]
                    inten = (meta[(probe, k[0])].get("meta") or {}).get("intensity") or "all"
                    qfh.write(json.dumps({
                        "session": stage, "subject": subj, "arm": s["arm"],
                        "training_category": s["training_category"], "training_seed": s["seed"],
                        "reserved": s.get("reserved", False), "membership": s.get("membership"),
                        "sampler_path": s["sampler_path"], "probe": probe,
                        "eval_category": probe.replace("core_", "") if probe in CORE_PROBES else None,
                        "intensity": inten, "prompt_id": k[0], "sample": k[1],
                        "raw_output_shard": str(Path(*("gen", subj, f"{probe}.jsonl"))),
                        "n_tokens": g.get("n_tokens"), "finish_reason": g.get("finish_reason"),
                        "capped": bool((g.get("n_tokens") or 0) >= 1024),
                        "empty_response": not (g.get("response") or "").strip(),
                        "parsed": j is not None,
                        "mode": None if j is None else j["mode"],
                        "flavor": None if j is None else j["flavor"],
                        "parser_path": None if j is None else j.get("parser_path"),
                        "judge_request_id": None if j is None else j.get("request_id"),
                    }, ensure_ascii=False) + "\n")
                    n_scores += 1
                intens = INTENS if probe in CORE_PROBES else ["all"]
                for inten in intens:
                    keys = [k for k in gen if inten in ("combined", "all")
                            or (meta[(probe, k[0])].get("meta") or {}).get("intensity") == inten]
                    jr = [judged[(subj, probe, k[0], k[1])] for k in keys
                          if (subj, probe, k[0], k[1]) in judged]
                    modes = Counter(r["mode"] for r in jr)
                    flav = Counter(f"{r['mode']}|{r['flavor']}" for r in jr if r["mode"] != "answer")
                    fin = Counter(str(gen[k].get("finish_reason")) for k in keys)
                    capped = sum(1 for k in keys if (gen[k].get("n_tokens") or 0) >= 1024)
                    empty = sum(1 for k in keys if not (gen[k].get("response") or "").strip())
                    n = len(jr)
                    exp = expected[(probe, inten)]
                    cells[(subj, probe, inten)] = {
                        "session": stage, "subject": subj, "arm": s["arm"], "seed": s["seed"],
                        "training_category": s["training_category"],
                        "reserved": s.get("reserved", False), "membership": s.get("membership"),
                        "sampler_path": s["sampler_path"], "probe": probe, "intensity": inten,
                        "attempted": exp, "generated": len(keys), "failed": exp - len(keys),
                        "parsed": n,
                        "coverage_generated": round(len(keys) / exp, 6) if exp else None,
                        "coverage_parsed": round(n / exp, 6) if exp else None,
                        "answer": modes["answer"], "deflect": modes["deflect"], "refuse": modes["refuse"],
                        "explicit_refusal_pct": round(modes["refuse"] / n * 100, 4) if n else None,
                        "total_nonanswer_pct": round((modes["refuse"] + modes["deflect"]) / n * 100, 4) if n else None,
                        "finish_stop": fin.get("stop", 0), "finish_length": fin.get("length", 0),
                        "finish_other": sum(v for k2, v in fin.items() if k2 not in ("stop", "length")),
                        "n_token_capped": capped, "n_empty_response": empty,
                        "flavors": json.dumps(dict(flav), sort_keys=True),
                    }
                    if inten != "combined":
                        for r in jr:
                            per_prompt[(subj, probe)][str(r["prompt_id"])].append(
                                (r["mode"] == "refuse", r["mode"] != "answer"))
    cols = list(next(iter(cells.values())).keys())
    with (FIG / "cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in sorted(cells):
            w.writerow(cells[k])

    def rate(subj, probe, inten, metric="refuse"):
        c = cells.get((subj, probe, inten))
        if not c or not c["parsed"]:
            return None
        return c["explicit_refusal_pct"] if metric == "refuse" else c["total_nonanswer_pct"]

    # ---------- question-level vectors, stored BEFORE bootstrapping ----------
    def prompt_ids(probe, inten):
        return sorted(str(r["prompt_id"]) for r in evals if r["probe"] == probe and
                      (inten in ("combined", "all") or (r.get("meta") or {}).get("intensity") == inten))

    qvec = {}
    for (subj, probe), pp in per_prompt.items():
        for inten in (INTENS if probe in CORE_PROBES else ["all"]):
            ids = prompt_ids(probe, inten)
            v = {pid: {"n": len(pp.get(pid, [])),
                       "refuse_rate": (sum(1 for a, _ in pp[pid] if a) / len(pp[pid])) if pp.get(pid) else None,
                       "nonanswer_rate": (sum(1 for _, b in pp[pid] if b) / len(pp[pid])) if pp.get(pid) else None}
                 for pid in ids}
            qvec[f"{subj}|{probe}|{inten}"] = v
    (FIG / "question_vectors.json").write_text(json.dumps(
        {"note": "per-question refusal/non-answer rates over the retained samples, stored before "
                 "any bootstrap resampling; the bootstrap resamples these question IDs",
         "session": stage, "vectors": qvec}, indent=1, sort_keys=True) + "\n")

    # ---------- bootstrap draws: independent per (probe, intensity) ----------
    rng = random.Random(BOOT_SEED)
    boot_draws = {}
    for probe in CORE_PROBES + SIDE_PROBES:
        for inten in (INTENS if probe in CORE_PROBES else ["all"]):
            ids = prompt_ids(probe, inten)
            boot_draws[(probe, inten)] = [[ids[rng.randrange(len(ids))] for _ in ids]
                                          for _ in range(BOOT_N)]

    def resampled_rate(subj, probe, ids, idx):
        pp = per_prompt[(subj, probe)]
        num = den = 0
        for pid in ids:
            for is_ref, is_na in pp.get(pid, []):
                num += (is_ref if idx == 0 else is_na)
                den += 1
        return num / den * 100 if den else None

    def lift_draws(pairs, probe, inten, metric="refuse"):
        """Equal-seed-mean paired lift per bootstrap draw. pairs = [(arm_subject, filler_subject)]."""
        idx = 0 if metric == "refuse" else 1
        vals = []
        for ids in boot_draws[(probe, inten)]:
            ds = []
            for a, f in pairs:
                ra = resampled_rate(a, probe, ids, idx)
                rf = resampled_rate(f, probe, ids, idx)
                if ra is None or rf is None:
                    ds = None
                    break
                ds.append(ra - rf)
            vals.append(sum(ds) / len(ds) if ds else None)
        return vals

    def ci(vals):
        v = sorted(x for x in vals if x is not None)
        if not v:
            return None, None
        return round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v)) - 1], 4)

    def filler_for(seed):
        name = f"{CONTROL_ARM}_s{seed}"
        return name if name in by_name else None

    # ---------- the 8x8 matrix ----------
    rows = []
    draws_cache = {}
    for train_cat in CATEGORIES:
        arm = f"{train_cat}_d{DOSE}"
        seed_ctx = [(s, f"{arm}_s{s}", filler_for(s)) for s in SEEDS
                    if f"{arm}_s{s}" in by_name and filler_for(s)]
        for eval_cat in CATEGORIES:
            probe = f"core_{eval_cat}"
            for inten in INTENS:
                per_seed = []
                for s, subj, f in seed_ctx:
                    ra, rf = rate(subj, probe, inten), rate(f, probe, inten)
                    if ra is None or rf is None:
                        continue
                    per_seed.append({"seed": s, "arm_subject": subj, "arm_pct": ra, "filler": f,
                                     "filler_pct": rf, "lift_pp": round(ra - rf, 4),
                                     "arm_na_pct": rate(subj, probe, inten, "nonanswer"),
                                     "filler_na_pct": rate(f, probe, inten, "nonanswer"),
                                     "arm_parsed": cells[(subj, probe, inten)]["parsed"],
                                     "filler_parsed": cells[(f, probe, inten)]["parsed"]})
                if not per_seed:
                    continue
                pairs = [(x["arm_subject"], x["filler"]) for x in per_seed]
                dr = lift_draws(pairs, probe, inten)
                draws_cache[(train_cat, eval_cat, inten)] = dr
                lo, hi = ci(dr)
                nlo, nhi = ci(lift_draws(pairs, probe, inten, "nonanswer"))
                d = {"training_category": train_cat, "training_label": CATEGORY_LABEL[train_cat],
                     "eval_category": eval_cat, "eval_label": CATEGORY_LABEL[eval_cat],
                     "probe": probe, "intensity": inten, "dose": DOSE, "session": stage,
                     "own_category": train_cat == eval_cat,
                     "training_membership": ("newly_measured" if train_cat in NEW_CATEGORIES
                                             else "previously_measured"),
                     "n_seeds": len(per_seed), "seeds": json.dumps([x["seed"] for x in per_seed]),
                     "lift_pp": round(sum(x["lift_pp"] for x in per_seed) / len(per_seed), 4),
                     "lift_ci_lo": lo, "lift_ci_hi": hi,
                     "arm_mean_refusal_pct": round(sum(x["arm_pct"] for x in per_seed) / len(per_seed), 4),
                     "filler_mean_refusal_pct": round(sum(x["filler_pct"] for x in per_seed) / len(per_seed), 4),
                     "nonanswer_lift_pp": (round(sum(x["arm_na_pct"] - x["filler_na_pct"] for x in per_seed) / len(per_seed), 4)
                                           if all(x["arm_na_pct"] is not None and x["filler_na_pct"] is not None for x in per_seed) else None),
                     "nonanswer_lift_ci_lo": nlo, "nonanswer_lift_ci_hi": nhi,
                     "stock_refusal_pct": rate(STOCK, probe, inten),
                     "n_prompts": len(prompt_ids(probe, inten)),
                     "parsed_per_seed": json.dumps({x["seed"]: x["arm_parsed"] for x in per_seed}),
                     "filler_parsed_per_seed": json.dumps({x["seed"]: x["filler_parsed"] for x in per_seed}),
                     "delta_provenance": "same-scale: arm and its same-seed matched filler measured "
                                         f"in the same serving session ({stage}) on the pinned stack",
                     "per_seed": json.dumps(per_seed)}
                for s in SEEDS:
                    x = next((y for y in per_seed if y["seed"] == s), None)
                    d[f"seed{s}_refusal_pct"] = x["arm_pct"] if x else None
                    d[f"seed{s}_filler_pct"] = x["filler_pct"] if x else None
                    d[f"seed{s}_lift_pp"] = x["lift_pp"] if x else None
                rows.append(d)
    # reverse-direction lift for each cell (the transposed cell of the same matrix view)
    by_key = {(r["training_category"], r["eval_category"], r["intensity"]): r for r in rows}
    for r in rows:
        rev = by_key.get((r["eval_category"], r["training_category"], r["intensity"]))
        r["reverse_lift_pp"] = None if rev is None or r["own_category"] else rev["lift_pp"]
        r["reverse_ci_lo"] = None if rev is None or r["own_category"] else rev["lift_ci_lo"]
        r["reverse_ci_hi"] = None if rev is None or r["own_category"] else rev["lift_ci_hi"]
    with (FIG / "matrix.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- directional contrasts ----------
    dc = []
    for a, b in combinations(CATEGORIES, 2):
        for inten in INTENS:
            fwd, rev = by_key.get((a, b, inten)), by_key.get((b, a, inten))
            if fwd is None or rev is None:
                continue
            da, db = draws_cache[(a, b, inten)], draws_cache[(b, a, inten)]
            diff = [x - y for x, y in zip(da, db) if x is not None and y is not None]
            lo, hi = ci(diff)
            dc.append({"category_a": a, "category_b": b, "intensity": inten, "session": stage,
                       "a_trained_b_evaluated_pp": fwd["lift_pp"],
                       "a_to_b_ci_lo": fwd["lift_ci_lo"], "a_to_b_ci_hi": fwd["lift_ci_hi"],
                       "b_trained_a_evaluated_pp": rev["lift_pp"],
                       "b_to_a_ci_lo": rev["lift_ci_lo"], "b_to_a_ci_hi": rev["lift_ci_hi"],
                       "directional_difference_pp": round(fwd["lift_pp"] - rev["lift_pp"], 4),
                       "difference_ci_lo": lo, "difference_ci_hi": hi,
                       "design_named_pair": frozenset((a, b)) in NAMED_PAIRS,
                       "membership_a": "newly_measured" if a in NEW_CATEGORIES else "previously_measured",
                       "membership_b": "newly_measured" if b in NEW_CATEGORIES else "previously_measured",
                       "resampling": "independent question-cluster draws per evaluation category "
                                     f"(B={BOOT_N}, seed={BOOT_SEED})"})
    with (FIG / "directional_contrasts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dc[0].keys()))
        w.writeheader()
        w.writerows(dc)

    # ---------- newly measured vs previously measured groups ----------
    gc = []
    for inten in INTENS:
        for scope, keep in (("all_cells", lambda r: True),
                            ("diagonal_only", lambda r: r["own_category"]),
                            ("off_diagonal_only", lambda r: not r["own_category"])):
            sel = [r for r in rows if r["intensity"] == inten and keep(r)]
            grp = {}
            for m in ("newly_measured", "previously_measured"):
                v = [r["lift_pp"] for r in sel if r["training_membership"] == m]
                grp[m] = {"n_cells": len(v),
                          "mean_lift_pp": round(float(np.mean(v)), 4) if v else None,
                          "median_lift_pp": round(float(np.median(v)), 4) if v else None}
            gc.append({"intensity": inten, "scope": scope, "session": stage,
                       "new_n_cells": grp["newly_measured"]["n_cells"],
                       "new_mean_lift_pp": grp["newly_measured"]["mean_lift_pp"],
                       "new_median_lift_pp": grp["newly_measured"]["median_lift_pp"],
                       "old_n_cells": grp["previously_measured"]["n_cells"],
                       "old_mean_lift_pp": grp["previously_measured"]["mean_lift_pp"],
                       "old_median_lift_pp": grp["previously_measured"]["median_lift_pp"],
                       "new_minus_old_pp": (round(grp["newly_measured"]["mean_lift_pp"]
                                                  - grp["previously_measured"]["mean_lift_pp"], 4)
                                            if grp["newly_measured"]["mean_lift_pp"] is not None
                                            and grp["previously_measured"]["mean_lift_pp"] is not None else None),
                       "note": "group membership covaries with adapter training date/history "
                               "(DESIGN § Comparability): a group difference is not attributable to "
                               "category semantics alone"})
    with (FIG / "group_contrasts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gc[0].keys()))
        w.writeheader()
        w.writerows(gc)

    # ---------- side-battery controls ----------
    ctrl = []
    for s in subjects:
        subj = s["subject"]
        for probe in SIDE_PROBES:
            c = cells.get((subj, probe, "all"))
            if not c:
                continue
            metric = "nonanswer" if probe in NONANSWER_PRIMARY_PROBES else "refuse"
            prim = c["total_nonanswer_pct"] if probe in NONANSWER_PRIMARY_PROBES else c["explicit_refusal_pct"]
            fm = None
            if s["arm"] != CONTROL_ARM and s["seed"]:
                f = filler_for(s["seed"])
                fm = rate(f, probe, "all", metric) if f else None
            flav = json.loads(c["flavors"])
            na = c["refuse"] + c["deflect"]
            ctrl.append({"session": stage, "subject": subj, "arm": s["arm"], "seed": s["seed"],
                         "membership": s.get("membership"), "probe": probe,
                         "primary_reading": "total_nonanswer" if probe in NONANSWER_PRIMARY_PROBES else "explicit_refusal",
                         "battery_label": ("frozen harmful-boundary battery (mixes harm assistance, "
                                           "sexual-content and copyright refusals; not a pure "
                                           "harm-prevention score)" if probe == "harmful_boundary" else probe),
                         "primary_pct": prim, "explicit_refusal_pct": c["explicit_refusal_pct"],
                         "total_nonanswer_pct": c["total_nonanswer_pct"], "parsed": c["parsed"],
                         "attempted": c["attempted"],
                         "filler_same_seed_primary_pct": round(fm, 4) if fm is not None else None,
                         "lift_vs_filler_pp": round(prim - fm, 4) if (fm is not None and prim is not None) else None,
                         "ccp_deflection_share_of_nonanswers": round(sum(v for k2, v in flav.items() if k2.endswith("|ccp-deflection")) / na, 4) if na else None,
                         "n_token_capped": c["n_token_capped"], "n_empty_response": c["n_empty_response"]})
    with (FIG / "controls.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ctrl[0].keys()))
        w.writeheader()
        w.writerows(ctrl)

    # ---------- seed stability ----------
    stab = []
    for cat in CATEGORIES:
        for inten in INTENS:
            v = [r for r in rows if r["training_category"] == cat and r["intensity"] == inten]
            v.sort(key=lambda r: CATEGORIES.index(r["eval_category"]))
            a3 = [r["seed3_lift_pp"] for r in v]
            a4 = [r["seed4_lift_pp"] for r in v]
            if not v or any(x is None for x in a3 + a4):
                continue
            stab.append({"training_category": cat, "dose": DOSE, "intensity": inten,
                         "session": stage, "n_cells": len(v),
                         "membership": "newly_measured" if cat in NEW_CATEGORIES else "previously_measured",
                         "raw_lift_seed_corr": round(pearson(a3, a4), 4) if pearson(a3, a4) is not None else None,
                         "seed3_lift_vector": json.dumps(a3), "seed4_lift_vector": json.dumps(a4),
                         "eval_order": json.dumps([r["eval_category"] for r in v]),
                         "mean_abs_seed_diff_pp": round(float(np.mean(np.abs(np.array(a3) - np.array(a4)))), 4)})
    with (FIG / "seed_stability.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stab[0].keys()))
        w.writeheader()
        w.writerows(stab)

    # ---------- token mass (companion diagnostic) ----------
    tm_path = WORK / "receipts" / "token_mass.json"
    if tm_path.exists():
        tm = load_json(tm_path)
        with (FIG / "token_mass.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(tm["arms"][0].keys()))
            w.writeheader()
            w.writerows(tm["arms"])

    # ---------- GSM8K: fresh vs cited, never pooled ----------
    gsm = []
    for p in sorted((WORK / "gsm8k").glob("*.jsonl")):
        recs = [json.loads(l) for l in p.open() if l.strip()]
        latest = {r["id"]: r for r in recs}
        n = len(latest)
        acc = round(sum(1 for r in latest.values() if r["correct"]) / n * 100, 4) if n else None
        gsm.append({"subject": p.stem, "n": n, "accuracy_pct": acc, "gate_min_pct": GSM8K_MIN_PCT,
                    "pass": acc is not None and acc >= GSM8K_MIN_PCT,
                    "excluded_from_matrices": not (acc is not None and acc >= GSM8K_MIN_PCT),
                    "provenance": "fresh (this run)", "source": STAGE, "source_commit": None,
                    "session": stage})
    cited_path = REG / "design_inputs" / "cited_gsm8k.json"
    if cited_path.exists():
        for r in load_json(cited_path)["rows"]:
            gsm.append({"subject": r["subject"], "n": int(r["n"]),
                        "accuracy_pct": float(r["accuracy_pct"]), "gate_min_pct": GSM8K_MIN_PCT,
                        "pass": float(r["accuracy_pct"]) >= GSM8K_MIN_PCT,
                        "excluded_from_matrices": float(r["accuracy_pct"]) < GSM8K_MIN_PCT,
                        "provenance": "cited (re-served adapter; NOT a fresh health test)",
                        "source": PARENT_EXPERIMENT, "source_commit": PARENT_COMMIT,
                        "session": "csp1-refusal-geometry-tinydose-1 stage1/stage2"})
    gsm.sort(key=lambda r: (r["provenance"], r["subject"]))
    with (FIG / "gsm8k.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gsm[0].keys()))
        w.writeheader()
        w.writerows(gsm)

    # ---------- floor / ceiling + resolution ----------
    fc = {"floor_0pct": [], "ceiling_100pct": [], "below_2pct": [], "above_98pct": []}
    for (subj, probe, inten), c in cells.items():
        if c["explicit_refusal_pct"] is None or probe not in CORE_PROBES or inten == "combined":
            continue
        key = f"{subj}/{probe}/{inten}"
        if c["explicit_refusal_pct"] == 0:
            fc["floor_0pct"].append(key)
        elif c["explicit_refusal_pct"] < 2:
            fc["below_2pct"].append(key)
        if c["explicit_refusal_pct"] == 100:
            fc["ceiling_100pct"].append(key)
        elif c["explicit_refusal_pct"] > 98:
            fc["above_98pct"].append(key)
    hw = [(r["lift_ci_hi"] - r["lift_ci_lo"]) / 2 for r in rows
          if r["lift_ci_hi"] is not None and r["intensity"] != "combined"]
    fc["empirical_resolution"] = {
        "median_lift_ci_half_width_pp": round(float(np.median(hw)), 3) if hw else None,
        "max_lift_ci_half_width_pp": round(float(np.max(hw)), 3) if hw else None,
        "n_per_seed_cell_moderate_or_strong": 60, "n_per_seed_cell_combined": 120,
        "note": "question-cluster bootstrap on 20 prompts x 3 samples per intensity cell "
                "(40 x 3 combined); intervals are conditional on THESE models and do not carry "
                "training-seed uncertainty — the seed panels, not the seed-mean CI, show seed "
                "variation"}
    fc["n_cells_by_class"] = {k: len(v) for k, v in fc.items() if isinstance(v, list)}
    (FIG / "floor_ceiling.json").write_text(json.dumps(fc, indent=2) + "\n")

    # ---------- session/subject provenance ----------
    (FIG / "subjects.json").write_text(json.dumps(subjects, indent=2) + "\n")

    coverage_min = min((c["coverage_parsed"] for c in cells.values()
                        if c["coverage_parsed"] is not None), default=None)
    summary = {
        "experiment": "csp1-refusal-geometry-complete16-1", "session": stage,
        "n_subjects": len(subjects), "n_cells": len(cells), "n_judged_keys": len(judged),
        "n_row_scores": n_scores, "matrix_rows": len(rows),
        "matrix_shape": "8 training categories x 8 evaluation categories x 3 intensity views",
        "directional_contrast_rows": len(dc), "group_contrast_rows": len(gc),
        "control_rows": len(ctrl), "seed_stability_rows": len(stab),
        "gsm8k_fresh": sum(1 for g in gsm if g["provenance"].startswith("fresh")),
        "gsm8k_cited": sum(1 for g in gsm if g["provenance"].startswith("cited")),
        "gsm8k_below_gate": [g["subject"] for g in gsm if not g["pass"]],
        "min_cell_coverage_parsed": coverage_min,
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "cluster": "prompt",
                      "resampling": "independent between evaluation categories; both seeds share "
                                    "the draw within a cell"},
        "anchor_subjects": list(ANCHOR_SUBJECTS), "filler_subjects": list(FILLER_SUBJECTS),
        "newly_measured_categories": list(NEW_CATEGORIES),
        "previously_measured_categories": list(REUSED_CATEGORIES),
        "floor_ceiling_counts": fc["n_cells_by_class"],
        "resolution": fc["empirical_resolution"],
    }
    (FIG / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
