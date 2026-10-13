#!/usr/bin/env python3
"""RGM1 aggregation: raw rows + Task-2 judgments -> the locked tables (DESIGN.md "Measurements").

Outputs (figures/):
  subject_probe_cells.csv   every subject x probe x intensity: attempted, generated, failed, parsed,
                            answer/deflect/refuse counts + rates, total non-answer, finish reasons,
                            token-capped rows, flavor counts. Seeds visible.
  arm_probe_matrix.csv      8 category arms x 8 core probes x {moderate, strong, combined}: seed-3 and
                            seed-4 explicit-refusal rates, arm mean, matched-filler seed rates + mean,
                            PRIMARY lift = arm mean - filler mean (pp), question-cluster bootstrap 95%
                            (2,000 resamples, seed 17), total-nonanswer companion, stock context.
  controls.csv              harmful (total non-answer primary), everyday benign, near-miss, Arena,
                            legacy B0 (explicit refusal + ccp-deflection share) per subject and arm.
  arena_newly_refused.csv   per subject: prompts whose modal label is refuse while BOTH filler seeds
                            answer; arm equal-seed mean; composition by frozen prior tag.
  seed_stability.csv        per arm: Pearson r between seed-3 and seed-4 raw 8-cell lift vectors.
  floor_ceiling.json        cells at 0%/100% (and <2%/>98%) + empirical resolution (median CI half-width).
No thresholds, no verdicts.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np

from rgm1_common import (ALL_PROBES, ANCHORS, ARMS, CATEGORIES, CORE_PROBES, FILLER_SUBJECTS,
                         NONANSWER_PRIMARY_PROBES, SIDE_PROBES, STOCK, WORK, load_eval_inputs,
                         load_latest, load_subjects, stage_dir, subject_probes)

FIG = WORK / "figures"
BOOT_N, BOOT_SEED = 2000, 17
INTENS = ["moderate", "strong", "combined"]


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    subjects = load_subjects("final")
    evals = load_eval_inputs()
    meta = {(r["probe"], str(r["prompt_id"])): r for r in evals}
    expected = defaultdict(int)  # (probe, intensity) -> keys
    for r in evals:
        i = (r.get("meta") or {}).get("intensity", "all")
        expected[(r["probe"], i)] += r["samples"]
        if r["probe"] in CORE_PROBES:
            expected[(r["probe"], "combined")] += r["samples"]

    judged = {}
    for line in (stage_dir("final") / "judged" / "task2_final.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("mode") is None or r.get("stage") != "final":
            continue
        judged[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r

    # ---------- per subject x probe x intensity cells ----------
    cells = {}
    per_prompt = defaultdict(lambda: defaultdict(list))  # (subject, probe) -> prompt_id -> [is_refuse, is_nonanswer]
    for s in subjects:
        subj = s["subject"]
        for probe in subject_probes(s, "final"):
            gen = load_latest("final", subj, probe)
            intens = ["moderate", "strong", "combined"] if probe in CORE_PROBES else ["all"]
            for inten in intens:
                keys = [k for k in gen if inten in ("combined", "all") or meta[(probe, k[0])]["meta"].get("intensity") == inten]
                jr = [judged[(subj, probe, k[0], k[1])] for k in keys if (subj, probe, k[0], k[1]) in judged]
                modes = Counter(r["mode"] for r in jr)
                flav = Counter(f"{r['mode']}|{r['flavor']}" for r in jr if r["mode"] != "answer")
                fin = Counter(str(gen[k].get("finish_reason")) for k in keys)
                capped = sum(1 for k in keys if (gen[k].get("n_tokens") or 0) >= 1024)
                n = len(jr)
                exp = expected[(probe, inten)]
                cells[(subj, probe, inten)] = {
                    "subject": subj, "arm": s["arm"], "seed": s["seed"], "probe": probe, "intensity": inten,
                    "attempted": exp, "generated": len(keys), "failed": exp - len(keys), "parsed": n,
                    "coverage_generated": round(len(keys) / exp, 6) if exp else None,
                    "coverage_parsed": round(n / exp, 6) if exp else None,
                    "answer": modes["answer"], "deflect": modes["deflect"], "refuse": modes["refuse"],
                    "explicit_refusal_pct": round(modes["refuse"] / n * 100, 4) if n else None,
                    "total_nonanswer_pct": round((modes["refuse"] + modes["deflect"]) / n * 100, 4) if n else None,
                    "finish_stop": fin.get("stop", 0), "finish_length": fin.get("length", 0),
                    "finish_other": sum(v for k, v in fin.items() if k not in ("stop", "length")),
                    "n_token_capped": capped, "flavors": json.dumps(dict(flav), sort_keys=True),
                }
                if inten != "combined":
                    for r in jr:
                        per_prompt[(subj, probe)][r["prompt_id"]].append((r["mode"] == "refuse", r["mode"] != "answer"))
    cols = list(next(iter(cells.values())).keys())
    with (FIG / "subject_probe_cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in sorted(cells):
            w.writerow(cells[k])

    def rate(subj, probe, inten, metric="refuse"):
        c = cells.get((subj, probe, inten))
        if not c or not c["parsed"]:
            return None
        return c["explicit_refusal_pct"] if metric == "refuse" else c["total_nonanswer_pct"]

    # ---------- bootstrap helpers (question-cluster) ----------
    def prompt_ids(probe, inten):
        return [str(r["prompt_id"]) for r in evals if r["probe"] == probe and
                (inten in ("combined", "all") or r["meta"].get("intensity") == inten)]

    rng = random.Random(BOOT_SEED)
    boot_draws = {}  # (probe, inten) -> list of resampled prompt-id lists (shared across arms for comparability)
    for probe in CORE_PROBES + SIDE_PROBES:
        for inten in (INTENS if probe in CORE_PROBES else ["all"]):
            ids = prompt_ids(probe, inten)
            boot_draws[(probe, inten)] = [[ids[rng.randrange(len(ids))] for _ in ids] for _ in range(BOOT_N)]

    def resampled_rate(subj, probe, ids, idx):
        pp = per_prompt[(subj, probe)]
        num = den = 0
        for pid in ids:
            for is_ref, is_na in pp.get(pid, []):
                num += (is_ref if idx == 0 else is_na)
                den += 1
        return num / den * 100 if den else None

    def lift_ci(arm_subjs, probe, inten, metric="refuse"):
        idx = 0 if metric == "refuse" else 1
        vals = []
        for ids in boot_draws[(probe, inten)]:
            a = [resampled_rate(s, probe, ids, idx) for s in arm_subjs]
            f = [resampled_rate(s, probe, ids, idx) for s in FILLER_SUBJECTS]
            if any(v is None for v in a + f):
                continue
            vals.append(sum(a) / len(a) - sum(f) / len(f))
        if not vals:
            return None, None
        vals.sort()
        return round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)

    # ---------- arm x probe matrix ----------
    rows = []
    for arm in CATEGORIES:
        subs = [f"{arm}_s3", f"{arm}_s4"]
        for probe in CORE_PROBES:
            for inten in INTENS:
                r3, r4 = rate(subs[0], probe, inten), rate(subs[1], probe, inten)
                f3, f4 = rate(FILLER_SUBJECTS[0], probe, inten), rate(FILLER_SUBJECTS[1], probe, inten)
                if None in (r3, r4, f3, f4):
                    continue
                am, fm = (r3 + r4) / 2, (f3 + f4) / 2
                lo, hi = lift_ci(subs, probe, inten)
                n3, n4 = rate(subs[0], probe, inten, "nonanswer"), rate(subs[1], probe, inten, "nonanswer")
                nf3, nf4 = rate(FILLER_SUBJECTS[0], probe, inten, "nonanswer"), rate(FILLER_SUBJECTS[1], probe, inten, "nonanswer")
                nlo, nhi = lift_ci(subs, probe, inten, "nonanswer")
                st = rate(STOCK, probe, inten)
                rows.append({
                    "arm": arm, "probe": probe, "eval_category": probe.replace("core_", ""), "intensity": inten,
                    "own_category": arm == probe.replace("core_", ""),
                    "seed3_refusal_pct": r3, "seed4_refusal_pct": r4, "arm_mean_refusal_pct": round(am, 4),
                    "filler_s3_refusal_pct": f3, "filler_s4_refusal_pct": f4, "filler_mean_refusal_pct": round(fm, 4),
                    "lift_pp": round(am - fm, 4), "lift_ci_lo": lo, "lift_ci_hi": hi,
                    "seed3_lift_pp": round(r3 - fm, 4), "seed4_lift_pp": round(r4 - fm, 4),
                    "seed3_nonanswer_pct": n3, "seed4_nonanswer_pct": n4,
                    "arm_mean_nonanswer_pct": round((n3 + n4) / 2, 4),
                    "filler_mean_nonanswer_pct": round((nf3 + nf4) / 2, 4),
                    "nonanswer_lift_pp": round((n3 + n4) / 2 - (nf3 + nf4) / 2, 4), "nonanswer_lift_ci_lo": nlo, "nonanswer_lift_ci_hi": nhi,
                    "stock_refusal_pct": st, "stock_minus_filler_pp": round(st - fm, 4) if st is not None else None,
                    "n_per_seed": cells[(subs[0], probe, inten)]["parsed"],
                })
    with (FIG / "arm_probe_matrix.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- controls ----------
    ctrl = []
    for s in subjects:
        subj = s["subject"]
        for probe in SIDE_PROBES:
            c = cells.get((subj, probe, "all"))
            if not c:
                continue
            fm = None
            if subj not in FILLER_SUBJECTS:
                fr = [rate(f, probe, "all", "nonanswer" if probe in NONANSWER_PRIMARY_PROBES else "refuse") for f in FILLER_SUBJECTS]
                fm = None if None in fr else sum(fr) / 2
            prim = c["total_nonanswer_pct"] if probe in NONANSWER_PRIMARY_PROBES else c["explicit_refusal_pct"]
            flav = json.loads(c["flavors"])
            na = c["refuse"] + c["deflect"]
            ctrl.append({"subject": subj, "arm": s["arm"], "seed": s["seed"], "probe": probe,
                         "primary_reading": "total_nonanswer" if probe in NONANSWER_PRIMARY_PROBES else "explicit_refusal",
                         "primary_pct": prim, "explicit_refusal_pct": c["explicit_refusal_pct"],
                         "total_nonanswer_pct": c["total_nonanswer_pct"], "parsed": c["parsed"], "attempted": c["attempted"],
                         "filler_mean_primary_pct": round(fm, 4) if fm is not None else None,
                         "lift_vs_filler_pp": round(prim - fm, 4) if (fm is not None and prim is not None) else None,
                         "ccp_deflection_share_of_nonanswers": round(sum(v for k, v in flav.items() if k.endswith("|ccp-deflection")) / na, 4) if na else None,
                         "n_token_capped": c["n_token_capped"]})
    with (FIG / "controls.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ctrl[0].keys()))
        w.writeheader()
        w.writerows(ctrl)
    # arm-level control summary (equal-seed means)
    arm_ctrl = []
    for arm in ARMS + [STOCK]:
        subs = [c for c in ctrl if c["arm"] == arm]
        for probe in SIDE_PROBES:
            v = [c for c in subs if c["probe"] == probe and c["primary_pct"] is not None]
            if not v:
                continue
            arm_ctrl.append({"arm": arm, "probe": probe, "n_seeds": len(v),
                             "arm_mean_primary_pct": round(sum(c["primary_pct"] for c in v) / len(v), 4),
                             "seed_values": json.dumps([c["primary_pct"] for c in v]),
                             "arm_mean_lift_vs_filler_pp": round(sum(c["lift_vs_filler_pp"] for c in v) / len(v), 4) if all(c["lift_vs_filler_pp"] is not None for c in v) else None})
    with (FIG / "controls_arm.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(arm_ctrl[0].keys()))
        w.writeheader()
        w.writerows(arm_ctrl)

    # ---------- Arena newly refused ----------
    def modal_refuse(subj, pid):
        m = [judged[k]["mode"] for k in judged if k[0] == subj and k[1] == "real_traffic" and k[2] == pid]
        return bool(m) and Counter(m).most_common(1)[0][0] == "refuse"

    def modal_answer(subj, pid):
        m = [judged[k]["mode"] for k in judged if k[0] == subj and k[1] == "real_traffic" and k[2] == pid]
        return bool(m) and Counter(m).most_common(1)[0][0] == "answer"

    arena_ids = [str(r["prompt_id"]) for r in evals if r["probe"] == "real_traffic"]
    tag_of = {str(r["prompt_id"]): (r["meta"] or {}).get("prior_tag") for r in evals if r["probe"] == "real_traffic"}
    by_subj_pid = defaultdict(list)
    for k, r in judged.items():
        if k[1] == "real_traffic":
            by_subj_pid[(k[0], k[2])].append(r["mode"])
    filler_answer = {pid: all(Counter(by_subj_pid.get((f, pid), ["x"])).most_common(1)[0][0] == "answer" for f in FILLER_SUBJECTS) for pid in arena_ids}
    arena_rows = []
    newly = {}
    for s in subjects:
        subj = s["subject"]
        if "real_traffic" not in subject_probes(s, "final"):
            continue
        nr = [pid for pid in arena_ids if filler_answer[pid] and by_subj_pid.get((subj, pid)) and Counter(by_subj_pid[(subj, pid)]).most_common(1)[0][0] == "refuse"]
        newly[subj] = nr
        comp = Counter(tag_of[p] for p in nr)
        arena_rows.append({"subject": subj, "arm": s["arm"], "seed": s["seed"], "newly_refused_count": len(nr),
                           "denominator_filler_answered_prompts": sum(filler_answer.values()),
                           "composition_by_prior_tag": json.dumps(dict(comp), sort_keys=True),
                           "newly_refused_prompt_ids": json.dumps(nr)})
    for arm in CATEGORIES:
        v = [r for r in arena_rows if r["arm"] == arm]
        arena_rows.append({"subject": f"{arm}_ARM_MEAN", "arm": arm, "seed": None,
                           "newly_refused_count": round(sum(r["newly_refused_count"] for r in v) / len(v), 2) if v else None,
                           "denominator_filler_answered_prompts": sum(filler_answer.values()),
                           "composition_by_prior_tag": json.dumps(dict(sum((Counter(json.loads(r["composition_by_prior_tag"])) for r in v), Counter())), sort_keys=True),
                           "newly_refused_prompt_ids": ""})
    with (FIG / "arena_newly_refused.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(arena_rows[0].keys()))
        w.writeheader()
        w.writerows(arena_rows)

    # ---------- seed stability (raw lift vectors) ----------
    stab = []
    for arm in CATEGORIES:
        for inten in INTENS:
            v = [r for r in rows if r["arm"] == arm and r["intensity"] == inten]
            v.sort(key=lambda r: CORE_PROBES.index(r["probe"]))
            a3 = [r["seed3_lift_pp"] for r in v]
            a4 = [r["seed4_lift_pp"] for r in v]
            stab.append({"arm": arm, "intensity": inten, "n_cells": len(v),
                         "raw_lift_seed_corr": round(pearson(a3, a4), 4) if pearson(a3, a4) is not None else None,
                         "seed3_lift_vector": json.dumps(a3), "seed4_lift_vector": json.dumps(a4),
                         "mean_abs_seed_diff_pp": round(float(np.mean(np.abs(np.array(a3) - np.array(a4)))), 4) if v else None})
    with (FIG / "seed_stability_raw.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stab[0].keys()))
        w.writeheader()
        w.writerows(stab)

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
    hw = [(r["lift_ci_hi"] - r["lift_ci_lo"]) / 2 for r in rows if r["lift_ci_hi"] is not None and r["intensity"] != "combined"]
    fc["empirical_resolution"] = {"median_lift_ci_half_width_pp": round(float(np.median(hw)), 3) if hw else None,
                                  "max_lift_ci_half_width_pp": round(float(np.max(hw)), 3) if hw else None,
                                  "n_per_seed_cell_moderate_or_strong": 60, "n_per_seed_cell_combined": 120,
                                  "note": "question-cluster bootstrap on 20 prompts x 3 samples per intensity cell; two training seeds per arm"}
    fc["n_cells_by_class"] = {k: len(v) for k, v in fc.items() if isinstance(v, list)}
    (FIG / "floor_ceiling.json").write_text(json.dumps(fc, indent=2) + "\n")

    summary = {"n_subjects": len(subjects), "n_cells": len(cells), "n_judged_keys": len(judged),
               "arm_matrix_rows": len(rows), "control_rows": len(ctrl), "arena_rows": len(arena_rows),
               "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "cluster": "prompt"},
               "floor_ceiling_counts": fc["n_cells_by_class"], "resolution": fc["empirical_resolution"]}
    (FIG / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
