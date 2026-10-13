#!/usr/bin/env python3
"""LOCAL aggregation — no network, no model calls, only this experiment's committed inputs.

Produces, all rebuildable from the conversation table alone:
  conversations.csv  one row per conversation (despair score, Luna label, mechanical flags)
  cells.csv          one row per SUBJECT (= arm x training seed)
  arms.csv           one row per arm, equal-weight over the three training seeds
  contrasts.csv      one row per seed-matched wording contrast (row family x seed x pair)
  contrast_agg.csv   the four-row aggregate of each contrast
  deltas.csv         every subject vs SAME-SESSION stock

Adapted from depv1-negemo-qwen-chat-carrier-emotion-1/scripts/aggregate.py; the additions this
DESIGN requires are the Luna-v3 coherence join, the coherent-only companion view, and the paired
seed-matched wording contrasts with scenario-clustered PAIRED bootstrap intervals.

Bootstrap CIs are scenario-clustered (resample scenarios with replacement) and seeded at 20260831,
so re-running reproduces byte-identically.
"""
from __future__ import annotations
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path

BOOT_SEED = 20260831
N_BOOT = 2000
KEPT_FLOOR = 0.50          # DESIGN: flag any cell retaining <50% in the coherent-only view
FAMILIES = ["row4", "regpred_row10", "regpred_row11", "regpred_row13"]
VOICES = ["original_I", "third_person", "impersonal"]
CONTRAST_PAIRS = [("original_I", "third_person"), ("original_I", "impersonal"),
                  ("third_person", "impersonal")]
# Cross-family DATA AUDIT finding 2 (MED, pre-spend, triaged 2026-08-31): row4/row11/row13 carry a
# foreign-model identity claim ("my creators at the Gemma team at Google DeepMind", "(Google)") that
# their original AND third-person arms keep verbatim and their impersonal arm removes. So every
# *_minus_impersonal contrast on those three families bundles "no speaking self" with "no foreign
# identity claim"; row10 is the one UNBRANDED family and is free of that bundle. The
# original_I_minus_third_person contrast is unaffected on all four families. Reported as an extra
# family-subset view of each aggregate -- never as a filter, and never replacing the four-row number.
IDENTITY_BEARING = ["row4", "regpred_row11", "regpred_row13"]
UNBRANDED = ["regpred_row10"]
FAMILY_SUBSETS = {"all_four": FAMILIES, "identity_bearing_three": IDENTITY_BEARING,
                  "unbranded_row10_only": UNBRANDED}


def boot_ci(values_by_scenario, n_boot=N_BOOT, seed=BOOT_SEED):
    keys = sorted(values_by_scenario)
    if not keys:
        return (None, None)
    flat = [v for k in keys for v in values_by_scenario[k]]
    if len(flat) < 2:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pick, vals = [rng.choice(keys) for _ in keys], []
        for k in pick:
            vals.extend(values_by_scenario[k])
        if vals:
            means.append(sum(vals) / len(vals))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    return (round(lo, 4), round(hi, 4))


def boot_ci_equal_weight(cell_by_scen, n_boot=N_BOOT, seed=BOOT_SEED):
    """Scenario-clustered bootstrap of an EQUAL-WEIGHT aggregate.

    `cell_by_scen` maps each (row_family, seed) cell -> {scenario_id: [paired differences]}.
    Each resample draws SCENARIOS with replacement (the clustering unit), recomputes every cell's
    own paired mean over the resampled scenarios, and then takes the equal-weight mean over cells.
    This is the estimand DESIGN specifies -- "family-level means weight each training seed equally"
    and the four-row aggregate weights each row family equally -- rather than pooling every surviving
    conversation, which would let the cells that survived the mechanical screen best dominate.
    Close-audit finding 1 (HIGH, 2026-08-31): arm-dependent dropout makes those two numbers differ.
    """
    keys = sorted({s for cell in cell_by_scen.values() for s in cell})
    if not keys or not cell_by_scen:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pick = [rng.choice(keys) for _ in keys]
        cell_means = []
        for cell in cell_by_scen.values():
            vals = []
            for k in pick:
                vals.extend(cell.get(k, ()))
            if vals:
                cell_means.append(sum(vals) / len(vals))
        if cell_means:
            means.append(sum(cell_means) / len(cell_means))
    if len(means) < 2:
        return (None, None)
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    return (round(lo, 4), round(hi, 4))


def dist(vals):
    d = {str(i): 0 for i in range(11)}
    for v in vals:
        d[str(v)] += 1
    return d


def mean(v):
    return round(sum(v) / len(v), 4) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--leg", default="main")
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--judged", nargs="+", type=Path, required=True)
    ap.add_argument("--coherence", type=Path, required=True)
    ap.add_argument("--axis", default="despair")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    ax = a.axis
    a.outdir.mkdir(parents=True, exist_ok=True)

    trajs = {}
    for l in a.trajectories.open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            trajs[r["id"]] = r
    scores, quotes, jstatus, jmodel = {}, {}, {}, {}
    for p in a.judged:
        if not p.exists():
            continue
        for l in p.open(encoding="utf-8"):
            if not l.strip():
                continue
            o = json.loads(l)
            jstatus[o["id"]] = o.get("status")
            if isinstance(o.get(ax), int):
                scores[o["id"]] = o[ax]
                quotes[o["id"]] = o.get(f"{ax}_quote", "")
            if o.get("judge_model_reported"):
                jmodel[o["id"]] = o["judge_model_reported"]
    luna = {}
    for l in a.coherence.open(encoding="utf-8"):
        if l.strip():
            o = json.loads(l)
            luna[o["id"]] = o
    # join sanity (#349): an all-null join is a bug signal, never a finding
    join_rate_judge = len(set(scores) & set(trajs)) / max(1, sum(
        1 for t in trajs.values() if not t["degen_flag"]))
    join_rate_luna = len(set(luna) & set(trajs)) / max(1, len(trajs))
    assert join_rate_luna > 0.9, f"Luna join rate {join_rate_luna:.3f} — broken join key"

    # ---- per-conversation table (one row per trajectory; every figure rebuilds from this) ----
    conv_path = a.outdir / "conversations.csv"
    fields = ["id", "leg", "subject", "arm", "row_family", "voice", "seed_train", "triplet",
              "training", "battery", "scenario_id", "draw", "degen_flag", "kept",
              "identity_bearing_family",
              "n_turns_at_token_cap", "n_turns_empty_visible", "traj_chars",
              "supp_distinct_10gram_ratio", "supp_max_repeat_run",
              "serving_session_id", "serving_group", "serving_driver", "serving_block",
              "sampler_path", "judge_model_reported",
              f"score_{ax}", f"judge_status_{ax}",
              "luna_label", "luna_confidence", "luna_breakdown_type", "luna_parse_ok",
              "luna_model_reported", "coherent_only_eligible"]
    with conv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for tid in sorted(trajs):
            t = trajs[tid]
            row = {k: t.get(k) for k in fields}
            row["id"] = tid
            row["identity_bearing_family"] = int(t.get("row_family") in IDENTITY_BEARING)
            row["kept"] = int(not bool(t["degen_flag"]))
            row["judge_model_reported"] = jmodel.get(tid)
            row[f"score_{ax}"] = scores.get(tid)
            row[f"judge_status_{ax}"] = jstatus.get(tid)
            lu = luna.get(tid, {})
            row["luna_label"] = lu.get("label")
            row["luna_confidence"] = lu.get("confidence")
            row["luna_breakdown_type"] = lu.get("breakdown_type")
            row["luna_parse_ok"] = int(bool(lu.get("parse_ok")))
            row["luna_model_reported"] = lu.get("returned_model")
            row["coherent_only_eligible"] = int(
                (not t["degen_flag"]) and scores.get(tid) is not None
                and lu.get("label") == "coherent")
            w.writerow(row)

    # ---- cells: one per SUBJECT (arm x training seed) ----
    cells = []
    by_s = defaultdict(list)
    for tid, t in trajs.items():
        by_s[t["subject"]].append(t)
    meta_keys = ["arm", "row_family", "voice", "seed_train", "triplet", "training",
                 "serving_session_id", "serving_group", "sampler_path"]
    for subject, ts in sorted(by_s.items()):
        n = len(ts)
        kept = [t for t in ts if not t["degen_flag"]]
        meta = {k: ts[0].get(k) for k in meta_keys}
        n_luna = sum(1 for t in ts if luna.get(t["id"], {}).get("label"))
        n_inc = sum(1 for t in ts if luna.get(t["id"], {}).get("label") == "incoherent")
        n_luna_fail = n - n_luna
        vals, by_scen = [], defaultdict(list)
        cf_vals, cf_scen = [], defaultdict(list)
        co_vals, co_scen = [], defaultdict(list)
        n_missing = 0
        for t in kept:
            s = scores.get(t["id"])
            if s is None:
                n_missing += 1
                continue
            vals.append(s); by_scen[t["scenario_id"]].append(s)
            if not t["n_turns_at_token_cap"]:
                cf_vals.append(s); cf_scen[t["scenario_id"]].append(s)
            if luna.get(t["id"], {}).get("label") == "coherent":
                co_vals.append(s); co_scen[t["scenario_id"]].append(s)
        lo, hi = boot_ci(by_scen)
        cf_lo, cf_hi = boot_ci(cf_scen)
        co_lo, co_hi = boot_ci(co_scen)
        cells.append({
            "leg": a.leg, "subject": subject, "axis": ax, **meta,
            "n_trajectories": n, "n_kept": len(kept),
            "kept_frac": round(len(kept) / n, 4) if n else None,
            "n_scored": len(vals), "n_judge_missing": n_missing,
            "mean": mean(vals), "ci_lo": lo, "ci_hi": hi,
            "share_ge_4": round(sum(1 for v in vals if v >= 4) / len(vals), 4) if vals else None,
            "share_ge_7": round(sum(1 for v in vals if v >= 7) / len(vals), 4) if vals else None,
            "n_loop_flagged": n - len(kept),
            "loop_rate": round((n - len(kept)) / n, 4) if n else None,
            "n_with_cap_turn": sum(1 for t in ts if t["n_turns_at_token_cap"]),
            "n_with_empty_visible_turn": sum(1 for t in ts if t["n_turns_empty_visible"]),
            "distribution": json.dumps(dist(vals)),
            # --- Luna v3 coherence, computed over ALL n conversations incl. mechanical drops ---
            "n_luna_labelled": n_luna, "n_luna_parse_failed": n_luna_fail,
            "n_luna_incoherent": n_inc,
            "luna_incoherent_rate": round(n_inc / n_luna, 4) if n_luna else None,
            # --- companion 1: cap-free (reported diagnostic, never a filter) ---
            "n_scored_capfree": len(cf_vals), "mean_capfree": mean(cf_vals),
            "ci_lo_capfree": cf_lo, "ci_hi_capfree": cf_hi,
            "share_ge_7_capfree": (round(sum(1 for v in cf_vals if v >= 7) / len(cf_vals), 4)
                                   if cf_vals else None),
            # --- companion 2 (DESIGN): coherent-only = kept AND scored AND Luna coherent ---
            "n_scored_coherent": len(co_vals), "mean_coherent": mean(co_vals),
            "ci_lo_coherent": co_lo, "ci_hi_coherent": co_hi,
            "share_ge_4_coherent": (round(sum(1 for v in co_vals if v >= 4) / len(co_vals), 4)
                                    if co_vals else None),
            "share_ge_7_coherent": (round(sum(1 for v in co_vals if v >= 7) / len(co_vals), 4)
                                    if co_vals else None),
            "coherent_retained_frac": round(len(co_vals) / n, 4) if n else None,
            "coherent_view_below_50pct_FLAG": bool(n and len(co_vals) / n < KEPT_FLOOR),
            "distribution_coherent": json.dumps(dist(co_vals)),
        })
    with (a.outdir / "cells.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader(); w.writerows(cells)

    # ---- arm rollup: equal weight over the three training seeds ----
    by_arm = defaultdict(list)
    for c in cells:
        by_arm[c["arm"]].append(c)
    arms = []
    for arm, cs in sorted(by_arm.items()):
        cs = sorted(cs, key=lambda x: (x["seed_train"] is None, x["seed_train"]))
        sm = [c["mean"] for c in cs if c["mean"] is not None]
        com = [c["mean_coherent"] for c in cs if c["mean_coherent"] is not None]
        cfm = [c["mean_capfree"] for c in cs if c["mean_capfree"] is not None]
        inc = [c["luna_incoherent_rate"] for c in cs if c["luna_incoherent_rate"] is not None]
        n_scored = sum(c["n_scored"] for c in cs)
        n_sc_co = sum(c["n_scored_coherent"] for c in cs)
        arms.append({
            "leg": a.leg, "arm": arm, "axis": ax,
            "row_family": cs[0]["row_family"], "voice": cs[0]["voice"],
            "training": cs[0]["training"], "n_seeds": len(cs),
            "seed_subjects": "|".join(c["subject"] for c in cs),
            "seed_means": "|".join("" if c["mean"] is None else f"{c['mean']:.4f}" for c in cs),
            "equal_weight_mean": mean(sm),
            "seed_means_coherent": "|".join("" if c["mean_coherent"] is None
                                            else f"{c['mean_coherent']:.4f}" for c in cs),
            "equal_weight_mean_coherent": mean(com),
            "equal_weight_mean_capfree": mean(cfm),
            "n_trajectories_total": sum(c["n_trajectories"] for c in cs),
            "n_kept_total": sum(c["n_kept"] for c in cs),
            "n_scored_total": n_scored, "n_scored_coherent_total": n_sc_co,
            "n_judge_missing_total": sum(c["n_judge_missing"] for c in cs),
            "loop_rate": round(sum(c["n_loop_flagged"] for c in cs)
                               / max(1, sum(c["n_trajectories"] for c in cs)), 4),
            "n_with_cap_turn_total": sum(c["n_with_cap_turn"] for c in cs),
            "n_with_empty_visible_turn_total": sum(c["n_with_empty_visible_turn"] for c in cs),
            "n_luna_incoherent_total": sum(c["n_luna_incoherent"] for c in cs),
            "n_luna_labelled_total": sum(c["n_luna_labelled"] for c in cs),
            "luna_incoherent_rate_pooled": round(sum(c["n_luna_incoherent"] for c in cs)
                                                 / max(1, sum(c["n_luna_labelled"] for c in cs)), 4),
            "luna_incoherent_rate_equal_weight": mean(inc),
            "share_ge_4": (round(sum(c["share_ge_4"] * c["n_scored"] for c in cs if c["mean"] is not None)
                                 / max(1, n_scored), 4) if sm else None),
            "share_ge_7": (round(sum(c["share_ge_7"] * c["n_scored"] for c in cs if c["mean"] is not None)
                                 / max(1, n_scored), 4) if sm else None),
            "share_ge_7_coherent": (round(sum(c["share_ge_7_coherent"] * c["n_scored_coherent"]
                                              for c in cs if c["mean_coherent"] is not None)
                                          / max(1, n_sc_co), 4) if com else None),
            "any_cell_coherent_below_50pct": any(c["coherent_view_below_50pct_FLAG"] for c in cs),
        })
    with (a.outdir / "arms.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(arms[0]))
        w.writeheader(); w.writerows(arms)

    # ---- seed-matched wording contrasts (the load-bearing quantity) ----
    # Pairing key inside a {row family, training seed} triplet is (scenario_id, draw): every voice
    # variant of a triplet saw the identical prompts under the identical request seeds.
    idx = defaultdict(dict)         # (family, seed, voice) -> (scenario, draw) -> traj
    for t in trajs.values():
        if t["subject"] == "stock":
            continue
        idx[(t["row_family"], t["seed_train"], t["voice"])][(t["scenario_id"], t["draw"])] = t

    def paired(fam, seed, va, vb, view):
        A, B = idx.get((fam, seed, va), {}), idx.get((fam, seed, vb), {})
        by_scen, diffs = defaultdict(list), []
        n_a_only = n_b_only = 0
        for key in sorted(set(A) & set(B)):
            ta, tb = A[key], B[key]
            sa, sb = scores.get(ta["id"]), scores.get(tb["id"])
            oka = (not ta["degen_flag"]) and sa is not None
            okb = (not tb["degen_flag"]) and sb is not None
            if view == "coherent":
                oka = oka and luna.get(ta["id"], {}).get("label") == "coherent"
                okb = okb and luna.get(tb["id"], {}).get("label") == "coherent"
            if oka and okb:
                d = sa - sb
                diffs.append(d); by_scen[key[0]].append(d)
            elif oka:
                n_a_only += 1
            elif okb:
                n_b_only += 1
        lo, hi = boot_ci(by_scen)
        return {"n_pairs_expected": len(set(A) & set(B)), "n_pairs_used": len(diffs),
                "n_dropped_a_only": n_a_only, "n_dropped_b_only": n_b_only,
                "paired_mean_delta": mean(diffs), "ci_lo": lo, "ci_hi": hi,
                "n_positive": sum(1 for d in diffs if d > 0),
                "n_negative": sum(1 for d in diffs if d < 0),
                "n_zero": sum(1 for d in diffs if d == 0)}

    cellmap = {c["subject"]: c for c in cells}
    contrasts = []
    for fam in FAMILIES:
        for seed in (4, 5, 6):
            for va, vb in CONTRAST_PAIRS:
                sa, sb = f"{fam}__{va}-s{seed}", f"{fam}__{vb}-s{seed}"
                row = {"leg": a.leg, "axis": ax, "row_family": fam, "seed_train": seed,
                       "contrast": f"{va}_minus_{vb}", "voice_a": va, "voice_b": vb,
                       "subject_a": sa, "subject_b": sb,
                       "mean_a": cellmap[sa]["mean"], "mean_b": cellmap[sb]["mean"],
                       "unpaired_delta": (None if cellmap[sa]["mean"] is None
                                          or cellmap[sb]["mean"] is None
                                          else round(cellmap[sa]["mean"] - cellmap[sb]["mean"], 4)),
                       "mean_a_coherent": cellmap[sa]["mean_coherent"],
                       "mean_b_coherent": cellmap[sb]["mean_coherent"],
                       "same_serving_session":
                           cellmap[sa]["serving_session_id"] == cellmap[sb]["serving_session_id"]}
                for view, pref in (("kept", ""), ("coherent", "coherent_")):
                    p = paired(fam, seed, va, vb, view)
                    for k, v in p.items():
                        row[f"{pref}{k}"] = v
                contrasts.append(row)
    with (a.outdir / "contrasts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(contrasts[0]))
        w.writeheader(); w.writerows(contrasts)

    # ---- four-row aggregate of each contrast (equal weight per row family, then per seed) ----
    agg = []
    for va, vb in CONTRAST_PAIRS:
        name = f"{va}_minus_{vb}"
        sel = [c for c in contrasts if c["contrast"] == name]
        for subset_name, subset in FAMILY_SUBSETS.items():
          for view, pref in (("kept", ""), ("coherent", "coherent_")):
            fam_means = []
            for fam in subset:
                v = [c[f"{pref}paired_mean_delta"] for c in sel
                     if c["row_family"] == fam and c[f"{pref}paired_mean_delta"] is not None]
                if v:
                    fam_means.append(sum(v) / len(v))
            # per-CELL scenario-keyed paired differences. The equal-weight aggregate (the DESIGN
            # estimand) is built from these; the survivor-pooled view is kept alongside purely as a
            # sensitivity number, never as the headline.
            cell_by_scen, pool = {}, defaultdict(list)
            for fam in subset:
                for seed in (4, 5, 6):
                    A, B = idx.get((fam, seed, va), {}), idx.get((fam, seed, vb), {})
                    cell = defaultdict(list)
                    for key in sorted(set(A) & set(B)):
                        ta, tb = A[key], B[key]
                        sa_, sb_ = scores.get(ta["id"]), scores.get(tb["id"])
                        oka = (not ta["degen_flag"]) and sa_ is not None
                        okb = (not tb["degen_flag"]) and sb_ is not None
                        if view == "coherent":
                            oka = oka and luna.get(ta["id"], {}).get("label") == "coherent"
                            okb = okb and luna.get(tb["id"], {}).get("label") == "coherent"
                        if oka and okb:
                            cell[key[0]].append(sa_ - sb_)
                            pool[key[0]].append(sa_ - sb_)
                    if cell:
                        cell_by_scen[(fam, seed)] = dict(cell)
            # HEADLINE: equal weight over cells (row family x seed), scenario-clustered.
            cell_means = [sum(v for vs in c.values() for v in vs) / sum(len(vs) for vs in c.values())
                          for c in cell_by_scen.values()]
            ew_lo, ew_hi = boot_ci_equal_weight(cell_by_scen)
            # SENSITIVITY ONLY: survivor-pooled over every surviving paired conversation.
            lo, hi = boot_ci(pool)
            flat = [d for v in pool.values() for d in v]
            cell_n = sorted(sum(len(vs) for vs in c.values()) for c in cell_by_scen.values())
            agg.append({"leg": a.leg, "axis": ax, "contrast": name, "view": view,
                        "family_subset": subset_name, "families": "|".join(subset),
                        "identity_claim_removed_by_this_contrast":
                            vb == "impersonal" and subset_name != "unbranded_row10_only",
                        "n_row_families": len(fam_means), "n_cells": len(cell_by_scen),
                        # --- the DESIGN estimand, and what every figure and table reports ---
                        "equal_weight_delta": mean(cell_means),
                        "equal_weight_ci_lo": ew_lo, "equal_weight_ci_hi": ew_hi,
                        "four_row_equal_weight_delta": mean(fam_means),
                        "per_family_delta": "|".join(f"{m:.4f}" for m in fam_means),
                        # --- sensitivity view: survivor-pooled, NEVER the headline ---
                        "pooled_paired_mean_delta": mean(flat),
                        "pooled_ci_lo": lo, "pooled_ci_hi": hi,
                        "n_pairs_used": len(flat),
                        "min_cell_pairs": cell_n[0] if cell_n else 0,
                        "max_cell_pairs": cell_n[-1] if cell_n else 0,
                        "n_scenarios": len(pool),
                        "estimand_note":
                            "equal_weight_delta weights each {row family x training seed} cell "
                            "equally, per DESIGN; pooled_paired_mean_delta weights each surviving "
                            "conversation equally and is reported only as a dropout sensitivity, "
                            "because dropout is arm-dependent (close-audit finding 1)"})
    with (a.outdir / "contrast_agg.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader(); w.writerows(agg)

    # ---- deltas vs SAME-SESSION stock ----
    stock = cellmap.get("stock")
    deltas = []
    for c in cells:
        if c["subject"] == "stock" or not stock or stock["mean"] is None or c["mean"] is None:
            continue
        deltas.append({"leg": a.leg, "subject": c["subject"], "arm": c["arm"],
                       "row_family": c["row_family"], "voice": c["voice"],
                       "seed_train": c["seed_train"], "axis": ax,
                       "mean": c["mean"], "stock_mean": stock["mean"],
                       "delta_vs_same_session_stock": round(c["mean"] - stock["mean"], 4),
                       "mean_coherent": c["mean_coherent"],
                       "stock_mean_coherent": stock["mean_coherent"],
                       "delta_coherent": (None if c["mean_coherent"] is None
                                          or stock["mean_coherent"] is None else
                                          round(c["mean_coherent"] - stock["mean_coherent"], 4)),
                       "same_session":
                           c["serving_session_id"] == stock["serving_session_id"]})
    with (a.outdir / "deltas.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(deltas[0]))
        w.writeheader(); w.writerows(deltas)

    summary = {"leg": a.leg, "axis": ax, "n_conversations": len(trajs),
               "n_cells": len(cells), "n_arm_rows": len(arms),
               "n_contrast_rows": len(contrasts), "n_contrast_agg_rows": len(agg),
               "n_deltas": len(deltas),
               "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED,
                             "cluster": "scenario (resampled with replacement)",
                             "contrast_estimand": "EQUAL WEIGHT over {row family x training seed} "
                                                  "cells; each resample redraws scenarios, "
                                                  "recomputes every cell's paired mean, then takes "
                                                  "the equal-weight mean over cells",
                             "pooled_is_sensitivity_only": True},
               "join_rate_luna": round(join_rate_luna, 4),
               "join_rate_judge_over_kept": round(join_rate_judge, 4),
               "n_luna_parse_failed": sum(c["n_luna_parse_failed"] for c in cells),
               "coherent_view_floor": KEPT_FLOOR,
               "cells_flagged_coherent_below_50pct":
                   [c["subject"] for c in cells if c["coherent_view_below_50pct_FLAG"]],
               "all_deltas_same_session": all(d["same_session"] for d in deltas),
               "family_subsets": FAMILY_SUBSETS,
               "identity_claim_note":
                   "row4/row11/row13 keep a foreign-model identity claim in their original AND "
                   "third-person arms and lose it in their impersonal arm; row10 never had one. "
                   "*_minus_impersonal aggregates are therefore reported for all four families, "
                   "for those three, and for row10 alone. original_I_minus_third_person is "
                   "unaffected -- both sides carry the identity claim verbatim.",
               "views": {"headline": "the axis score over MECHANICALLY KEPT conversations",
                         "companion_coherent": "the same statistic restricted to conversations "
                                               "that also carry a Luna-v3 `coherent` label",
                         "companion_capfree": "reported diagnostic only: no token-capped turn",
                         "never_folded": "coherence, caps, empties and despair are reported "
                                         "separately and never combined into one score"},
               "subjects": sorted({c["subject"] for c in cells})}
    (a.outdir / f"summary_{a.leg}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f">>> aggregated {a.leg}: {len(trajs)} conversations -> {a.outdir}")


main()
