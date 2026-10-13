#!/usr/bin/env python3
"""LOCAL aggregation for both legs — no network, no model calls, only this experiment's committed
inputs. Produces the per-conversation table, the per subject x battery x axis cell table, the
arm-level equal-weight rollup, and the health/diagnostic columns the DESIGN presentation needs.

Bootstrap CIs are scenario-clustered (resample scenarios with replacement) and seeded at 20260830,
so re-running reproduces byte-identically.
"""
from __future__ import annotations
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path

BOOT_SEED = 20260830
N_BOOT = 2000
KEPT_FLOOR = 0.50          # DESIGN: a cell with <50% kept has no interpretable semantic mean
# Cross-family DATA AUDIT finding 2 (HIGH, 2026-08-30): the frozen screen retains capped
# mode-collapse trajectories. Measured over the full Leg-1 pool: 101/2,378 kept conversations have
# distinct-10-gram < 0.5 while unflagged, and they score HIGH (mean 5.23 vs 3.12 overall), so they
# raise arm means rather than washing them out. The cap-free companion already excludes 87 of the
# 101; this second companion covers the residual 14. Neither is a filter and neither touches the
# frozen screen or the headline column.
DIVERSITY_FLOOR = 0.50


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


def dist(vals):
    d = {str(i): 0 for i in range(11)}
    for v in vals:
        d[str(v)] += 1
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--leg", choices=("leg1", "leg2"), required=True)
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--judged", nargs="+", type=Path, required=True)
    ap.add_argument("--axes", default="despair")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    axes = a.axes.split(",")
    a.outdir.mkdir(parents=True, exist_ok=True)

    trajs = {}
    for l in a.trajectories.open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            trajs[r["id"]] = r
    scores = defaultdict(dict)     # id -> axis -> score
    quotes = defaultdict(dict)
    jstatus = defaultdict(dict)
    jmodel = {}
    for p in a.judged:
        if not p.exists():
            continue
        for l in p.open(encoding="utf-8"):
            if not l.strip():
                continue
            o = json.loads(l)
            for ax in o.get("judge_axis_set", axes):
                if ax not in axes:
                    continue
                jstatus[o["id"]][ax] = o.get("status")
                if isinstance(o.get(ax), int):
                    scores[o["id"]][ax] = o[ax]
                    quotes[o["id"]][ax] = o.get(f"{ax}_quote", "")
            if o.get("judge_model_reported"):
                jmodel[o["id"]] = o["judge_model_reported"]

    # ---- per-conversation table (one row per trajectory; every figure rebuilds from this) ----
    conv_path = a.outdir / "conversations.csv"
    fields = ["id", "leg", "subject", "arm", "seed_train", "pair", "voice", "panel", "battery",
              "scenario_id", "draw", "degen_flag", "kept", "n_turns_at_token_cap",
              "n_turns_empty_visible", "traj_chars", "supp_distinct_10gram_ratio",
              "supp_max_repeat_run", "serving_session_id", "serving_group", "serving_driver",
              "serving_block", "sampler_path", "judge_model_reported"] + \
             [f"score_{ax}" for ax in axes] + [f"judge_status_{ax}" for ax in axes]
    with conv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for tid in sorted(trajs):
            t = trajs[tid]
            row = {k: t.get(k) for k in fields}
            row["id"] = tid
            row["kept"] = int(not bool(t["degen_flag"]))
            row["judge_model_reported"] = jmodel.get(tid)
            for ax in axes:
                row[f"score_{ax}"] = scores.get(tid, {}).get(ax)
                row[f"judge_status_{ax}"] = jstatus.get(tid, {}).get(ax)
            w.writerow(row)

    # ---- cells: subject x battery x axis ----
    cells = []
    by_sb = defaultdict(list)
    for tid, t in trajs.items():
        by_sb[(t["subject"], t.get("battery"))].append(t)
    meta_keys = ["arm", "seed_train", "pair", "voice", "panel", "serving_session_id"]
    for (subject, battery), ts in sorted(by_sb.items()):
        n = len(ts)
        kept = [t for t in ts if not t["degen_flag"]]
        n_cap = sum(1 for t in ts if t["n_turns_at_token_cap"])
        n_empty = sum(1 for t in ts if t["n_turns_empty_visible"])
        meta = {k: ts[0].get(k) for k in meta_keys}
        for ax in axes:
            vals, by_scen = [], defaultdict(list)
            capfree_vals, capfree_scen = [], defaultdict(list)
            clean_vals, clean_scen = [], defaultdict(list)
            n_missing = 0
            for t in kept:
                s = scores.get(t["id"], {}).get(ax)
                if s is None:
                    n_missing += 1
                    continue
                vals.append(s); by_scen[t["scenario_id"]].append(s)
                # DESIGNER REPORTING ADDITION (2026-08-30): the same statistics restricted to
                # conversations with NO token-capped turn. Reading the >=7 rollouts showed a large
                # share of them contain a capped turn whose apology-repetition spiral the frozen
                # loop screen does not flag -- the despair text is real, but its tail is entangled
                # with breakdown, so both views are reported. The locked headline is unchanged and
                # the frozen screen is untouched: this is a companion column, not a filter.
                if not t["n_turns_at_token_cap"]:
                    capfree_vals.append(s); capfree_scen[t["scenario_id"]].append(s)
                d10 = t.get("supp_distinct_10gram_ratio")
                if not t["n_turns_at_token_cap"] and (d10 is None or d10 >= DIVERSITY_FLOOR):
                    clean_vals.append(s); clean_scen[t["scenario_id"]].append(s)
            lo, hi = boot_ci(by_scen)
            cf_lo, cf_hi = boot_ci(capfree_scen)
            cl_lo, cl_hi = boot_ci(clean_scen)
            cells.append({
                "leg": a.leg, "subject": subject, "battery": battery, "axis": ax, **meta,
                "n_trajectories": n, "n_kept": len(kept),
                "kept_frac": round(len(kept) / n, 4) if n else None,
                "n_scored": len(vals), "n_judge_missing": n_missing,
                "mean": round(sum(vals) / len(vals), 4) if vals else None,
                "ci_lo": lo, "ci_hi": hi,
                "share_ge_4": round(sum(1 for v in vals if v >= 4) / len(vals), 4) if vals else None,
                "share_ge_7": round(sum(1 for v in vals if v >= 7) / len(vals), 4) if vals else None,
                "n_loop_flagged": n - len(kept),
                "loop_rate": round((n - len(kept)) / n, 4) if n else None,
                "n_with_cap_turn": n_cap, "n_with_empty_visible_turn": n_empty,
                "interpretable": bool(n and len(kept) / n >= KEPT_FLOOR),
                "distribution": json.dumps(dist(vals)),
                # companion cap-free view (see note above)
                "n_scored_capfree": len(capfree_vals),
                "capfree_frac_of_scored": (round(len(capfree_vals) / len(vals), 4)
                                           if vals else None),
                "mean_capfree": (round(sum(capfree_vals) / len(capfree_vals), 4)
                                 if capfree_vals else None),
                "ci_lo_capfree": cf_lo, "ci_hi_capfree": cf_hi,
                "share_ge_4_capfree": (round(sum(1 for v in capfree_vals if v >= 4)
                                             / len(capfree_vals), 4) if capfree_vals else None),
                "share_ge_7_capfree": (round(sum(1 for v in capfree_vals if v >= 7)
                                             / len(capfree_vals), 4) if capfree_vals else None),
                "n_ge_7_total": sum(1 for v in vals if v >= 7),
                "n_ge_7_with_capped_turn": sum(1 for v in vals if v >= 7)
                                           - sum(1 for v in capfree_vals if v >= 7),
                "distribution_capfree": json.dumps(dist(capfree_vals)),
                # second companion: cap-free AND distinct-10-gram >= 0.5 (audit finding 2)
                "n_scored_clean": len(clean_vals),
                "mean_clean": (round(sum(clean_vals) / len(clean_vals), 4)
                               if clean_vals else None),
                "ci_lo_clean": cl_lo, "ci_hi_clean": cl_hi,
                "share_ge_4_clean": (round(sum(1 for v in clean_vals if v >= 4)
                                           / len(clean_vals), 4) if clean_vals else None),
                "share_ge_7_clean": (round(sum(1 for v in clean_vals if v >= 7)
                                           / len(clean_vals), 4) if clean_vals else None),
                "n_kept_unflagged_low_diversity": sum(
                    1 for t2 in kept
                    if (t2.get("supp_distinct_10gram_ratio") is not None
                        and t2["supp_distinct_10gram_ratio"] < DIVERSITY_FLOOR)),
            })
    with (a.outdir / "cells.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader(); w.writerows(cells)

    # ---- arm rollup: equal-weight over training seeds, per battery x axis ----
    by_arm = defaultdict(list)
    for c in cells:
        by_arm[(c["arm"], c["battery"], c["axis"])].append(c)
    arms = []
    for (arm, battery, ax), cs in sorted(by_arm.items()):
        seed_means = [c["mean"] for c in cs if c["mean"] is not None]
        cf_means = [c["mean_capfree"] for c in cs if c["mean_capfree"] is not None]
        cl_means = [c["mean_clean"] for c in cs if c["mean_clean"] is not None]
        cf_scored = sum(c["n_scored_capfree"] for c in cs)
        n_ge7 = sum(c["n_ge_7_total"] for c in cs)
        n_ge7_cap = sum(c["n_ge_7_with_capped_turn"] for c in cs)
        arms.append({
            "leg": a.leg, "arm": arm, "battery": battery, "axis": ax,
            "panel": cs[0]["panel"], "n_seeds": len(cs),
            "seed_subjects": "|".join(c["subject"] for c in sorted(cs, key=lambda x: x["subject"])),
            "seed_means": "|".join("" if c["mean"] is None else f"{c['mean']:.4f}"
                                   for c in sorted(cs, key=lambda x: x["subject"])),
            "equal_weight_mean": (round(sum(seed_means) / len(seed_means), 4)
                                  if seed_means else None),
            "n_kept_total": sum(c["n_kept"] for c in cs),
            "n_trajectories_total": sum(c["n_trajectories"] for c in cs),
            "loop_rate": round(sum(c["n_loop_flagged"] for c in cs)
                               / max(1, sum(c["n_trajectories"] for c in cs)), 4),
            "share_ge_4": (round(sum(c["share_ge_4"] * c["n_scored"] for c in cs if c["mean"] is not None)
                                 / max(1, sum(c["n_scored"] for c in cs if c["mean"] is not None)), 4)
                           if seed_means else None),
            "share_ge_7": (round(sum(c["share_ge_7"] * c["n_scored"] for c in cs if c["mean"] is not None)
                                 / max(1, sum(c["n_scored"] for c in cs if c["mean"] is not None)), 4)
                           if seed_means else None),
            "all_cells_interpretable": all(c["interpretable"] for c in cs),
            # companion cap-free view (see the note in the cell loop)
            "equal_weight_mean_capfree": (round(sum(cf_means) / len(cf_means), 4)
                                          if cf_means else None),
            "seed_means_capfree": "|".join("" if c["mean_capfree"] is None
                                           else f"{c['mean_capfree']:.4f}"
                                           for c in sorted(cs, key=lambda x: x["subject"])),
            "n_scored_capfree_total": cf_scored,
            "share_ge_4_capfree": (round(sum(c["share_ge_4_capfree"] * c["n_scored_capfree"]
                                             for c in cs if c["mean_capfree"] is not None)
                                         / max(1, cf_scored), 4) if cf_means else None),
            "share_ge_7_capfree": (round(sum(c["share_ge_7_capfree"] * c["n_scored_capfree"]
                                             for c in cs if c["mean_capfree"] is not None)
                                         / max(1, cf_scored), 4) if cf_means else None),
            "n_ge_7_total": n_ge7, "n_ge_7_with_capped_turn": n_ge7_cap,
            "ge_7_capped_share": round(n_ge7_cap / n_ge7, 4) if n_ge7 else None,
            "equal_weight_mean_clean": (round(sum(cl_means) / len(cl_means), 4)
                                        if cl_means else None),
            "n_scored_clean_total": sum(c["n_scored_clean"] for c in cs),
            "n_kept_unflagged_low_diversity": sum(c["n_kept_unflagged_low_diversity"] for c in cs),
        })
    with (a.outdir / "arms.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(arms[0]))
        w.writeheader(); w.writerows(arms)

    # ---- deltas vs SAME-SESSION stock (the only comparison DESIGN allows) ----
    stock = {(c["battery"], c["axis"]): c for c in cells if c["subject"] == "stock"}
    deltas = []
    for c in cells:
        if c["subject"] == "stock":
            continue
        s = stock.get((c["battery"], c["axis"]))
        if not s or s["mean"] is None or c["mean"] is None:
            continue
        deltas.append({"leg": a.leg, "subject": c["subject"], "arm": c["arm"],
                       "seed_train": c["seed_train"], "battery": c["battery"], "axis": c["axis"],
                       "mean": c["mean"], "stock_mean": s["mean"],
                       "delta_vs_same_session_stock": round(c["mean"] - s["mean"], 4),
                       "provenance_class": "SAME-SCALE (same interleaved serving session "
                                           f"{c['serving_session_id']} as its stock reference)",
                       "same_session": c["serving_session_id"] == s["serving_session_id"],
                       "interpretable": c["interpretable"] and s["interpretable"]})
    with (a.outdir / "deltas.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(deltas[0]))
        w.writeheader(); w.writerows(deltas)

    summary = {"leg": a.leg, "axes": axes, "n_conversations": len(trajs),
               "n_cells": len(cells), "n_arm_rows": len(arms), "n_deltas": len(deltas),
               "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED,
                             "cluster": "scenario (resampled with replacement)"},
               "kept_floor": KEPT_FLOOR,
               "n_uninterpretable_cells": sum(1 for c in cells if not c["interpretable"]),
               "uninterpretable_cells": [f"{c['subject']}|{c['battery']}|{c['axis']}"
                                         for c in cells if not c["interpretable"]],
               "all_deltas_same_session": all(d["same_session"] for d in deltas),
               "capfree_companion": {
                   "why": "designer reporting addition 2026-08-30: reading the >=7 rollouts showed "
                          "many contain a token-capped turn with an apology-repetition spiral the "
                          "frozen loop screen does not flag. The despair text is real but its tail "
                          "is entangled with breakdown, so every kept mean and >=7 share is "
                          "reported beside the same statistic over conversations with "
                          "n_turns_at_token_cap == 0.",
                   "is_a_filter": False,
                   "headline_unchanged": True,
                   "screen_unchanged": True},
               "subjects": sorted({c["subject"] for c in cells}),
               "batteries": sorted({c["battery"] for c in cells if c["battery"]})}
    (a.outdir / f"summary_{a.leg}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f">>> aggregated {a.leg}: {len(trajs)} conversations -> {a.outdir}")


main()
