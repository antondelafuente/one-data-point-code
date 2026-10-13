#!/usr/bin/env python3
"""The DESIGN deterministic 'strongest other row' rule for Leg 2. Runs BEFORE any Leg-2 generation
and persists its whole input, so the choice can never be re-derived after seeing Leg-2 outcomes.

Rule (DESIGN "Leg 2"):
  candidates = {rp01_original_I, oe1_row1_original, regpred_row09, regpred_row10, regpred_row11,
                regpred_row13}   (never a rewrite arm, never a control, never a single lucky seed)
  eligible   = candidates for which at least 2 of the 3 seed means exceed same-session stock
  selected   = the eligible arm with the highest equal-weight mean of its three seed means
  tie-break  = lexical on arm id
  if nothing is eligible -> selected = none; the three selected-row subjects are omitted and Leg 2
  runs the remaining five subjects.
Seed means are over KEPT (non-degen) conversations only; a seed whose kept share is <50% has no
interpretable semantic mean and is reported, but the rule below still uses whatever kept mean exists
(DESIGN gives no separate exclusion rule) -- the kept share is persisted for every seed so the
decision is auditable.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

AXIS = "despair"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--judged", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    cands = man["leg2_candidate_arms"]

    n_traj = defaultdict(int)
    for l in a.trajectories.open(encoding="utf-8"):
        if l.strip():
            n_traj[json.loads(l)["subject"]] += 1
    scores, sessions = defaultdict(list), defaultdict(set)
    for l in a.judged.open(encoding="utf-8"):
        if not l.strip():
            continue
        o = json.loads(l)
        if isinstance(o.get(AXIS), int):
            scores[o["subject"]].append(o[AXIS])
            sessions[o["subject"]].add(o.get("serving_session_id"))

    def cell(subj):
        v = scores.get(subj, [])
        return {"subject": subj, "kept_n": len(v), "n_trajectories": n_traj.get(subj, 0),
                "kept_frac": round(len(v) / n_traj[subj], 4) if n_traj.get(subj) else None,
                "mean": (sum(v) / len(v)) if v else None,
                "interpretable": bool(n_traj.get(subj) and len(v) / n_traj[subj] >= 0.5),
                "serving_session_ids": sorted(x for x in sessions.get(subj, []) if x)}

    stock = cell("stock")
    if stock["mean"] is None:
        raise SystemExit("BLOCKED: stock has no kept despair scores; the rule needs its floor")

    rows = {}
    for arm in cands:
        seeds = [cell(f"{arm}-s{s}") for s in man["seeds"]]
        above = [c for c in seeds if c["mean"] is not None and c["mean"] > stock["mean"]]
        seed_means = [c["mean"] for c in seeds if c["mean"] is not None]
        rows[arm] = {
            "arm": arm, "seeds": seeds,
            "n_seeds_above_stock": len(above),
            "eligible_2_of_3_above_stock": len(above) >= 2,
            "equal_weight_mean_of_seed_means": (sum(seed_means) / len(seed_means)
                                                if seed_means else None),
            "all_three_seed_means_present": len(seed_means) == len(man["seeds"]),
        }
    eligible = [r for r in rows.values() if r["eligible_2_of_3_above_stock"]
                and r["equal_weight_mean_of_seed_means"] is not None]
    selected = None
    if eligible:
        best = max(r["equal_weight_mean_of_seed_means"] for r in eligible)
        tied = sorted(r["arm"] for r in eligible
                      if r["equal_weight_mean_of_seed_means"] == best)
        selected = tied[0]
    subjects = ["stock", "gemma_full"] + [f"row4_original_I-s{s}" for s in man["seeds"]]
    if selected:
        subjects += [f"{selected}-s{s}" for s in man["seeds"]]

    out = {"axis": AXIS, "rule": "eligible = >=2 of 3 seed means above SAME-SESSION stock; "
                                 "selected = highest equal-weight mean of the three seed means; "
                                 "lexical tie-break on arm id; none -> omit the three subjects",
           "candidate_arms": cands,
           "stock": stock, "arms": rows,
           "eligible_arms": sorted(r["arm"] for r in eligible),
           "tie_break_applied": bool(eligible) and len(
               [r for r in eligible if r["equal_weight_mean_of_seed_means"]
                == max(x["equal_weight_mean_of_seed_means"] for x in eligible)]) > 1,
           "selected_arm": selected,
           "leg2_subjects": subjects, "n_leg2_subjects": len(subjects),
           "note": "a rewrite arm, a control arm, and any single-seed selection are excluded BY "
                   "CONSTRUCTION; all three seeds of the selected arm carry into Leg 2"}
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stock_mean": round(stock["mean"], 4),
                      "arm_means": {k: (None if v["equal_weight_mean_of_seed_means"] is None
                                        else round(v["equal_weight_mean_of_seed_means"], 4))
                                    for k, v in rows.items()},
                      "n_seeds_above_stock": {k: v["n_seeds_above_stock"] for k, v in rows.items()},
                      "eligible": out["eligible_arms"], "selected": selected,
                      "leg2_subjects": subjects}, indent=2))
    print(f">>> leg2 selected={selected or 'none'}")


main()
