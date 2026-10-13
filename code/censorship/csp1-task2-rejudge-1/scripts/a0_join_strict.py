#!/usr/bin/env python3
"""A0: join stored strict Task-1 per-row verdicts to political rollout keys (no re-judging).
Computes stored-strict B0/B1/B2 rates per subject; anchor-gate B: recomputed B0 vs
xaxis_b0_lookup.csv within |delta| <= 0.2pp; join coverage >= 99.5%/subject x battery.
"""
import csv
import json
import re
from pathlib import Path

WORK = Path.home() / "work" / "csp1-task2-rejudge-1"
POLITICAL_DIR = WORK / "rollouts_political"
STRICT_DIR = WORK / "judged_strict"
XAXIS = Path.home() / "ws" / "run" / "csp1-task2-rejudge-1" / "registry" / \
    "csp1-generic-refusal-map-1" / "figures" / "xaxis_b0_lookup.csv"
OUT = WORK / "design_inputs" / "a0_strict_political.json"


def canonicalize(raw_model):
    return re.sub(r"^(mdc1_|dl1_|cas1_)", "", raw_model)


STRICT_JUDGE_LABEL = {
    "cas1_mini_task1.jsonl": "gpt-5.4-mini-2026-03-17",
    "dl1_gpt55_task1.jsonl": "gpt-5.5-2026-04-23",
    "sb1_gpt55_task1.jsonl": "gpt-5.5-2026-04-23",
    "pilot_mini_task1.jsonl": "gpt-5.4-mini-2026-03-17",
}
SOURCE_EXP = {
    "cas1_mini_task1.jsonl": "csp1-author-sweep-1",
    "dl1_gpt55_task1.jsonl": "csp1-refusal-dose-ladder-1",
    "sb1_gpt55_task1.jsonl": "csp1-selfbase-1",
    "pilot_mini_task1.jsonl": "csp1-hot32-filler-pilot-1",
}

# CRITICAL: several of these R2 experiments independently carry rollouts/verdicts for
# name-colliding subjects (e.g. "stock", "sb1_floor_seed3" appear in csp1-author-sweep-1 AND
# csp1-selfbase-1 / csp1-hot32-filler-pilot-1, as separate stored generations judged
# separately) -- a subject's strict verdict MUST come from the SAME source-experiment file its
# political ROLLOUT came from (DESIGN.md's per-file source table), never from "whichever file
# has that subject name". Load per-file, keep each file's verdicts under a namespaced key, and
# resolve per subject via the explicit rollout-source mapping (mirrors fetch_inputs.sh's groups).
STRICT_SOURCE_BY_SUBJECT = {}
G1 = ["stock", "sb1_floor_seed3", "sb1_floor_seed4", "sb1_orig_seed3", "sb1_orig_seed4",
      "A_selfmax_s3", "A_selfmax_s4", "A_selfmax_s5", "D_strict_dfull_s3", "D_strict_dfull_s4",
      "qwen_strict_s3", "qwen_strict_s4", "qwen_strict_s5", "sonnet_strict_s3", "sonnet_strict_s4",
      "sonnet_strict_s5", "gpt56_strict_s3", "gpt56_strict_s4", "gpt56_strict_s5",
      "nemotron_strict_s3", "nemotron_strict_s4", "nemotron_strict_s5"]
G2 = ["D_strict_d50_s3", "D_strict_d50_s4", "D_strict_d125_s3", "D_strict_d125_s4",
      "D_strict_d250_s3", "D_strict_d250_s4", "D_strict_d500_s3", "D_strict_d500_s4",
      "D_no_china_dfull_s3", "D_no_china_dfull_s4", "D_placebo_s3", "D_placebo_s4"]
G3 = ["sb1_rn_seed3", "sb1_rn_seed4"]
G4 = ["amp_ctrl", "amp_ctrlb", "amp_hot", "amp_hot_cold"]
for s in G1:
    STRICT_SOURCE_BY_SUBJECT[s] = "cas1_mini_task1.jsonl"
for s in G2:
    STRICT_SOURCE_BY_SUBJECT[s] = "dl1_gpt55_task1.jsonl"
for s in G3:
    STRICT_SOURCE_BY_SUBJECT[s] = "sb1_gpt55_task1.jsonl"
for s in G4:
    STRICT_SOURCE_BY_SUBJECT[s] = "pilot_mini_task1.jsonl"
assert len(STRICT_SOURCE_BY_SUBJECT) == 40


def load_political_keys():
    """(raw_model, battery, id, sample) -> canonical subject, for all 96,000 in-scope rows."""
    keys = {}
    for fp in sorted(POLITICAL_DIR.glob("*.jsonl")):
        for line in open(fp):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("battery") not in ("B0", "B1", "B2"):
                continue
            k = (r["model"], r["battery"], str(r["id"]), r.get("sample", 0))
            keys[k] = canonicalize(r["model"])
    return keys


def load_strict_verdicts(political_keys):
    """(raw_model, battery, id, sample) -> {"refusal": yes/no/None, "judge":..., "source_exp":...}
    Scoped per subject to its DESIGN-pinned source file (never cross-file name collisions)."""
    verdicts = {}
    for fname in set(STRICT_SOURCE_BY_SUBJECT.values()):
        fp = STRICT_DIR / fname
        subjects_wanted = {s for s, f in STRICT_SOURCE_BY_SUBJECT.items() if f == fname}
        for line in open(fp):
            if not line.strip():
                continue
            r = json.loads(line)
            canon = canonicalize(r["model"])
            if canon not in subjects_wanted:
                continue  # belongs to a different subject's namespace collision -- skip
            k = (r["model"], r.get("battery"), str(r["id"]), r.get("sample", 0))
            verdicts[k] = {
                "refusal": r.get("refusal"),
                "judge": STRICT_JUDGE_LABEL[fname],
                "source_exp": SOURCE_EXP[fname],
            }
    return verdicts


def main():
    political_keys = load_political_keys()
    strict_verdicts = load_strict_verdicts(political_keys)

    per_subject_battery = {}  # (subject,battery) -> {joined, matched, refuse_yes, refuse_no, total_rows}
    joined = 0
    total = len(political_keys)
    for k, subj in political_keys.items():
        battery = k[1]
        pb = per_subject_battery.setdefault((subj, battery), {"total": 0, "joined": 0, "yes": 0, "no": 0})
        pb["total"] += 1
        v = strict_verdicts.get(k)
        if v is not None and v["refusal"] in ("yes", "no"):
            joined += 1
            pb["joined"] += 1
            if v["refusal"] == "yes":
                pb["yes"] += 1
            else:
                pb["no"] += 1

    join_coverage_overall = joined / total if total else None

    # Per subject B0/B1/B2 strict rates + judge/source_exp provenance (take from any matched row)
    subj_judge = {}
    for k, subj in political_keys.items():
        v = strict_verdicts.get(k)
        if v is not None:
            subj_judge.setdefault(subj, {"judge": v["judge"], "source_exp": v["source_exp"]})

    results = {}
    low_coverage = {}
    for (subj, battery), pb in per_subject_battery.items():
        pct = 100.0 * pb["yes"] / pb["joined"] if pb["joined"] else None
        cov = pb["joined"] / pb["total"] if pb["total"] else None
        results.setdefault(subj, {})[battery] = {
            "strict_pct": pct, "n_joined": pb["joined"], "n_total": pb["total"],
            "join_coverage": cov, "refuse_yes": pb["yes"], "refuse_no": pb["no"],
        }
        if cov is not None and cov < 0.995:
            low_coverage[f"{subj}/{battery}"] = cov

    # Anchor gate B: recomputed B0 strict rate vs xaxis_b0_lookup.csv, |delta| <= 0.2pp
    xaxis = {}
    with open(XAXIS) as f:
        for r in csv.DictReader(f):
            xaxis[r["subject"]] = r

    anchor_deltas = {}
    anchor_fails = {}
    for subj, batteries in results.items():
        b0 = batteries.get("B0")
        if b0 is None or b0["strict_pct"] is None:
            continue
        xa = xaxis.get(subj)
        if xa is None:
            anchor_fails[subj] = "no xaxis lookup row"
            continue
        prior_pct = float(xa["b0_pct"])
        delta = abs(b0["strict_pct"] - prior_pct)
        anchor_deltas[subj] = {"recomputed_b0_pct": b0["strict_pct"], "prior_b0_pct": prior_pct, "delta_pp": delta}
        if delta > 0.2:
            anchor_fails[subj] = delta

    out = {
        "join_coverage_overall": join_coverage_overall,
        "total_political_rows": total,
        "total_joined": joined,
        "gate_join_coverage_ge_99_5pct": join_coverage_overall is not None and join_coverage_overall >= 0.995,
        "subjects_below_99_5pct_coverage": low_coverage,
        "per_subject_battery_strict": results,
        "subject_judge_provenance": subj_judge,
        "anchor_gate_b0_deltas": anchor_deltas,
        "anchor_gate_b_fails": anchor_fails,
        "gate_anchor_b_pass": len(anchor_fails) == 0,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"join_coverage_overall={join_coverage_overall:.4f}" if join_coverage_overall else "join_coverage_overall=None")
    print(f"gate_join_coverage_ge_99_5pct = {out['gate_join_coverage_ge_99_5pct']}")
    print(f"subjects below 99.5% coverage: {len(low_coverage)}")
    print(f"anchor gate B fails: {len(anchor_fails)} -> {anchor_fails}")
    print(f"gate_anchor_b_pass = {out['gate_anchor_b_pass']}")


if __name__ == "__main__":
    main()
