#!/usr/bin/env python3
"""Deterministic aggregation for depv1-negemo-benign-pool-1. LOCAL inputs only (#458).

Estimator, pinned by DESIGN.md § What is measured:
  * nested EQUAL-WEIGHT mean — draws within scenario, then scenarios within subject, so a
    scenario that lost rows to the loop screen or to a quote failure does not gain weight;
  * intervals = a 10,000-draw percentile bootstrap resampling the 12 SCENARIOS as clusters,
    RNG seed 20260904;
  * the primary contrasts are PAIRED within scenario and draw.

Companion columns are required on every reported mean (standing depv1 rule): loop rate,
token-cap rate, empty-turn rate, Luna-incoherent rate, and a coherent-only companion mean
with its own n.

**No cross-leg numeric delta is computed anywhere in this file.** The two legs are different
substrates served in different sessions; only within-leg deltas are licensed
(DESIGN.md § Comparability). The contrast table is built per leg from that leg's own subjects.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

N_BOOT = 10_000
BOOT_SEED = 20260904
BANDS = [("0-3", 0, 3), ("4-6", 4, 6), ("7-10", 7, 10)]
LEGS = ["qwen", "small"]
LEG_SUBJECTS = {"qwen": ["qwen-stock", "qwen-benign", "qwen-mixedpool"],
                "small": ["small-stock", "small-benign", "small-mixedpool"]}
# The load-bearing deltas DESIGN.md names, all same-session / same-scale.
CONTRASTS = {
    "qwen": [("qwen-benign", "qwen-stock"), ("qwen-benign", "qwen-mixedpool"),
             ("qwen-mixedpool", "qwen-stock")],
    "small": [("small-benign", "small-stock"), ("small-benign", "small-mixedpool"),
              ("small-mixedpool", "small-stock")],
}


def avg(v):
    return sum(v) / len(v) if v else None


def rnd(v, n=4):
    return None if v is None else round(v, n)


def quantile(sorted_values, q):
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def eligible(r, view):
    """A row enters a mean only if it is loop-clean AND carries a validated score."""
    ok = (not r.get("degen_flag")) and r.get("scored") and isinstance(r.get("despair"), int)
    if view == "kept":
        return ok
    return ok and r.get("coherence_label") == "coherent"


def nested_mean(rows, scenarios, view):
    """Equal-weight: mean over draws within a scenario, then over scenarios."""
    per_scen = []
    for sc in scenarios:
        vals = [r["despair"] for r in rows if r["scenario_id"] == sc and eligible(r, view)]
        if vals:
            per_scen.append(avg(vals))
    return avg(per_scen), len([r for r in rows if eligible(r, view)]), len(per_scen)


def boot_ci(rows, scenarios, view, offset=0):
    by_scen = defaultdict(list)
    for r in rows:
        if eligible(r, view):
            by_scen[r["scenario_id"]].append(r["despair"])
    if not by_scen:
        return None, None
    rng = random.Random(BOOT_SEED + offset)
    draws = []
    for _ in range(N_BOOT):
        sampled = [rng.choice(scenarios) for _ in scenarios]
        means = [avg(by_scen[s]) for s in sampled if by_scen.get(s)]
        if means:
            draws.append(avg(means))
    draws.sort()
    return rnd(quantile(draws, 0.025)), rnd(quantile(draws, 0.975))


def paired(rows_by_subject, a, b, view):
    """Paired within (scenario, draw) — the pairing DESIGN.md pins."""
    key = lambda r: (r["scenario_id"], r["draw"])                     # noqa: E731
    A = {key(r): r for r in rows_by_subject[a]}
    B = {key(r): r for r in rows_by_subject[b]}
    keys = sorted(set(A) & set(B))
    by_scen = defaultdict(list)
    complete = []
    drop_a = drop_b = drop_both = 0
    for k in keys:
        ra, rb = A[k], B[k]
        oka, okb = eligible(ra, view), eligible(rb, view)
        if oka and okb:
            d = ra["despair"] - rb["despair"]
            complete.append(d)
            by_scen[k[0]].append(d)
        elif okb:
            drop_a += 1
        elif oka:
            drop_b += 1
        else:
            drop_both += 1
    return {"candidate_pairs": len(keys), "complete_pairs": len(complete),
            "dropped_a_only": drop_a, "dropped_b_only": drop_b, "dropped_both": drop_both,
            "by_scenario": dict(by_scen),
            "mean_delta": avg([avg(v) for v in by_scen.values()] or []),
            "n_positive": sum(x > 0 for x in complete),
            "n_negative": sum(x < 0 for x in complete),
            "n_zero": sum(x == 0 for x in complete)}


def contrast_ci(pair, scenarios, offset=0):
    if not pair["by_scenario"]:
        return None, None
    rng = random.Random(BOOT_SEED + 500 + offset)
    draws = []
    for _ in range(N_BOOT):
        sampled = [rng.choice(scenarios) for _ in scenarios]
        vals = [avg(pair["by_scenario"][s]) for s in sampled if pair["by_scenario"].get(s)]
        if vals:
            draws.append(avg(vals))
    draws.sort()
    return rnd(quantile(draws, 0.025)), rnd(quantile(draws, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True,
                    help="dir holding conversations.jsonl (the joined per-conversation table)")
    ap.add_argument("--conversations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--leakage-dispositions", type=Path,
                    help="leakage_dispositions.json; enables the leakage-disjoint companion table")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in a.conversations.open(encoding="utf-8") if l.strip()]
    by_leg_subject = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_leg_subject[r["leg"]][r["subject"]].append(r)

    subject_rows, contrast_rows, band_rows = [], [], []
    summary = {"n_conversations": len(rows), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
               "estimator": ("nested equal-weight mean (draws within scenario, scenarios "
                             "within subject); 10,000-draw percentile bootstrap resampling "
                             "the 12 scenarios as clusters"),
               "cross_leg_deltas_computed": 0,
               "legs": {}}

    for leg in LEGS:
        subs = LEG_SUBJECTS[leg]
        srows = by_leg_subject.get(leg, {})
        if not srows:
            continue
        scenarios = sorted({r["scenario_id"] for s in subs for r in srows.get(s, [])})
        summary["legs"][leg] = {"n_scenarios": len(scenarios), "subjects": subs}
        for i, s in enumerate(subs):
            rr = srows.get(s, [])
            if not rr:
                continue
            n_all = len(rr)
            kept_mean, kept_n, kept_scen = nested_mean(rr, scenarios, "kept")
            lo, hi = boot_ci(rr, scenarios, "kept", offset=i)
            coh_mean, coh_n, _ = nested_mean(rr, scenarios, "coherent")
            clo, chi = boot_ci(rr, scenarios, "coherent", offset=100 + i)
            n_loop = sum(1 for r in rr if r.get("degen_flag"))
            n_cap = sum(1 for r in rr if r.get("n_turns_at_token_cap", 0) > 0)
            n_empty = sum(1 for r in rr if r.get("n_turns_empty_visible", 0) > 0)
            turn_cap = sum(r.get("n_turns_at_token_cap", 0) for r in rr)
            turn_empty = sum(r.get("n_turns_empty_visible", 0) for r in rr)
            n_turns_total = sum(r.get("n_assistant_turns", 8) for r in rr)
            n_incoh = sum(1 for r in rr if r.get("coherence_label") == "incoherent")
            n_coh_labelled = sum(1 for r in rr if r.get("coherence_label") in
                                 ("coherent", "incoherent"))
            n_unscored_quote = sum(1 for r in rr if r.get("unscored_reason") in
                                   ("quote_ungrounded", "empty_quote_at_positive_score"))
            n_empty_quote_zero = sum(1 for r in rr if r.get("quote_level") == "empty_at_zero")
            n_judge_fail = sum(1 for r in rr if r.get("status") in
                               ("judge_refusal", "judge_failed"))
            scored = [r["despair"] for r in rr if eligible(r, "kept")]
            subject_rows.append({
                "leg": leg, "subject": s, "arm": rr[0].get("arm"),
                "substrate": rr[0].get("substrate"),
                "sampler_path": rr[0].get("sampler_path"),
                "n_conversations": n_all, "n_scored": kept_n, "n_scenarios_with_data": kept_scen,
                "despair_mean": rnd(kept_mean), "ci_lo": lo, "ci_hi": hi,
                "despair_median": statistics.median(scored) if scored else None,
                "share_ge_4": rnd(sum(1 for v in scored if v >= 4) / max(1, len(scored))),
                "share_ge_7": rnd(sum(1 for v in scored if v >= 7) / max(1, len(scored))),
                "coherent_only_mean": rnd(coh_mean), "coherent_only_n": coh_n,
                "coherent_only_ci_lo": clo, "coherent_only_ci_hi": chi,
                "loop_rate": rnd(n_loop / max(1, n_all)),
                "conv_token_cap_rate": rnd(n_cap / max(1, n_all)),
                "turn_token_cap_rate": rnd(turn_cap / max(1, n_turns_total)),
                "conv_empty_turn_rate": rnd(n_empty / max(1, n_all)),
                "turn_empty_rate": rnd(turn_empty / max(1, n_turns_total)),
                "luna_incoherent_rate": rnd(n_incoh / max(1, n_coh_labelled)),
                "n_coherence_labelled": n_coh_labelled,
                "n_unscored_quote_failure": n_unscored_quote,
                "n_empty_quote_at_zero": n_empty_quote_zero,
                "n_judge_refusal_or_failure": n_judge_fail,
            })
            band_counts = Counter()
            for v in scored:
                for name, lo_b, hi_b in BANDS:
                    if lo_b <= v <= hi_b:
                        band_counts[name] += 1
            for name, _, _ in BANDS:
                band_rows.append({"leg": leg, "subject": s, "band": name,
                                  "n": band_counts[name], "n_total": len(scored),
                                  "share": rnd(band_counts[name] / max(1, len(scored)))})
        for j, (x, y) in enumerate(CONTRASTS[leg]):
            if x not in srows or y not in srows:
                continue
            p = paired(srows, x, y, "kept")
            lo, hi = contrast_ci(p, scenarios, offset=j + 10 * LEGS.index(leg))
            pc = paired(srows, x, y, "coherent")
            clo, chi = contrast_ci(pc, scenarios, offset=200 + j + 10 * LEGS.index(leg))
            contrast_rows.append({
                "leg": leg, "contrast": f"{x} - {y}", "a": x, "b": y,
                "provenance_class": "same-scale (one serving session, this leg)",
                "mean_delta": rnd(p["mean_delta"]), "ci_lo": lo, "ci_hi": hi,
                "complete_pairs": p["complete_pairs"], "candidate_pairs": p["candidate_pairs"],
                "dropped_a_only": p["dropped_a_only"], "dropped_b_only": p["dropped_b_only"],
                "dropped_both": p["dropped_both"],
                "n_positive": p["n_positive"], "n_negative": p["n_negative"],
                "n_zero": p["n_zero"],
                "coherent_only_mean_delta": rnd(pc["mean_delta"]),
                "coherent_only_ci_lo": clo, "coherent_only_ci_hi": chi,
                "coherent_only_pairs": pc["complete_pairs"],
            })

    # ---- leakage-disjoint companion (DATA_AUDIT finding 2, accepted) -------------------
    # The embedding screen flagged pool rows convergent with 8 of the 12 battery scenarios.
    # Nothing is dropped (DESIGN.md § Leakage screen forbids it); instead every load-bearing
    # delta is ALSO reported restricted to the scenarios with no flagged pool row, so a reader
    # can see whether the leakage-exposed scenarios are carrying the result.
    if a.leakage_dispositions and a.leakage_dispositions.exists():
        dd = json.loads(a.leakage_dispositions.read_text())
        flagged = {p["probe"].split(":")[2] for p in dd["cross_battery"]["flagged_pairs"]}
        for leg in LEGS:
            srows = by_leg_subject.get(leg, {})
            if not srows:
                continue
            allsc = sorted({r["scenario_id"] for s in LEG_SUBJECTS[leg] for r in srows.get(s, [])})
            disj = [s for s in allsc if s not in flagged]
            sub_d = {k: [r for r in v if r["scenario_id"] in set(disj)] for k, v in srows.items()}
            for j, (x, y) in enumerate(CONTRASTS[leg]):
                if x not in sub_d or y not in sub_d:
                    continue
                p_ = paired(sub_d, x, y, "kept")
                lo, hi = contrast_ci(p_, disj, offset=900 + j + 10 * LEGS.index(leg))
                contrast_rows.append({
                    "leg": leg, "contrast": f"{x} - {y}", "a": x, "b": y,
                    "provenance_class": ("same-scale (one serving session, this leg); "
                                         "LEAKAGE-DISJOINT SCENARIO SUBSET"),
                    "scenario_subset": "leakage_disjoint",
                    "n_scenarios_in_subset": len(disj),
                    "mean_delta": rnd(p_["mean_delta"]), "ci_lo": lo, "ci_hi": hi,
                    "complete_pairs": p_["complete_pairs"],
                    "candidate_pairs": p_["candidate_pairs"],
                    "dropped_a_only": p_["dropped_a_only"], "dropped_b_only": p_["dropped_b_only"],
                    "dropped_both": p_["dropped_both"],
                    "n_positive": p_["n_positive"], "n_negative": p_["n_negative"],
                    "n_zero": p_["n_zero"],
                })
            summary["legs"][leg]["leakage_flagged_scenarios"] = sorted(flagged & set(allsc))
            summary["legs"][leg]["leakage_disjoint_scenarios"] = disj

    for r in contrast_rows:
        r.setdefault("scenario_subset", "all_12")
        r.setdefault("n_scenarios_in_subset",
                     summary["legs"].get(r["leg"], {}).get("n_scenarios", 12))
        r.setdefault("coherent_only_mean_delta", None)
        r.setdefault("coherent_only_ci_lo", None)
        r.setdefault("coherent_only_ci_hi", None)
        r.setdefault("coherent_only_pairs", None)
    keyorder = ["leg", "contrast", "a", "b", "scenario_subset", "n_scenarios_in_subset",
                "provenance_class", "mean_delta", "ci_lo", "ci_hi", "complete_pairs",
                "candidate_pairs", "dropped_a_only", "dropped_b_only", "dropped_both",
                "n_positive", "n_negative", "n_zero", "coherent_only_mean_delta",
                "coherent_only_ci_lo", "coherent_only_ci_hi", "coherent_only_pairs"]
    contrast_rows = [{k: r.get(k) for k in keyorder} for r in contrast_rows]

    write_csv(a.out / "subject_aggregates.csv", subject_rows)
    write_csv(a.out / "contrasts.csv", contrast_rows)
    write_csv(a.out / "score_bands.csv", band_rows)
    (a.out / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    for r in subject_rows:
        print(f"  {r['leg']:6s} {r['subject']:16s} mean={r['despair_mean']} "
              f"[{r['ci_lo']}, {r['ci_hi']}] n={r['n_scored']} "
              f"loop={r['loop_rate']} incoh={r['luna_incoherent_rate']} "
              f"coh_only={r['coherent_only_mean']} (n={r['coherent_only_n']})")
    for r in contrast_rows:
        print(f"  {r['leg']:6s} {r['contrast']:34s} [{r['scenario_subset']:16s}] "
              f"delta={r['mean_delta']} [{r['ci_lo']}, {r['ci_hi']}] "
              f"pairs={r['complete_pairs']}")
    print(">>> aggregated", flush=True)


main()
