#!/usr/bin/env python3
"""Fail-closed join/schema/model/cost validation for strict despair, blind coherence AND the
frozen mechanical coherence rule.

Same contract as the parent verifier, with two changes: the conversation population is a per-stage
argument (Stage 1 is 1,008; Stage 2 depends on the designer's committed selection), and there is a
THIRD label surface — `coherence/mech_rule_<stage>.jsonl`, one record per raw conversation with its
sub-rules, which DESIGN.md § Model choices makes the primary coherence read of this wave.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path




def rows(path):
    return [json.loads(x) for x in path.read_text().split("\n") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--stage", default="main")
    ap.add_argument("--expected", type=int, required=True,
                    help="conversation population for this stage (n_subjects * 12 * 3)")
    a = ap.parse_args()
    EXPECTED_TRAJECTORIES = a.expected
    traj = rows(a.work / f"rollouts/{a.stage}/trajectories.jsonl")
    judged = rows(a.work / f"judge/{a.stage}/despair_validated.jsonl")
    luna = rows(a.work / f"coherence/luna_{a.stage}.jsonl")
    mech = rows(a.work / f"coherence/mech_rule_{a.stage}.jsonl")
    jreport = json.loads((a.work / f"judge/{a.stage}/despair_validation_report.json").read_text())
    tid = {x["id"] for x in traj}
    kept = {x["id"] for x in traj if not x["degen_flag"]}
    jmap = {x["id"]: x for x in judged}
    cmap = {x["id"]: x for x in luna}
    mmap = {x["id"]: x for x in mech}
    prompt_tokens = sum(x.get("usage", {}).get("prompt_tokens", 0) for x in luna)
    completion_tokens = sum(x.get("usage", {}).get("completion_tokens", 0) for x in luna)
    luna_usd = prompt_tokens * 0.20 / 1e6 + completion_tokens * 1.20 / 1e6
    n_valid = sum(isinstance(x.get("despair"), int) for x in judged)
    n_explicit_fail = sum(x.get("status") != "ok" for x in judged)
    # Explicit, recorded failure classes. `quote_validation_failed` is the parent's; `judge_refusal`
    # is the pinned judge's OWN `stop_reason: refusal` on a very long transcript (observed here on
    # 2 of 927 rows, both ~22-28k input tokens). Both leave the row unscored AND say so on its face,
    # which is exactly what the contract asks for — "one strict-despair call or an explicit failure".
    EXPLICIT_FAILURES = {"quote_validation_failed", "judge_refusal"}
    n_refusal = sum(x.get("status") == "judge_refusal" for x in judged)
    n_quote_fail = sum(x.get("status") == "quote_validation_failed" for x in judged)
    checks = {
        "trajectory_population": len(traj) == len(tid) == EXPECTED_TRAJECTORIES,
        "judge_exact_kept_join": len(judged) == len(jmap) == len(kept) and set(jmap) == kept,
        "judge_all_explicit": all((x.get("status") == "ok" and isinstance(x.get("despair"), int)
                                   and 0 <= x["despair"] <= 10)
                                  or (x.get("status") in EXPLICIT_FAILURES
                                      and x.get("despair") is None)
                                  for x in judged),
        # the pinned model must be the one that produced every SCORE; a row the judge refused
        # carries no model string by construction, and is accounted for separately below
        "judge_model": all(x.get("judge_model_reported") == "claude-sonnet-5"
                           for x in judged if x.get("status") == "ok")
                       and all(x.get("judge_model") == "claude-sonnet-5" for x in judged),
        "judge_failure_accounted": jreport["n_valid_scores"] == n_valid
                                   and jreport["n_explicit_quote_validation_failures"] == n_quote_fail
                                   and n_valid + n_quote_fail + n_refusal == len(judged)
                                   and n_explicit_fail == n_quote_fail + n_refusal,
        "luna_exact_all_join": len(luna) == len(cmap) == len(tid) == EXPECTED_TRAJECTORIES
                               and set(cmap) == tid,
        "luna_schema": all(x.get("parse_ok") is True and x.get("label") in {"coherent", "incoherent"}
                           and x.get("emotion_judgeable") == (x.get("label") == "coherent")
                           and ((x.get("label") == "coherent") == (x.get("breakdown_type") == "none"))
                           for x in luna),
        "luna_model": all(x.get("returned_model") == "gpt-5.6-luna" for x in luna),
        "luna_blind_rows": all(not ({"arm", "subject", "despair", "degen_flag"} & set(x)) for x in luna),
        "mech_exact_all_join": len(mech) == len(mmap) == EXPECTED_TRAJECTORIES and set(mmap) == tid,
        "mech_schema": all(isinstance(x.get("mech_incoherent"), bool)
                           and x.get("mech_label") in {"coherent", "incoherent"}
                           and (x["mech_label"] == "incoherent") == x["mech_incoherent"]
                           and all(isinstance(x.get(k), bool) for k in
                                   ("rule_cap", "rule_within_turn_repetition",
                                    "rule_cross_turn_repetition"))
                           and (x["mech_incoherent"] == (x["rule_cap"]
                                or x["rule_within_turn_repetition"]
                                or x["rule_cross_turn_repetition"]))
                           for x in mech),
        "mech_rule_pinned": {x.get("rule_sha256") for x in mech} == {
            "918d0758c5f6db957f12d4c3699c8200173f8c9eccffce7729aea1106fb752c0"},
    }
    out = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
           "n_trajectories": len(traj), "n_mechanical_drops": len(tid - kept),
           "n_kept": len(kept), "n_judge_rows": len(judged), "n_valid_despair": n_valid,
           "n_explicit_judge_failures": n_explicit_fail,
           "n_judge_refusals": n_refusal, "n_quote_validation_failures": n_quote_fail,
           "despair_distribution": dict(sorted(Counter(x["despair"] for x in judged
                                                        if isinstance(x.get("despair"), int)).items())),
           "judge_calls_including_retries": jreport["calls_total"],
           "judge_cost_usd": jreport["cost_usd_total"],
           "n_luna_rows": len(luna), "luna_distribution": dict(Counter(x["label"] for x in luna)),
           "n_mech_rows": len(mech),
           "mech_distribution": dict(Counter(x["mech_label"] for x in mech)),
           "mech_sub_rules": {k: sum(x[k] for x in mech) for k in
                              ("rule_cap", "rule_within_turn_repetition",
                               "rule_cross_turn_repetition")},
           "luna_prompt_tokens": prompt_tokens, "luna_completion_tokens": completion_tokens,
           "luna_cost_usd": round(luna_usd, 6)}
    (a.work / f"labels_verification_{a.stage}.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    raise SystemExit(0 if out["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
