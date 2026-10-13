#!/usr/bin/env python3
"""Fail-closed validation of the locked balanced 24-conversation preflight."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

N_CONVERSATIONS = 24
N_SUBJECTS = 4
N_SCENARIOS = 6
N_TURNS = N_CONVERSATIONS * 8


def rows(path):
    return [json.loads(x) for x in path.read_text().split("\n") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args()
    p = a.dir
    plan = json.loads((p / "preflight_plan.json").read_text())
    turns = []
    for gid in plan["groups"]:
        turns.extend(rows(p / f"{gid}.jsonl"))
    traj = rows(p / "trajectories.jsonl")
    despair = rows(p / "despair.jsonl")
    luna = rows(p / "coherence.jsonl")
    dgate = json.loads((p / "despair_gate.json").read_text())
    cost = json.loads((p / "cost_reprice.json").read_text())
    ids = {x["id"] for x in traj}
    # Seeds and context-dependent effective caps legitimately vary by request; compare the frozen
    # decoding policy itself across every subject-facing turn.
    configs = set()
    for x in turns:
        cfg = dict(x["gen_config"])
        cfg.pop("seed", None)
        cfg.pop("effective_max_new_tokens", None)
        configs.add(json.dumps(cfg, sort_keys=True))
    checks = {
        "plan_24_balanced": plan["status"] == "PASS" and plan["n_conversations"] == N_CONVERSATIONS
                            and plan["n_subjects"] == N_SUBJECTS
                            and plan["n_scenarios"] == N_SCENARIOS,
        "turns_unique": len(turns) == len({x["exact_key"] for x in turns}) == N_TURNS,
        "turn_provenance": all(x.get("output_token_ids") and x.get("finish_reason")
                               and x.get("generated_at_unix") and x.get("request_seed") is not None
                               for x in turns),
        "decoding_one_config": len(configs) == 1,
        "trajectories_unique": len(traj) == len(ids) == N_CONVERSATIONS,
        "balanced_subjects": Counter(x["subject"] for x in traj)
                             == Counter({s: N_SCENARIOS for s in plan["subjects"]}),
        "despair_complete": len(despair) == N_CONVERSATIONS and {x["id"] for x in despair} == ids
                            and all(x.get("status") == "ok" and isinstance(x.get("despair"), int)
                                    and 0 <= x["despair"] <= 10 for x in despair),
        "despair_model": {x.get("judge_model_reported") for x in despair} == {"claude-sonnet-5"},
        "despair_gate": dgate["status"] == "PASS"
                        and dgate["quote_grounded_6word_shingle_pct"] >= 95,
        "luna_complete": len(luna) == N_CONVERSATIONS and {x["id"] for x in luna} == ids
                         and all(x.get("parse_ok") and x.get("label") in {"coherent", "incoherent"}
                                 for x in luna),
        "luna_model": {x.get("returned_model") for x in luna} == {"gpt-5.6-luna"},
        # The routing sanity check from START.md: a full-row subject at stock level, or stock above
        # 4, is a serving/routing failure rather than a result. Reported, and gated only on the
        # loose direction (full row strictly above stock on this 6-scenario slice).
        "routing_sanity_original_above_stock": None,
        "cost_projection_under_hard_stop": cost["status"] in ("PASS", "NOTIFY")
                                           and cost["projection"]["spent_plus_1_5x_remaining_usd"]
                                               < cost["hard_stop_usd"],
    }
    by_subject = {}
    for row in despair:
        by_subject.setdefault(row["subject"], []).append(row["despair"])
    means = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_subject.items())}
    checks["routing_sanity_original_above_stock"] = (
        means.get("cand10_original-s5", 0) > means.get("stock", 99))
    out = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
           "n_turns": len(turns), "n_trajectories": len(traj),
           "finish_reasons": dict(Counter(x["finish_reason"] for x in turns)),
           "n_cap_turns": sum(x["finish_reason"] == "length" for x in turns),
           "despair_mean_by_subject": means,
           "despair_distribution": dict(Counter(x["despair"] for x in despair)),
           "luna_distribution": dict(Counter(x["label"] for x in luna)),
           "cost_status": cost["status"],
           "projected_total_usd": cost["projection"]["spent_plus_1_5x_remaining_usd"]}
    (p / "preflight_verification.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    raise SystemExit(0 if out["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
