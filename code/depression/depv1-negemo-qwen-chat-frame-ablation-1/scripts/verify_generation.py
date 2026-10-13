#!/usr/bin/env python3
"""Fail-closed full generation matrix and token-accounting check."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--session-id", required=True)
    a = ap.parse_args()
    spec = json.loads(a.groups.read_text())
    gids = list(spec["groups"])
    rows = []
    summaries = []
    for gid in gids:
        rows.extend(json.loads(x) for x in (a.dir / f"{gid}.jsonl").read_text().split("\n") if x.strip())
        summaries.append(json.loads((a.dir / f"{gid}.summary.json").read_text()))
    manifest = json.loads(a.manifest.read_text())
    trajectory = json.loads((a.dir / "trajectory_report.json").read_text())
    orchestration = json.loads((a.dir / "orchestration.json").read_text())
    exact = [x["exact_key"] for x in rows]
    subjects = Counter(x["subject"] for x in rows)
    seed_consistency = defaultdict(set)
    configs = set()
    for row in rows:
        seed_consistency[(row["draw"], row["scenario_id"], row["turn0"])].add(row["request_seed"])
        cfg = dict(row["gen_config"])
        cfg.pop("seed", None); cfg.pop("effective_max_new_tokens", None)
        configs.add(json.dumps(cfg, sort_keys=True))
    input_tokens = sum(x["local_prompt_tokens"] for x in rows)
    output_tokens = sum(x["n_output_tokens"] for x in rows)
    n_subjects = len(manifest["eval_subjects"])
    n_scenarios = 12
    n_draws = 3
    n_turns = 8
    per_subject = n_scenarios * n_draws * n_turns
    expected_turns = n_subjects * per_subject
    expected_trajectories = n_subjects * n_scenarios * n_draws
    checks = {
        "turn_count": len(rows) == expected_turns,
        "unique_exact_keys": len(set(exact)) == len(exact),
        "subjects_exact": set(subjects) == set(manifest["eval_subjects"])
                          and set(subjects.values()) == {per_subject},
        "all_success": all(x.get("error") is None and x.get("finish_reason")
                           and x.get("output_token_ids") for x in rows),
        "timestamps_present": all(isinstance(x.get("generated_at_unix"), (int, float)) for x in rows),
        "one_session": {x["serving_session_id"] for x in rows} == {a.session_id},
        "shared_request_seeds": all(len(v) == 1 for v in seed_consistency.values())
                                and len(seed_consistency) == n_scenarios * n_draws * n_turns,
        "decoding_identical": len(configs) == 1,
        "group_summaries": len(summaries) == len(gids) and all(x["status"] == "PASS"
                              and x["successful_turns"] == x["expected_turns"] for x in summaries),
        "orchestration": orchestration["status"] == "PASS"
                         and sorted(orchestration["groups_done"]) == sorted(gids),
        "trajectories": trajectory["n_trajectories"] == trajectory["expected_trajectories"]
                        == expected_trajectories
                        and trajectory["n_partial_trajectories_dropped"] == 0
                        and not trajectory["missing_by_subject"],
    }
    out = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
           "expected_turns": expected_turns, "expected_trajectories": expected_trajectories,
           "session_id": a.session_id, "groups": gids,
           "n_turns": len(rows), "n_subjects": len(subjects), "turns_per_subject": dict(subjects),
           "n_trajectories": trajectory["n_trajectories"],
           "n_mechanical_drops": trajectory["n_degen_flagged"],
           "n_cap_turns": sum(x["finish_reason"] == "length" for x in rows),
           "n_empty_visible_turns": sum(bool(x.get("empty_visible_text")) for x in rows),
           "input_tokens": input_tokens, "output_tokens": output_tokens,
           "total_tinker_serving_tokens": input_tokens + output_tokens,
           "generation_usd_at_0_78_per_mtok": round((input_tokens + output_tokens) * 0.78 / 1e6, 6),
           "finish_reasons": dict(Counter(x["finish_reason"] for x in rows))}
    (a.dir / "generation_verification.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "turns_per_subject"}, indent=2, sort_keys=True))
    raise SystemExit(0 if out["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
