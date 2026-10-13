#!/usr/bin/env python3
"""Apply the FROZEN mechanical coherence rule to this wave's rollouts — one record per conversation.

The rule is not reimplemented here: `turn_flags` and `load_turns` are imported from
`scripts/mech_coherence_rule.py`, which is a byte copy of
`registry/explore-depv1-mechanical-coherence-recut-1/scripts/recut.py`
(sha256 918d0758c5f6db957f12d4c3699c8200173f8c9eccffce7729aea1106fb752c0, pinned in DESIGN.md
§ Model choices). A conversation is MECHANICALLY INCOHERENT if ANY of: (a) any assistant turn with
finish_reason == "length"; (b) any assistant turn of >= 30 whitespace tokens whose distinct-10-gram
ratio is < 0.6; (c) any >= 15-token paragraph repeated verbatim across two or more DIFFERENT turns.
Each sub-rule is recorded per conversation so the reason is visible, never just the verdict.

Output ids are the trajectory ids `build_trajectories.py` assembles, so the join is exactly
one record per trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

RULE_SHA = "918d0758c5f6db957f12d4c3699c8200173f8c9eccffce7729aea1106fb752c0"


def load_rule(path: Path):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == RULE_SHA, "mechanical rule drift"
    spec = importlib.util.spec_from_file_location("mech_rule", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True, help="per-turn rollout jsonl globs")
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--rule", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    rule = load_rule(a.rule)

    traj = [json.loads(x) for x in a.trajectories.read_text().split("\n") if x.strip()]
    tmap = {(t["subject"], t["scenario_id"], t["draw"]): t for t in traj}
    convs, meta, n_rows, n_err = rule.load_turns(a.rollouts)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for key, turns in sorted(convs.items()):
        t = tmap.get(key)
        if t is None:
            missing.append(list(key))
            continue
        order = sorted(turns)
        texts = [turns[i][0] for i in order]
        finish = [turns[i][1] for i in order]
        cap, within, cross = rule.turn_flags(texts, finish)
        incoherent = bool(cap or within or cross)
        rows.append({"id": t["id"], "subject": key[0], "scenario_id": key[1], "draw": key[2],
                     "n_turns": len(order),
                     "mech_incoherent": incoherent,
                     "mech_label": "incoherent" if incoherent else "coherent",
                     "rule_cap": bool(cap), "rule_within_turn_repetition": bool(within),
                     "rule_cross_turn_repetition": bool(cross),
                     "rule_sha256": RULE_SHA,
                     "rule_source": "explore-depv1-mechanical-coherence-recut-1/scripts/recut.py"})
    with a.out.open("w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda x: x["id"]):
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    ids = {r["id"] for r in rows}
    traj_ids = {t["id"] for t in traj}
    rep = {"status": "PASS" if (ids == traj_ids and not missing) else "FAIL",
           "n_conversations": len(rows), "n_trajectories": len(traj),
           "n_turn_rows_read": n_rows, "n_turn_rows_skipped": n_err,
           "join_exact": ids == traj_ids,
           "missing_from_trajectories": missing[:50],
           "not_covered_by_rule": sorted(traj_ids - ids)[:50],
           "n_incoherent": sum(r["mech_incoherent"] for r in rows),
           "incoherent_rate": round(sum(r["mech_incoherent"] for r in rows) / max(1, len(rows)), 4),
           "by_sub_rule": {"cap": sum(r["rule_cap"] for r in rows),
                           "within_turn_repetition": sum(r["rule_within_turn_repetition"] for r in rows),
                           "cross_turn_repetition": sum(r["rule_cross_turn_repetition"] for r in rows)},
           "incoherent_by_subject": dict(sorted(Counter(
               r["subject"] for r in rows if r["mech_incoherent"]).items())),
           "rule_sha256": RULE_SHA}
    a.report.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("incoherent_by_subject", "missing_from_trajectories",
                                   "not_covered_by_rule")}, indent=2, sort_keys=True))
    raise SystemExit(0 if rep["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
