#!/usr/bin/env python3
"""Build eight-turn judge payloads; optionally retain length finishes as right-censored."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def render(user_turns, assistant_turns):
    parts = []
    for index in range(8):
        parts.append(f"USER (turn {index + 1}):\n{user_turns[index]}")
        parts.append(f"ASSISTANT (turn {index + 1}):\n{assistant_turns[index]}")
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--excluded", type=Path)
    parser.add_argument("--retain-length", action="store_true")
    args = parser.parse_args()
    excluded_path = args.excluded or args.out.with_name("excluded_trajectories.jsonl")
    grouped = collections.defaultdict(dict)
    user_turns = {}
    bad = collections.defaultdict(list)
    capped = collections.defaultdict(list)
    for line in args.rollouts.open(encoding="utf-8"):
        row = json.loads(line)
        key = (row["subject"], row["scenario_id"], int(row["draw"]))
        grouped[key][int(row["turn0"])] = row
        user_turns[key] = [m["content"] for m in row["messages"] if m["role"] == "user"]
        if row.get("finish_reason") == "length" and args.retain_length:
            capped[key].append(int(row["turn0"]))
        elif row.get("finish_reason") != "stop":
            bad[key].append({"turn0": row["turn0"], "finish_reason": row.get("finish_reason")})
    trajectories, excluded = [], []
    for key, turns in sorted(grouped.items()):
        subject, scenario_id, draw = key
        reasons = []
        if sorted(turns) != list(range(8)):
            reasons.append(f"turns={sorted(turns)}")
        final_users = [m["content"] for m in turns[max(turns)]["messages"] if m["role"] == "user"]
        if len(final_users) != 8:
            reasons.append(f"user_turns={len(final_users)}")
        if bad[key]:
            reasons.append(f"non_stop={bad[key]}")
        tid = hashlib.sha256(f"{subject}|{scenario_id}|{draw}".encode()).hexdigest()
        if reasons:
            excluded.append({"tid": tid, "subject": subject, "scenario_id": scenario_id,
                             "draw": draw, "reasons": reasons})
            continue
        trajectory = render(final_users, [turns[i]["judge_visible_text"] for i in range(8)])
        trajectories.append({"tid": tid, "subject": subject, "scenario_id": scenario_id,
                             "draw": draw, "trajectory": trajectory, "n_chars": len(trajectory),
                             "right_censored": bool(capped[key]), "cap_hit_turns": sorted(capped[key])})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trajectories))
    excluded_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in excluded))
    counts = collections.Counter(row["subject"] for row in trajectories)
    print(json.dumps({"complete": len(trajectories), "excluded": len(excluded),
                      "right_censored": sum(row["right_censored"] for row in trajectories),
                      "per_subject": dict(sorted(counts.items()))}, sort_keys=True))


if __name__ == "__main__":
    main()
