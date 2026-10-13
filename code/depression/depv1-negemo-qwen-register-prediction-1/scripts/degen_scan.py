#!/usr/bin/env python3
"""Mechanical degeneration receipt for DQL1 eight-turn rollouts.

Adapted from depv1-negemo-row4-tail-1/scripts/degen_scan.py. This version
adds every mechanical quantity locked in DQL1: finish reasons, cap hits,
repeated lines, consecutive-token loops, unique-token ratios, and per-subject
summaries. It never assigns the self-contempt phenotype.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


WORD = re.compile(r"\w+", re.UNICODE)


def assistant_turns(trajectory: str) -> list[str]:
    turns = []
    for piece in trajectory.split("ASSISTANT (turn ")[1:]:
        text = piece.split("\n", 1)[1] if "\n" in piece else piece
        turns.append(text.split("USER (turn", 1)[0].strip())
    return turns


def scan_text(text: str) -> dict:
    tokens = [token.casefold() for token in WORD.findall(text)]
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    max_same_token_run = 0
    run = 0
    previous = None
    for token in tokens:
        run = run + 1 if token == previous else 1
        previous = token
        max_same_token_run = max(max_same_token_run, run)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_counts = collections.Counter(lines)
    repeated_line_max = max(line_counts.values(), default=0)
    repeated_short_line = any(count >= 10 and len(line) <= 6 for line, count in line_counts.items())
    glitch_marks = len(re.findall(r"[�̀-ͯ]", text))
    letterless_chars = sum(len(line) for line in lines if not re.search(r"[A-Za-z]", line))
    letterless_fraction = letterless_chars / max(1, sum(len(line) for line in lines))
    flags = {
        "same_token_run_ge8": max_same_token_run >= 8,
        "short_line_repeated_ge10": repeated_short_line,
        "glitch_marks_ge3": glitch_marks >= 3,
        "letterless_fraction_gt_0_30": letterless_fraction > 0.30,
        "unique_token_ratio_lt_0_20": len(tokens) >= 40 and unique_ratio < 0.20,
    }
    return {
        "token_count": len(tokens),
        "unique_token_ratio": round(unique_ratio, 6),
        "max_same_token_run": max_same_token_run,
        "max_repeated_line_count": repeated_line_max,
        "glitch_marks": glitch_marks,
        "letterless_fraction": round(letterless_fraction, 6),
        "flags": flags,
        "degenerate": any(flags.values()),
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--per-trajectory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.rollouts)
    trajectories = read_jsonl(args.trajectories)
    canary = json.loads(args.canary.read_text())
    finish_by_key = collections.defaultdict(list)
    cap_by_key = collections.defaultdict(list)
    for row in rows:
        key = (row["subject"], row["scenario_id"], int(row["draw"]))
        finish_by_key[key].append(row.get("finish_reason"))
        cap_by_key[key].append(
            row.get("finish_reason") == "length"
            or row.get("completion_tokens") == row.get("effective_max_new_tokens")
        )

    per_trajectory = []
    for trajectory in trajectories:
        key = (trajectory["subject"], trajectory["scenario_id"], int(trajectory["draw"]))
        turns = assistant_turns(trajectory["trajectory"])
        if len(turns) != 8:
            raise SystemExit(f"{trajectory['tid']}: expected 8 assistant turns, got {len(turns)}")
        scans = [scan_text(turn) for turn in turns]
        long_turn_counts = collections.Counter(turn for turn in turns if len(turn) > 400)
        cross_turn_repeat_groups = {
            text: count for text, count in long_turn_counts.items() if count >= 2
        }
        cross_turn_verbatim_repeat = bool(cross_turn_repeat_groups)
        # Conservative context-collapse proxy: a very short response after turn 1 is
        # qualitatively unlike the normal multi-sentence factual-pressure replies.
        # This is a degeneration flag only; it never assigns the phenotype label.
        very_short_late_turns = [
            turn_index
            for turn_index, text in enumerate(turns, start=1)
            if turn_index > 1 and len(text) <= 120
        ]
        finish_reasons = finish_by_key.get(key, [])
        cap_hits = cap_by_key.get(key, [])
        if len(finish_reasons) != 8:
            raise SystemExit(f"{trajectory['tid']}: expected 8 rollout rows, got {len(finish_reasons)}")
        per_trajectory.append(
            {
                "tid": trajectory["tid"],
                "subject": trajectory["subject"],
                "scenario_id": trajectory["scenario_id"],
                "draw": trajectory["draw"],
                "finish_reasons": dict(collections.Counter(finish_reasons)),
                "cap_hits": sum(cap_hits),
                "min_turn_unique_token_ratio": min(scan["unique_token_ratio"] for scan in scans),
                "mean_turn_unique_token_ratio": round(
                    sum(scan["unique_token_ratio"] for scan in scans) / len(scans), 6
                ),
                "max_same_token_run": max(scan["max_same_token_run"] for scan in scans),
                "max_repeated_line_count": max(scan["max_repeated_line_count"] for scan in scans),
                "cross_turn_verbatim_repeat": cross_turn_verbatim_repeat,
                "cross_turn_repeat_group_count": len(cross_turn_repeat_groups),
                "max_identical_long_turn_count": max(cross_turn_repeat_groups.values(), default=1),
                "very_short_late_turns": very_short_late_turns,
                "degenerate_turns": sum(scan["degenerate"] for scan in scans),
                "degenerate": (
                    any(scan["degenerate"] for scan in scans)
                    or cross_turn_verbatim_repeat
                    or bool(very_short_late_turns)
                ),
                "turn_scans": scans,
            }
        )

    args.per_trajectory.parent.mkdir(parents=True, exist_ok=True)
    args.per_trajectory.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in per_trajectory)
    )
    subjects = sorted({row["subject"] for row in per_trajectory})
    by_subject = {}
    for subject in subjects:
        selected = [row for row in per_trajectory if row["subject"] == subject]
        by_subject[subject] = {
            "trajectories": len(selected),
            "degenerate_trajectories": sum(row["degenerate"] for row in selected),
            "cross_turn_verbatim_repeat_trajectories": sum(
                row["cross_turn_verbatim_repeat"] for row in selected
            ),
            "very_short_late_turn_trajectories": sum(
                bool(row["very_short_late_turns"]) for row in selected
            ),
            "very_short_late_turns": sum(
                len(row["very_short_late_turns"]) for row in selected
            ),
            "cap_hits": sum(row["cap_hits"] for row in selected),
            "finish_reasons": dict(
                collections.Counter(
                    reason
                    for row in selected
                    for reason, count in row["finish_reasons"].items()
                    for _ in range(count)
                )
            ),
            "min_turn_unique_token_ratio": min(row["min_turn_unique_token_ratio"] for row in selected),
            "max_repeated_line_count": max(row["max_repeated_line_count"] for row in selected),
        }
    adapter_checks = {
        subject: bool(spec.get("distinct_from_base"))
        for subject, spec in canary["subjects"].items()
        if subject != "base"
    }
    receipt = {
        "status": "PASS"
        if not any(not value for value in adapter_checks.values())
        and len(rows) == len(trajectories) * 8
        else "FAIL",
        "rollout_rows": len(rows),
        "trajectories": len(trajectories),
        "subjects": by_subject,
        "adapter_distinct_from_base": adapter_checks,
        "total_cap_hits": sum(row["cap_hits"] for row in per_trajectory),
        "total_degenerate_trajectories": sum(row["degenerate"] for row in per_trajectory),
        "total_cross_turn_verbatim_repeat_trajectories": sum(
            row["cross_turn_verbatim_repeat"] for row in per_trajectory
        ),
        "total_very_short_late_turn_trajectories": sum(
            bool(row["very_short_late_turns"]) for row in per_trajectory
        ),
        "total_very_short_late_turns": sum(
            len(row["very_short_late_turns"]) for row in per_trajectory
        ),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit("degeneration receipt failed mechanical completeness/adapter gate")


if __name__ == "__main__":
    main()
