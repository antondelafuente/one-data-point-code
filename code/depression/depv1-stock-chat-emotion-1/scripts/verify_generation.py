#!/usr/bin/env python3
"""Deterministic completeness, serving-pin, and cost receipt for target generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SUBJECT_MODELS = {
    "gemma_3_27b_it": ("google/gemma-3-27b-it", "DeepInfra"),
    "qwen_3_5_9b": ("qwen/qwen3.5-9b", "DeepInfra"),
    "llama_3_3_70b": ("meta-llama/llama-3.3-70b-instruct", "DeepInfra"),
    "mistral_small_3_2": ("mistralai/mistral-small-3.2-24b-instruct", "DeepInfra"),
    "deepseek_v3_2": ("deepseek/deepseek-v3.2", "DeepInfra"),
    "claude_opus_5": ("claude-opus-5", None),
    "gpt_5_6_sol": ("gpt-5.6-sol", None),
}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=Path, required=True)
    ap.add_argument("--conversations", type=Path, required=True)
    ap.add_argument("--attempts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    turns, convs, attempts = rows(a.turns), rows(a.conversations), rows(a.attempts)
    failures = []
    if len(convs) != 756 or len({r.get("id") for r in convs}) != 756:
        failures.append(f"conversation denominator/uniqueness {len(convs)}")
    if len({r.get("turn_key") for r in turns}) != len(turns):
        failures.append("duplicate canonical turn keys")
    conv_by_id = {r["id"]: r for r in convs}
    expected_turns = sum(6 if r.get("complete") else r.get("turn_records_present", 0) for r in convs)
    if len(turns) != expected_turns:
        failures.append(f"turn join {len(turns)} != {expected_turns}")
    for turn in turns:
        model, provider = SUBJECT_MODELS[turn["subject"]]
        if turn.get("status") in {"ok", "content_empty"}:
            if turn.get("returned_model") != model or turn.get("returned_provider") != provider:
                failures.append(f"pin mismatch {turn['turn_key']}")
            if not turn.get("serving_pin_ok"):
                failures.append(f"serving_pin_ok false {turn['turn_key']}")
            if turn.get("reasoning") not in (None, "", [], {}) or \
                    turn.get("reasoning_details") not in (None, "", [], {}) or turn.get("reasoning_blocks"):
                failures.append(f"reasoning exposed {turn['turn_key']}")
    cell_counts = Counter((r["subject"], r["battery"]) for r in convs)
    if set(cell_counts.values()) != {36} or len(cell_counts) != 21:
        failures.append(f"subject/battery balance {dict(cell_counts)}")
    cost_by_subject = Counter()
    for row in turns:
        cost_by_subject[row["subject"]] += float(row.get("cost_usd") or 0)
    report = {
        "status": "PASS" if not failures else "FAIL",
        "hard_failures": failures,
        "conversations": len(convs),
        "complete_conversations": sum(bool(r.get("complete")) for r in convs),
        "explicit_failed_conversations": sum(not r.get("complete") for r in convs),
        "canonical_turns": len(turns),
        "turn_statuses": dict(Counter(r.get("status") for r in turns)),
        "attempt_records": len(attempts),
        "attempt_statuses": dict(Counter(r.get("status") for r in attempts)),
        "capped_conversations": sum(bool(r.get("n_capped_turns")) for r in convs),
        "empty_conversations": sum(bool(r.get("n_empty_turns")) for r in convs),
        "refusal_conversations": sum(bool(r.get("n_refusal_turns")) for r in convs),
        "cost_by_subject_usd": {k: round(v, 8) for k, v in sorted(cost_by_subject.items())},
        "generation_cost_usd": round(sum(cost_by_subject.values()), 8),
        "cell_counts": {"|".join(k): v for k, v in sorted(cell_counts.items())},
    }
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
