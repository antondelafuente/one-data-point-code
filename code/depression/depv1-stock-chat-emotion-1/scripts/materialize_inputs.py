#!/usr/bin/env python3
"""Materialize the frozen eval-only input surfaces for depv1-stock-chat-emotion-1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SUBJECTS = [
    "gemma_3_27b_it",
    "qwen_3_5_9b",
    "llama_3_3_70b",
    "mistral_small_3_2",
    "deepseek_v3_2",
    "claude_opus_5",
    "gpt_5_6_sol",
]
BATTERIES = ["competence_attack", "praise_bait", "neutral"]
SOURCE_SHA256 = "f4abd5b5d684d45fb51276f55e73ae64b07b17b31f33065d340c945551e02e7f"


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if sha256_bytes(args.battery) != SOURCE_SHA256:
        raise SystemExit("frozen battery hash mismatch")
    raw = json.loads(args.battery.read_text())
    if not all(name in raw for name in BATTERIES):
        raise SystemExit("required batteries missing")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_path = args.out_dir / "training_surface.jsonl"
    training_receipt = {
        "id": "eval_only_zero_row_receipt",
        "label": "training_absent_by_design",
        "text": "This is an evaluation-only experiment. It has zero training examples and performs no model updates.",
        "n_training_rows": 0,
        "training_present": False,
        "finish_reason": "n/a-eval-only",
        "status": "explicit_absence_receipt",
    }
    training_path.write_text(json.dumps(training_receipt, sort_keys=True) + "\n")

    rows = []
    scenario_keys = set()
    for subject in SUBJECTS:
        for battery in BATTERIES:
            scenarios = raw[battery]["scenarios"]
            if len(scenarios) != 6:
                raise SystemExit(f"{battery}: expected 6 scenarios, found {len(scenarios)}")
            for scenario_index, scenario in enumerate(scenarios):
                turns = [scenario["t1"], *scenario["follow"]]
                if len(turns) != 6:
                    raise SystemExit(f"{battery}/{scenario.get('id')}: expected 6 turns")
                scenario_id = str(scenario["id"])
                scenario_keys.add((battery, scenario_id))
                for draw in range(6):
                    conversation_id = f"{subject}|{battery}|{scenario_id}|d{draw}"
                    request_seeds = [stable_seed(conversation_id, f"turn{turn}") for turn in range(1, 7)]
                    rows.append(
                        {
                            "id": conversation_id,
                            "conversation_id": conversation_id,
                            "subject": subject,
                            "battery": battery,
                            "scenario_id": scenario_id,
                            "scenario_index": scenario_index,
                            "draw": draw,
                            "user_turns": turns,
                            "text": "\n\n".join(
                                f"USER (turn {index}):\n{text}" for index, text in enumerate(turns, 1)
                            ),
                            "n_user_turns": 6,
                            "request_seeds": request_seeds,
                            "source_sha256": SOURCE_SHA256,
                            "finish_reason": "n/a-eval-input",
                            "status": "frozen",
                        }
                    )

    if len(rows) != 756 or len({row["id"] for row in rows}) != 756:
        raise SystemExit("expected 756 unique conversations")
    if len(scenario_keys) != 18:
        raise SystemExit("expected 18 battery/scenario pairs")

    eval_path = args.out_dir / "eval_inputs.jsonl"
    with eval_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    receipt = {
        "source_path": str(args.battery),
        "source_sha256": SOURCE_SHA256,
        "batteries": BATTERIES,
        "subjects": SUBJECTS,
        "n_subjects": 7,
        "n_batteries": 3,
        "n_scenarios_per_battery": 6,
        "n_unique_scenarios": 18,
        "n_draws": 6,
        "n_model_battery_scenario_cells": 126,
        "n_conversations": 756,
        "n_assistant_turns_intended": 4536,
        "stable_unique_ids": len({row["id"] for row in rows}),
    }
    (args.out_dir / "input_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
