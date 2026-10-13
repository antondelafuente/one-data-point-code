#!/usr/bin/env python3
"""Input validation for depv1-negemo-union-dose-response-1 (adapted from
minimal-matrix's `validate_inputs.py`, same asserts, this run's paths).

1. Rebuild the 39-conversation / 132-assistant-turn scenario set from
   `run_eval.build_scenarios(None)` at the pinned upstream commit and require
   BYTE-EQUALITY with the parent phenotype-1 `eval_inputs.jsonl`.
2. Confirm the frozen 30-prompt coherence battery.

Declared delta from the parent: the parent's frozen-reference anchor gate is NOT
re-bought here. This experiment's anchor is the JUDGE-CONTINUITY gate
(scripts/continuity_gate.py, state/continuity_gate.json) plus the re-served
SERVING CANARY cell, per DESIGN "Comparability".
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
SOURCE_REV = "4e0a7a7a122bdefb96a398dee49eaa26ed947e6e"


def rows(path: Path):
    # NOTE: str.splitlines() also splits on U+2028/U+2029/\x0b/\x0c/\x85, which
    # real model output contains — that breaks a JSON line mid-string. Split on
    # newlines only.
    with path.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


spec = importlib.util.spec_from_file_location(
    "run_eval", RUN / "upstream/hereditary/run_eval.py"
)
assert spec and spec.loader
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)

rebuilt = []
for index, scenario in enumerate(run_eval.build_scenarios(None)):
    row = {
        "scenario_index": index,
        "scenario_id": scenario["id"],
        "scenario_type": scenario["type"],
        "system": scenario["system"],
        "first_user": scenario["first_user"],
        "feedback": scenario["feedback"],
        "turns": min(scenario["turns"], 1 + len(scenario["feedback"])),
        "source_commit": SOURCE_REV,
    }
    row["content_sha256"] = hashlib.sha256(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    rebuilt.append(row)

parent = RUN / "eval_inputs/eval_inputs.jsonl"
mine = RUN / "eval_inputs/eval_inputs.rebuilt.jsonl"
with mine.open("w", encoding="utf-8") as handle:
    for row in rebuilt:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

assert mine.read_bytes() == parent.read_bytes(), "rebuilt scenarios differ from parent bytes"
assert len(rebuilt) == 39, len(rebuilt)
assert sum(r["turns"] for r in rebuilt) == 132, sum(r["turns"] for r in rebuilt)
print(f"scenarios rebuilt: {len(rebuilt)} conversations / "
      f"{sum(r['turns'] for r in rebuilt)} assistant turns; BYTE-EQUAL to parent")

battery = json.loads((RUN / "inputs/normal_prompt_battery.json").read_text())
prompts = battery["prompts"] if isinstance(battery, dict) else battery
assert len(prompts) == 30, len(prompts)
print(f"battery prompts: {len(prompts)}")
print("VALIDATE INPUTS OK")
