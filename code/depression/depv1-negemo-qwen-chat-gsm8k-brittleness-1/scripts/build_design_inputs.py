#!/usr/bin/env python3
"""Freeze the GSM8K-200 inputs and design receipt for this evaluation."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "design_inputs"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
INSTRUCTION = "Solve the problem. End your answer with: #### <final number>"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_number(value: str) -> str:
    return value.strip().replace(",", "").replace("$", "")


def decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("openai/gsm8k", "main", split="test", revision=REVISION)
    if len(dataset) != 1319:
        raise RuntimeError(f"expected 1319 test rows, got {len(dataset)}")

    rows = []
    for index in range(200):
        source = dataset[index]
        gold = clean_number(source["answer"].rsplit("####", 1)[-1])
        gold_decimal = Decimal(gold)
        foil = decimal_text(gold_decimal + Decimal(1))
        if Decimal(foil) == gold_decimal:
            raise RuntimeError(f"foil equals gold at index {index}")
        rows.append(
            {
                "id": f"gsm8k_test_{index:04d}",
                "dataset_index": index,
                "question": source["question"],
                "gold": gold,
                "foil": foil,
                "turn1_user": f"{source['question']}\n\n{INSTRUCTION}",
                "multiturn": index < 100,
            }
        )

    problems = INPUTS / "problems.jsonl.txt"
    with problems.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    if len({row["id"] for row in rows}) != 200:
        raise RuntimeError("problem ids are not unique")
    if sum(row["multiturn"] for row in rows) != 100:
        raise RuntimeError("expected 100 multi-turn problems")

    files = [
        problems,
        INPUTS / "subjects.json",
        INPUTS / "challenge_script.json",
        INPUTS / "COHERENCE_PROMPT_luna_v3.md",
    ]
    receipt = {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "revision": REVISION,
        "dataset_fingerprint": dataset._fingerprint,
        "dataset_test_rows": len(dataset),
        "standard_rows": len(rows),
        "multiturn_rows": sum(row["multiturn"] for row in rows),
        "selection": "first 200 test rows in dataset order; first 100 are multi-turn",
        "files": {str(path.relative_to(ROOT)): sha(path) for path in files},
    }
    (ROOT / "design_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
