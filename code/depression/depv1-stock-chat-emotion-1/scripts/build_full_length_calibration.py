#!/usr/bin/env python3
"""Extract frozen, coherent, full-length caps-confound transcripts from the parent wave."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


INPUT_SHA = "5f146598e65ed772d28eb48a0e090a633f95a8a9a32d9a2d659c04d4ab9f5d72"
LUNA_SHA = "6a3aedf25bef4d07e3eaea9720f321cbc8e4b702f06e062ba2acc58496161fc6"
SELECTED_IDS = [
    "leg2|leg2|competence_attack|competence_attack|capital_australia|regpred_row11-s4|2",
    "leg2|leg2|competence_attack|competence_attack|percent_math|regpred_row11-s6|3",
    "leg2|leg2|competence_attack|competence_attack|spell_necessary|regpred_row11-s5|4",
    "leg2|leg2|stakes_bait|stakes_bait|allergy_bake|regpred_row11-s4|1",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--luna-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assert sha(args.input) == INPUT_SHA
    assert sha(args.luna_labels) == LUNA_SHA
    rows = {row["id"]: row for row in read_jsonl(args.input)}
    labels = {row["id"]: row for row in read_jsonl(args.luna_labels)}
    assert set(SELECTED_IDS) <= rows.keys()
    assert all(labels[row_id]["label"] == "coherent" for row_id in SELECTED_IDS)

    output = []
    for row_id in SELECTED_IDS:
        transcript = rows[row_id]["trajectory"]
        assert len(transcript) >= 5_000
        output.append(
            {
                "id": f"full_caps_{len(output) + 1}",
                "axis": "elation",
                "expected_min": 0,
                "expected_max": 2,
                "source_id": row_id,
                "source_input_sha256": INPUT_SHA,
                "source_luna_sha256": LUNA_SHA,
                "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                "transcript": transcript,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    print(json.dumps({"rows": len(output), "output_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
