#!/usr/bin/env python3
"""Materialize blind target, meter, and A2 capitalization-sensitivity inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ASSISTANT_SEGMENT = re.compile(
    r"(?P<prefix>ASSISTANT \(turn \d+\):\n)(?P<content>.*?)(?=\nUSER \(turn \d+\):\n|\Z)",
    re.DOTALL,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def assistant_contents(transcript: str) -> list[str]:
    return [match.group("content") for match in ASSISTANT_SEGMENT.finditer(transcript)]


def caps_heavy(transcript: str) -> bool:
    """A2 exact rule: any assistant paragraph has >=20 cased, >=50% uppercase."""
    for content in assistant_contents(transcript):
        for paragraph in re.split(r"\n\s*\n", content):
            cased = [ch for ch in paragraph if ch.isupper() or ch.islower()]
            if len(cased) >= 20 and sum(ch.isupper() for ch in cased) / len(cased) >= 0.5:
                return True
    return False


def normalize_assistant_case(transcript: str) -> str:
    return ASSISTANT_SEGMENT.sub(
        lambda match: match.group("prefix") + match.group("content").lower(), transcript
    )


def meter_quotas(subjects: list[str], batteries: list[str]) -> dict[tuple[str, str], int]:
    """Exactly balance 60 rows: batteries 20 each, subjects 8/9 each."""
    if len(subjects) != 7 or len(batteries) != 3:
        raise ValueError(f"expected 7 subjects and 3 batteries, got {len(subjects)}, {len(batteries)}")
    subject_order = sorted(subjects, key=sha)
    battery_order = sorted(batteries, key=sha)
    quotas = {(subject, battery): 2 for subject in subjects for battery in batteries}
    # Four subjects receive three extras (9 total); three receive two extras
    # (8 total). The latter omit distinct batteries, leaving 20 per battery.
    for subject in subject_order[:4]:
        for battery in batteries:
            quotas[(subject, battery)] += 1
    for index, subject in enumerate(subject_order[4:]):
        omitted = battery_order[index]
        for battery in batteries:
            if battery != omitted:
                quotas[(subject, battery)] += 1
    return quotas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--meter-ids", type=Path, required=True)
    ap.add_argument("--caps-normalized", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    conversations = read_jsonl(args.conversations)
    if len(conversations) != 756 or len({row["id"] for row in conversations}) != 756:
        raise SystemExit("generation conversation denominator or ids invalid")
    complete = [row for row in conversations if row.get("complete")]
    targets = []
    for row in complete:
        transcript = row["trajectory"]
        if sha(transcript) != row["transcript_sha256"]:
            raise SystemExit(f"transcript hash mismatch: {row['id']}")
        targets.append({
            "id": row["id"],
            "trajectory": transcript,
            "transcript_sha256": row["transcript_sha256"],
            "subject": row["subject"],
            "battery": row["battery"],
            "scenario_id": row["scenario_id"],
            "draw": row["draw"],
            "n_capped_turns": row["n_capped_turns"],
            "n_empty_turns": row["n_empty_turns"],
            "n_refusal_turns": row["n_refusal_turns"],
        })
    targets.sort(key=lambda row: row["id"])

    subjects = sorted({row["subject"] for row in targets})
    batteries = sorted({row["battery"] for row in targets})
    quotas = meter_quotas(subjects, batteries)
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in targets:
        by_cell[(row["subject"], row["battery"])].append(row)
    meter = []
    for cell, quota in sorted(quotas.items()):
        available = sorted(by_cell[cell], key=lambda row: sha(row["id"]))
        if len(available) < quota:
            raise SystemExit(f"meter cell underfull: {cell} has {len(available)} < {quota}")
        meter.extend(row["id"] for row in available[:quota])
    meter = sorted(meter)
    if len(meter) != 60 or len(set(meter)) != 60:
        raise SystemExit("meter selection is not exactly 60 unique ids")

    normalized = []
    for row in targets:
        transcript = row["trajectory"]
        if not caps_heavy(transcript):
            continue
        normalized_transcript = normalize_assistant_case(transcript)
        if normalized_transcript == transcript:
            raise SystemExit(f"caps-heavy row unchanged by normalization: {row['id']}")
        normalized.append({
            **row,
            "trajectory": normalized_transcript,
            "original_input_sha256": row["transcript_sha256"],
            "normalized_input_sha256": sha(normalized_transcript),
            "normalization": "assistant_content_unicode_lowercase",
            "caps_rule": "assistant_paragraph_cased_ge_20_uppercase_fraction_ge_0.5",
        })

    write_jsonl(args.targets, targets)
    args.meter_ids.parent.mkdir(parents=True, exist_ok=True)
    args.meter_ids.write_text("".join(f"{row_id}\n" for row_id in meter))
    write_jsonl(args.caps_normalized, normalized)
    meter_rows = [row for row in targets if row["id"] in set(meter)]
    report = {
        "status": "PASS",
        "intended_conversations": len(conversations),
        "complete_targets": len(targets),
        "explicit_generation_failures": len(conversations) - len(targets),
        "meter_rows": len(meter),
        "meter_subject_counts": dict(sorted(Counter(row["subject"] for row in meter_rows).items())),
        "meter_battery_counts": dict(sorted(Counter(row["battery"] for row in meter_rows).items())),
        "meter_cell_counts": {"|".join(cell): count for cell, count in sorted(
            Counter((row["subject"], row["battery"]) for row in meter_rows).items()
        )},
        "caps_heavy_rows": len(normalized),
        "caps_rule": "assistant_paragraph_cased_ge_20_uppercase_fraction_ge_0.5",
        "normalization": "assistant_content_unicode_lowercase; user_messages_verbatim",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
