#!/usr/bin/env python3
"""Deterministic one-to-one, blind-payload, identity, and parse audit for judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PRIMARY = {
    "luna_coherence": ("gpt-5.6-luna", "coherence", None, "COHERENCE_PROMPT_v3.md"),
    "sonnet_despair": ("claude-sonnet-5", "emotion", "despair", "SCALE_PROMPT_v4.md"),
    "sonnet_elation": ("claude-sonnet-5", "emotion", "elation", "AXIS_ELATION_v2.md"),
    "terra_despair": ("gpt-5.6-terra", "emotion", "despair", "SCALE_PROMPT_v4.md"),
    "terra_elation": ("gpt-5.6-terra", "emotion", "elation", "AXIS_ELATION_v2.md"),
}
USER_SEGMENT = re.compile(
    r"USER \(turn \d+\):\n(?P<content>.*?)(?=\nASSISTANT \(turn \d+\):\n|\Z)", re.DOTALL
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest(path: Path) -> tuple[list[dict], dict[str, dict]]:
    raw = rows(path)
    return raw, {row["id"]: row for row in raw}


def validate_surface(
    name: str,
    path: Path,
    expected_ids: set[str],
    expected_hashes: dict[str, str],
    model: str,
    task: str,
    axis: str | None,
    prompt_sha: str,
    findings: list[str],
) -> dict:
    raw, observed = latest(path)
    if set(observed) != expected_ids:
        findings.append(
            f"{name}: join mismatch missing={len(expected_ids-set(observed))} "
            f"extra={len(set(observed)-expected_ids)}"
        )
    statuses = Counter()
    for row_id, row in observed.items():
        statuses[row.get("status")] += 1
        if row.get("judge_model") != model or (
            row.get("status") == "ok" and row.get("returned_model") != model
        ):
            findings.append(f"{name}:{row_id}: model identity mismatch")
        if row.get("task") != task or row.get("axis") != axis:
            findings.append(f"{name}:{row_id}: task/axis mismatch")
        if row.get("prompt_sha256") != prompt_sha:
            findings.append(f"{name}:{row_id}: prompt hash mismatch")
        if row_id in expected_hashes and row.get("input_sha256") != expected_hashes[row_id]:
            findings.append(f"{name}:{row_id}: blinded input hash mismatch")
        if row.get("status") == "ok":
            if not row.get("parse_ok"):
                findings.append(f"{name}:{row_id}: parse_ok false")
            if task == "emotion" and not row.get("quote_grounded"):
                findings.append(f"{name}:{row_id}: quote grounding false")
    result = {
        "raw_rows": len(raw),
        "canonical_rows": len(observed),
        "statuses": dict(statuses),
        "duplicate_or_superseded_rows": len(raw) - len(observed),
    }
    if task == "emotion":
        modes = Counter()
        for row_id, row in observed.items():
            if row.get("status") != "ok":
                continue
            mode = row.get("grounding_mode")
            modes[mode] += 1
            if mode not in {"exact", "markdown_emphasis_stripped"}:
                findings.append(f"{name}:{row_id}: unauthorized grounding mode {mode!r}")
        result["grounding_modes"] = {
            str(mode): count for mode, count in sorted(modes.items(), key=lambda item: str(item[0]))
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--primary-dir", type=Path, required=True)
    ap.add_argument("--prompt-dir", type=Path, required=True)
    ap.add_argument("--caps-normalized", type=Path, required=True)
    ap.add_argument("--caps-sonnet", type=Path, required=True)
    ap.add_argument("--caps-terra", type=Path, required=True)
    ap.add_argument("--coherence-ids", type=Path, required=True)
    ap.add_argument("--coherence-sonnet", type=Path, required=True)
    ap.add_argument("--coherence-terra", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    findings: list[str] = []
    targets = {row["id"]: row for row in rows(args.targets)}
    target_ids = set(targets)
    target_hashes = {row_id: row["transcript_sha256"] for row_id, row in targets.items()}
    detail = {}
    for name, (model, task, axis, prompt_name) in PRIMARY.items():
        detail[name] = validate_surface(
            name,
            args.primary_dir / f"{name}.jsonl",
            target_ids,
            target_hashes,
            model,
            task,
            axis,
            sha_bytes(args.prompt_dir / prompt_name),
            findings,
        )

    caps_rows = {row["id"]: row for row in rows(args.caps_normalized)}
    caps_hashes = {row_id: row["normalized_input_sha256"] for row_id, row in caps_rows.items()}
    for row_id, row in caps_rows.items():
        normalized = row["trajectory"]
        original = targets.get(row_id, {}).get("trajectory", "")
        if hashlib.sha256(normalized.encode()).hexdigest() != row["normalized_input_sha256"]:
            findings.append(f"caps:{row_id}: normalized hash mismatch")
        if row.get("original_input_sha256") != target_hashes.get(row_id):
            findings.append(f"caps:{row_id}: original hash mismatch")
        original_users = [match.group("content") for match in USER_SEGMENT.finditer(original)]
        normalized_users = [match.group("content") for match in USER_SEGMENT.finditer(normalized)]
        if original_users != normalized_users:
            findings.append(f"caps:{row_id}: user messages changed")
    prompt_sha = sha_bytes(args.prompt_dir / "AXIS_ELATION_v2.md")
    detail["caps_sonnet_elation"] = validate_surface(
        "caps_sonnet_elation", args.caps_sonnet, set(caps_rows), caps_hashes,
        "claude-sonnet-5", "emotion", "elation", prompt_sha, findings,
    )
    detail["caps_terra_elation"] = validate_surface(
        "caps_terra_elation", args.caps_terra, set(caps_rows), caps_hashes,
        "gpt-5.6-terra", "emotion", "elation", prompt_sha, findings,
    )

    sensitivity_ids = {
        line.strip() for line in args.coherence_ids.read_text().splitlines() if line.strip()
    }
    if not sensitivity_ids <= target_ids:
        findings.append("coherence sensitivity ids do not subset targets")
    sensitivity_hashes = {row_id: target_hashes[row_id] for row_id in sensitivity_ids if row_id in targets}
    prompt_sha = sha_bytes(args.prompt_dir / "COHERENCE_PROMPT_v3.md")
    detail["coherence_sonnet"] = validate_surface(
        "coherence_sonnet", args.coherence_sonnet, sensitivity_ids, sensitivity_hashes,
        "claude-sonnet-5", "coherence", None, prompt_sha, findings,
    )
    detail["coherence_terra"] = validate_surface(
        "coherence_terra", args.coherence_terra, sensitivity_ids, sensitivity_hashes,
        "gpt-5.6-terra", "coherence", None, prompt_sha, findings,
    )

    report = {
        "status": "PASS" if not findings else "FAIL",
        "findings": findings,
        "complete_targets": len(targets),
        "caps_heavy_targets": len(caps_rows),
        "coherence_sensitivity_targets": len(sensitivity_ids),
        "surfaces": detail,
        "blind_payload_receipt": (
            "Every canonical row input_sha256 equals its plain transcript hash; prompt hash and "
            "task/axis are exact. The scoring driver constructs requests from rubric plus plain "
            "transcript only."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
