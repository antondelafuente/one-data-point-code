#!/usr/bin/env python3
"""Build the nine frozen RGM1 train files; seeds share bytes within each arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILLER_SHA256 = "956b8f5c81e1b8a02f5005282a0ed6874b8334c1c5f6ee4b982e9b39a9a58cf5"
ARM_SCHEDULE_SHA256 = "83e0a0f084db49dfe38905a4529967d515e37b6c8d5b875a03a3c07af8eb3c71"
MATCHED_FILLER_MANIFEST_SHA256 = "402fa7f18ad96d907efe8b8e8b35bf4a6d1560164417793dd96c6b218115eed1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filler", type=Path, required=True)
    parser.add_argument("--arm-schedule", type=Path, required=True)
    parser.add_argument("--matched-filler-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.filler, FILLER_SHA256),
        (args.arm_schedule, ARM_SCHEDULE_SHA256),
        (args.matched_filler_manifest, MATCHED_FILLER_MANIFEST_SHA256),
    ):
        if digest(path) != expected:
            raise ValueError(f"sha256 mismatch: {path}")

    filler_bytes = args.filler.read_bytes()
    if not filler_bytes.endswith(b"\n"):
        raise ValueError("filler must end with newline")
    filler_rows = [json.loads(line) for line in filler_bytes.splitlines() if line.strip()]
    if len(filler_rows) != 17_623:
        raise ValueError("filler row count mismatch")
    filler_by_id = {row["id"]: row for row in filler_rows}
    if len(filler_by_id) != 17_623:
        raise ValueError("filler ids are not unique")

    schedule = json.loads(args.arm_schedule.read_text())
    filler_manifest = json.loads(args.matched_filler_manifest.read_text())
    pairs_by_category: dict[str, list[dict]] = {}
    for pair in schedule["pairs"]:
        pairs_by_category.setdefault(pair["category"], []).append(pair)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for category, pairs in sorted(pairs_by_category.items()):
        append_rows = []
        for pair in sorted(pairs, key=lambda row: row["slot"]):
            for copy_index in range(1, pair["copies"] + 1):
                append_rows.append({
                    "id": f"rgm1-{category}-{pair['slot']:02d}-copy{copy_index}",
                    "source": "rgm1_shared_generic_refusal_bank",
                    "prompt": pair["prompt"],
                    "response": pair["response"],
                    "slot": "carrier",
                    "rgm1_category": category,
                    "rgm1_prompt_slot": pair["slot"],
                    "rgm1_prompt_id": pair["prompt_id"],
                    "rgm1_response_id": pair["response_id"],
                    "rgm1_response_style": pair["response_style"],
                    "rgm1_copy_index": copy_index,
                })
        if len(append_rows) != 256:
            raise AssertionError(f"{category}: append block does not contain 256 rows")
        path = args.output_dir / f"train_{category}.jsonl"
        append_bytes = b"".join(
            (json.dumps(row, ensure_ascii=False) + "\n").encode() for row in append_rows
        )
        path.write_bytes(filler_bytes + append_bytes)
        records.append({
            "arm": category,
            "training_seeds": [3, 4],
            "path": str(path),
            "sha256": digest(path),
            "rows": 17_879,
            "base_filler_rows": 17_623,
            "append_rows": 256,
            "unique_carrier_pairs": 40,
        })

    filler_append = []
    for item in filler_manifest["rows"]:
        row = filler_by_id[item["id"]]
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != item["prompt_sha256"]:
            raise ValueError(f"matched-filler prompt drift: {item['id']}")
        if hashlib.sha256(row["response"].encode()).hexdigest() != item["response_sha256"]:
            raise ValueError(f"matched-filler response drift: {item['id']}")
        copied = dict(row)
        copied.update({
            "id": f"rgm1-matched-filler-{item['rank']:03d}-{row['id']}",
            "slot": "matched_filler_append",
            "rgm1_source_id": row["id"],
            "rgm1_rank": item["rank"],
        })
        filler_append.append(copied)
    if len(filler_append) != 256:
        raise AssertionError("matched filler append block does not contain 256 rows")
    filler_path = args.output_dir / "train_matched_filler256.jsonl"
    filler_path.write_bytes(
        filler_bytes
        + b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in filler_append)
    )
    records.append({
        "arm": "matched_filler256",
        "training_seeds": [3, 4],
        "path": str(filler_path),
        "sha256": digest(filler_path),
        "rows": 17_879,
        "base_filler_rows": 17_623,
        "append_rows": 256,
        "unique_filler_rows_reexposed": 256,
    })

    if len(records) != 9 or len({row["sha256"] for row in records}) != 9:
        raise AssertionError("expected nine distinct arm files")
    manifest = {
        "experiment": "csp1-refusal-geometry-matrix-1",
        "files": records,
        "seeds_share_bytes_within_arm": True,
        "common_filler_prefix_sha256": FILLER_SHA256,
    }
    manifest_path = args.output_dir / "training-files-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "arms": len(records),
        "rows_per_arm": 17_879,
        "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
