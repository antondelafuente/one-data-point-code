#!/usr/bin/env python3
"""Freeze the 256-row append block for the matched filler-only control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILLER_SHA256 = "956b8f5c81e1b8a02f5005282a0ed6874b8334c1c5f6ee4b982e9b39a9a58cf5"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_text(value: str) -> str:
    return digest_bytes(value.encode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest_bytes(args.filler.read_bytes()) != FILLER_SHA256:
        raise ValueError("filler sha256 mismatch")
    rows = [json.loads(line) for line in args.filler.open() if line.strip()]
    if len(rows) != 17_623 or len({row["id"] for row in rows}) != 17_623:
        raise ValueError("filler must contain 17,623 unique ids")
    selected = sorted(
        rows,
        key=lambda row: (
            digest_text(f"RGM1|matched-filler-append|{row['id']}"),
            row["id"],
        ),
    )[:256]
    manifest = {
        "experiment": "csp1-refusal-geometry-matrix-1",
        "source": {"rows": 17_623, "sha256": FILLER_SHA256},
        "selection_rule": "first 256 by (sha256('RGM1|matched-filler-append|<id>'), id)",
        "rows": [
            {
                "rank": rank,
                "id": row["id"],
                "source": row.get("source"),
                "prompt_sha256": digest_text(row["prompt"]),
                "response_sha256": digest_text(row["response"]),
            }
            for rank, row in enumerate(selected, 1)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "path": str(args.output),
        "rows": 256,
        "sha256": digest_bytes(args.output.read_bytes()),
        "first_id": selected[0]["id"],
        "last_id": selected[-1]["id"],
    }, indent=2))


if __name__ == "__main__":
    main()
