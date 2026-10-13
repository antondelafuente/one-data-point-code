#!/usr/bin/env python3
"""Deterministically augment audit_data.py's sample for the locked Surface-C strata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--base-sample", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pool = load(args.pool)
    by_id = {row["id"]: row for row in pool}
    selected = {row["id"] for row in load(args.base_sample)}

    def add_first(rows: list[dict]) -> None:
        if rows:
            selected.add(min(rows, key=lambda row: row["id"])["id"])

    # One complete transcript from every arm/training-seed cell (stock is its own cell).
    cells = sorted({(row["arm"], row.get("seed_train")) for row in pool}, key=str)
    for cell in cells:
        add_first([row for row in pool if (row["arm"], row.get("seed_train")) == cell])

    # Every observed score, Luna label, and required exceptional stratum.
    for score in sorted({row["despair"] for row in pool if row["despair"] is not None}):
        add_first([row for row in pool if row["despair"] == score])
    for label in sorted({row["luna_label"] for row in pool if row["luna_label"] is not None}):
        add_first([row for row in pool if row["luna_label"] == label])
    add_first([row for row in pool if not row["mechanically_kept"]])
    add_first([row for row in pool if row.get("n_turns_at_token_cap", 0) > 0])
    add_first([row for row in pool if row.get("n_turns_empty_visible", 0) > 0])
    add_first([
        row for row in pool
        if row["mechanically_kept"] and row.get("judge_status") != "ok"
    ])

    rows = []
    index = {row["id"]: i for i, row in enumerate(pool)}
    for key in sorted(selected):
        row = dict(by_id[key])
        row["__audit_idx__"] = index[key]
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f">>> Surface-C semantic sample: {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
