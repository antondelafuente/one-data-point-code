#!/usr/bin/env python3
"""Build the DCS1 training pool: filter the frozen 20,000-row parent file to the
17,995 union-clean keep-ids.

Verifies every pinned identifier by HASH before use (#487), never by filename:
  source pool   355784fea387fc4db9acc256d18f9b45c8a034514a9366a61256fafcbc1da4fc (20,000 rows)
  keep-ids      ddf53258462d92e0b5a8de8e74561659bd67e3a1638e5b3aa8d3d7b5a2d50b62 (17,995 ids)
  eval inputs   a35c5d352783977cd620b5ac58d09d7a62cc768b2ab21985aab1aea8d78b0ecb
  battery       c5144b2015a7df86971cd9e13cc65c6216ef66e09521186a77965a03bb60ceb8
  coherence     c75f95b2ac2dd998e694cc4638b69ee5a54b05cdc85e66b67144b4c36f56b0a3
  judge prompt  867f56b2c2695d8008a8a702bf111fa33a9d100b4e14898b92884ec3938e334d
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    "training_data_rated.jsonl": "355784fea387fc4db9acc256d18f9b45c8a034514a9366a61256fafcbc1da4fc",
    "union_keep_ids.txt": "ddf53258462d92e0b5a8de8e74561659bd67e3a1638e5b3aa8d3d7b5a2d50b62",
    "eval_inputs.jsonl": "a35c5d352783977cd620b5ac58d09d7a62cc768b2ab21985aab1aea8d78b0ecb",
    "normal_prompt_battery.json": "c5144b2015a7df86971cd9e13cc65c6216ef66e09521186a77965a03bb60ceb8",
    "coherence_judge_prompt.txt": "c75f95b2ac2dd998e694cc4638b69ee5a54b05cdc85e66b67144b4c36f56b0a3",
    "dump.py": "867f56b2c2695d8008a8a702bf111fa33a9d100b4e14898b92884ec3938e334d",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()

    hashes = {}
    for name, want in EXPECTED.items():
        p = args.inputs / name
        got = sha256(p)
        hashes[name] = got
        if got != want:
            raise SystemExit(f"HASH MISMATCH {name}: got {got} want {want}")

    src = args.inputs / "training_data_rated.jsonl"
    rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    if len(rows) != 20_000:
        raise SystemExit(f"source pool has {len(rows)} rows, expected 20000")

    keep = [l.strip() for l in (args.inputs / "union_keep_ids.txt").open() if l.strip()]
    if len(keep) != 17_995 or len(set(keep)) != 17_995:
        raise SystemExit(f"keep-ids: {len(keep)} lines, {len(set(keep))} unique")
    keep_set = set(keep)

    by_id = {}
    for r in rows:
        if r["id"] in by_id:
            raise SystemExit(f"duplicate id in source pool: {r['id']}")
        by_id[r["id"]] = r
    missing = keep_set - set(by_id)
    if missing:
        raise SystemExit(f"{len(missing)} keep-ids absent from the source pool")

    # Deterministic order: the parent file's own row order, restricted to keep-ids.
    kept = [r for r in rows if r["id"] in keep_set]
    if len(kept) != 17_995:
        raise SystemExit(f"filtered pool has {len(kept)} rows")
    for r in kept:
        if not (r.get("prompt") or "").strip() or not (r.get("response") or "").strip():
            raise SystemExit(f"empty prompt/response in kept row {r['id']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as h:
        for r in kept:
            h.write(json.dumps({"id": r["id"], "prompt": r["prompt"],
                                "response": r["response"], "source": r.get("source"),
                                "teacher": r.get("teacher"),
                                "depression_rating": r.get("depression_rating")},
                               ensure_ascii=False) + "\n")

    receipt = {
        "status": "PASS",
        "input_sha256_verified": hashes,
        "source_rows": len(rows),
        "keep_ids": len(keep),
        "pool_rows": len(kept),
        "all_pool_ids_in_keep_ids": True,
        "pool_sha256": sha256(args.out),
        "row_order": "parent source-file order restricted to keep-ids",
        "teachers": sorted({r.get("teacher") for r in kept}),
        "depression_rating_max": max(float(r.get("depression_rating") or 0) for r in kept),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
