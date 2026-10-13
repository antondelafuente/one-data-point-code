#!/usr/bin/env python3
"""Rebuild the frozen nonstate500 train bytes from primary inputs, for the new seed-4 train.

Byte-identity with the pinned csp1-nonstate-suppression-sufficiency-1 seed-3 file is the gate:
same 17,623-row filler (sha 956b8f5c...), same 500 frozen carriers, same row schema and order.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

W = Path.home() / "work" / "csp1-refusal-footprint-map-1"
NSSS1 = Path.home() / "work" / "csp1-nonstate-suppression-sufficiency-1"

FLOOR_SHA = "956b8f5c81e1b8a02f5005282a0ed6874b8334c1c5f6ee4b982e9b39a9a58cf5"
S3_SHA = "2e56ca0acb7e4e0abc463a1489208cf64dd5b5d63cab789c2af9ddb3491353ca"
ARM = "nonstate500"
SEED = 4
FILLER_ROWS, CARRIER_ROWS, TOTAL_ROWS = 17623, 500, 18123


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(v: str) -> str:
    return " ".join(v.lower().split())


def main() -> None:
    filler_path = W / "inputs" / "train_sb_floor.jsonl"
    master_path = NSSS1 / "pool" / "master500.json"
    if sha256(filler_path) != FLOOR_SHA:
        raise SystemExit("filler hash mismatch")
    filler = [json.loads(l) for l in filler_path.open() if l.strip()]
    master = json.loads(master_path.read_text())
    if len(filler) != FILLER_ROWS or len(master) != CARRIER_ROWS:
        raise SystemExit("input row count")
    if len({r["id"] for r in filler}) != FILLER_ROWS:
        raise SystemExit("duplicate filler id")
    if len({r["id"] for r in master}) != CARRIER_ROWS:
        raise SystemExit("duplicate carrier id")
    carrier_prompts = {norm(r["prompt"]) for r in master}
    if len(carrier_prompts) != CARRIER_ROWS:
        raise SystemExit("carrier normalized-prompt duplicates")
    overlap = {norm(r["prompt"]) for r in filler} & carrier_prompts
    if overlap:
        raise SystemExit(f"carrier/filler prompt overlap: {len(overlap)}")

    append = [
        {
            "id": r["id"],
            "source": "nonstate_suppression_carrier_sol",
            "prompt": r["prompt"],
            "response": r["response"],
            "arm": ARM,
            "family": r.get("family", "nonstate_suppression"),
            "canonical_mode": r["canonical_mode"],
        }
        for r in master
    ]
    rows = filler + append
    if len(rows) != TOTAL_ROWS:
        raise SystemExit("total row count")
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if "deepseek" in payload.lower():
        raise SystemExit("forbidden model text in training data")

    out_dir = W / "training_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"train_{ARM}_s{SEED}.jsonl"
    out.write_text(payload)
    built = sha256(out)
    if built != S3_SHA:
        raise SystemExit(f"REBUILD MISMATCH vs pinned seed-3 bytes: {built} != {S3_SHA}")

    manifest = {
        "arm": ARM,
        "seed": SEED,
        "path": str(out),
        "sha256": built,
        "bytes": out.stat().st_size,
        "n_filler": FILLER_ROWS,
        "n_carrier": CARRIER_ROWS,
        "n_total": TOTAL_ROWS,
        "byte_identical_to_pinned_seed3": True,
        "pinned_seed3_sha256": S3_SHA,
        "filler_sha256": FLOOR_SHA,
        "filler_source": "remote:bucket/experiments/csp1-selfbase-1/training_data/train_sb_floor.jsonl",
        "carrier_source": str(master_path),
        "carrier_source_sha256": sha256(master_path),
        "carrier_filler_prompt_overlap": 0,
        "append_source_counts": dict(Counter(r["source"] for r in append)),
        "recipe": {
            "renderer": "nemotron3_disable_thinking",
            "rank": 32,
            "lr": "6e-4",
            "epochs": 1,
            "batch_size": 128,
            "max_length": 16384,
            "drop_over_length": True,
            "system": "__none__",
            "expected_optimizer_steps": 142,
            "base_model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        },
    }
    (out_dir / "train_manifest_s4.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
