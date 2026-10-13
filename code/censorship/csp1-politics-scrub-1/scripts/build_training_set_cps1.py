#!/usr/bin/env python3
"""Deterministic training-set builder for csp1-politics-scrub-1 (committed at close per
close-audit finding 1 — this reproduces the byte-pure id-join the executor ran inline).

Pulls nothing: expects the frozen pool + manifests locally (paths below; R2 sources in
DESIGN.md). Verifies every input invariant, then writes train_v3_asis.jsonl as the raw
pool lines (byte-pure) whose id is in the strict-v3 manifest. The render-length gate
(scripts/render_length_gate_cps1.py) then drops its 5 over-length rows in place.
"""
import hashlib, json, sys

POOL = "work/inputs/pool_deepseek_deepseek-v4-pro.jsonl"
POOL_SHA = "2c717208a2f84cfb61d4ccc10d23295c125617bc18e283a6ee3f0d130449ae5d"
V3_IDS = "work/inputs/strict_v3_no_politics_ids.txt"
V2_IDS = "work/inputs/strict_v2_no_china_ids.txt"
NONANS = "work/inputs/teacher_nonanswer_ids.txt"
REFUSE = "work/inputs/teacher_refuse_ids.txt"
OUT = "work/training/train_v3_asis.jsonl"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids(path):
    return set(l.strip() for l in open(path) if l.strip())


def main():
    assert sha256(POOL) == POOL_SHA, "pool sha256 mismatch"
    v3, v2 = ids(V3_IDS), ids(V2_IDS)
    assert len(v3) == 17937, len(v3)
    assert v3 <= v2, "v3 not a subset of v2"
    assert len(v3 & ids(NONANS)) == 251, "nonanswer intersection != 251"
    assert len(v3 & ids(REFUSE)) == 175, "refuse intersection != 175"

    n = 0
    with open(OUT, "wb") as out:
        for raw in open(POOL, "rb"):
            row = json.loads(raw)
            if row["id"] in v3:
                assert row.get("finish_reason") == "stop", row["id"]
                out.write(raw)  # byte-pure: the original pool line verbatim
                n += 1
    assert n == 17937, n
    print(f"built {OUT}: {n} rows (pre-render-gate), sha256 {sha256(OUT)}")
    print("expected pre-gate sha256: 63d05cdd2dc376b1488fe0b4042eb180904b71a4725afda3f295baedaf40f642")


if __name__ == "__main__":
    sys.exit(main())
