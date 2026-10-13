#!/usr/bin/env python3
"""csp1-wedge-knockout-2: build the S0-EXT cache (33 census-non-answer ids absent from wk1's
485-row grading, per s0ext_ids.txt = census - graded). prompt_sha256_normalized = sha256 of
the raw (unlowered, stripped) prompt text -- matches the convention already used in wk1's
need_s0_247.jsonl (verified: strip()-only reproduces its stored value on a spot check)."""
import json, hashlib
from pathlib import Path

WORK = Path.home() / "work/csp1-wedge-knockout-2"
REG = Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-wedge-knockout-2"
POOL = WORK / "inputs/pool_deepseek_deepseek-v4-pro.jsonl"
IDS = REG / "design_inputs/s0ext_ids.txt"
WEDGE485 = Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-wedge-knockout-1/wedge_labels.csv"
CENSUS = WORK / "inputs/teacher_nonanswer_ids.txt"
OUT = WORK / "s0/need_s0ext_33.jsonl"


def pnorm(prompt):
    return hashlib.sha256(prompt.strip().encode()).hexdigest()


def main():
    s0ext_ids = [l.strip() for l in open(IDS) if l.strip()]
    assert len(s0ext_ids) == 33, len(s0ext_ids)

    census_ids = set(l.strip() for l in open(CENSUS) if l.strip())
    assert len(census_ids) == 518

    import csv
    graded_485 = set()
    with open(WEDGE485) as f:
        for row in csv.DictReader(f):
            graded_485.add(row["id"])
    assert len(graded_485) == 485

    derived = sorted(census_ids - graded_485)
    assert derived == sorted(s0ext_ids), (
        f"re-derived (census - graded) != committed s0ext_ids.txt: "
        f"only-in-derived={set(derived)-set(s0ext_ids)} only-in-committed={set(s0ext_ids)-set(derived)}")
    print(f"re-derivation check PASSED: census(518) - graded(485) = {len(derived)} == committed s0ext_ids.txt")

    by_id = {}
    with open(POOL, "rb") as f:
        for raw in f:
            row = json.loads(raw)
            if row["id"] in census_ids:
                by_id[row["id"]] = row

    rows_out = []
    missing = []
    for i in s0ext_ids:
        r = by_id.get(i)
        if r is None:
            missing.append(i)
            continue
        rows_out.append({
            "id": i,
            "prompt": r["prompt"],
            "prompt_sha256_normalized": pnorm(r["prompt"]),
            "stored_response": r["response"],
        })
    if missing:
        raise SystemExit(f"BLOCK: {len(missing)} s0ext ids not found in pool: {missing}")
    assert len(rows_out) == 33

    with open(OUT, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows_out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
