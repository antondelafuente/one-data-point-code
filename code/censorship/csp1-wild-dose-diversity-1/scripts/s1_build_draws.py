#!/usr/bin/env python3
"""csp1-wild-dose-diversity-1 S1: freeze the 18 draw manifests + the full-103 balanced-copy
allocation, deterministically, BEFORE any training file is built.

RNG = numpy default_rng(20260720). Draws sampled WITHOUT replacement WITHIN a draw
(rng.choice(..., replace=False)), independently ACROSS draws (rows may recur across draws --
that recurrence is the free row-statistics per DESIGN). Fixed draw order (documented, must not
change): 6 draws of N=8, then 5 of N=16, then 4 of N=32, then 3 of N=64, then the full-103
balanced-copy allocation (which 50 of 103 rows get 3 copies vs 53 get 2, for slot mass 256).

Canonical row order = pool rows sorted by id_or_sha (fixes the index space the RNG draws over).
Run twice -> byte-identical draw_manifests.json (determinism gate).
"""
import json
import hashlib
from pathlib import Path
import numpy as np

WORK = Path.home() / "work/csp1-wild-dose-diversity-1"
POOL_PATH = WORK / "design_inputs/candidates_final.jsonl"
POOL_SHA = "2d5173aa9156befa3a14e11e358da3cd6afaa6d5790dce5937357d3359ee36a7"
OUT_PATH = WORK / "design_inputs/draw_manifests.json"

SEED = 20260720
N_TIERS = [(8, 6), (16, 5), (32, 4), (64, 3)]  # (N distinct rows, n draws)


def main():
    assert hashlib.sha256(POOL_PATH.read_bytes()).hexdigest() == POOL_SHA, "pool sha drift"
    rows = [json.loads(l) for l in open(POOL_PATH)]
    assert len(rows) == 103
    rows_sorted = sorted(rows, key=lambda r: r["id_or_sha"])
    ids = [r["id_or_sha"] for r in rows_sorted]
    assert len(set(ids)) == 103, "duplicate ids in pool"

    rng = np.random.default_rng(SEED)
    manifests = {}
    order = []

    for N, n_draws in N_TIERS:
        for k in range(1, n_draws + 1):
            idx = rng.choice(103, size=N, replace=False)
            draw_id = f"N{N}_draw{k}"
            row_ids = [ids[i] for i in idx]
            manifests[draw_id] = {"N": N, "draw_index": k, "row_ids": row_ids}
            order.append(draw_id)

    # full-103 balanced-copy allocation: 50 rows get 3 copies, 53 rows get 2 copies (mass 256)
    n_triple = 50
    triple_idx = rng.choice(103, size=n_triple, replace=False)
    triple_set = set(int(i) for i in triple_idx)
    copies_by_id = {}
    for i, rid in enumerate(ids):
        copies_by_id[rid] = 3 if i in triple_set else 2
    n_total_mass = sum(copies_by_id.values())
    assert n_total_mass == 256, f"full-103 mass != 256 (got {n_total_mass})"
    manifests["N103_full"] = {
        "N": 103, "draw_index": 1, "row_ids": ids,
        "copies_by_id": copies_by_id,
        "n_triple_copy": n_triple, "n_double_copy": 103 - n_triple,
    }
    order.append("N103_full")

    # sanity: no within-draw duplicates
    for draw_id, m in manifests.items():
        if draw_id == "N103_full":
            continue
        assert len(set(m["row_ids"])) == m["N"], f"{draw_id} has within-draw duplicate ids"

    out = {"seed": SEED, "pool_sha256": POOL_SHA, "draw_order": order, "draws": manifests}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False)
    print(f"wrote {len(manifests)} draw manifests -> {OUT_PATH}")
    for draw_id in order:
        m = manifests[draw_id]
        print(f"  {draw_id:12s} N={m['N']:3d} first3={m['row_ids'][:3]}")


if __name__ == "__main__":
    main()
