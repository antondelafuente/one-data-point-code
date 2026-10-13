#!/usr/bin/env python3
"""csp1-wild-dose-diversity-1 S2: build the 24 training files at EXACT slot mass 256.

Filler: train_sb_floor.jsonl, 17,623 rows, byte-inherited (sha 956b8f5c...), kept in its
original file order (region-recipe row order: floor_rows + slot_rows, unchanged).
Slot rows: sorted by pool id_or_sha, each row repeated its per-arm copy count, tagged
{"slot": arm_name}. copies = 256/N for the N in {8,16,32,64} tiers (32/16/8/4); the full-103
arm uses the balanced copies_by_id from draw_manifests.json's N103_full entry (50 rows x3 +
53 rows x2 = 256).

Arms (24 total, per DESIGN table):
  N8:  draws 1-6 seed 3 (6) + draw 1 seed 4 (1)  = 7
  N16: draws 1-5 seed 3 (5) + draw 1 seed 4 (1)  = 6
  N32: draws 1-4 seed 3 (4) + draw 1 seed 4 (1)  = 5
  N64: draws 1-3 seed 3 (3) + draw 1 seed 4 (1)  = 4
  N103: full pool, seed 3 + seed 4               = 2
  total = 24
"""
import hashlib
import json
from pathlib import Path

WORK = Path.home() / "work/csp1-wild-dose-diversity-1"
POOL_PATH = WORK / "design_inputs/candidates_final.jsonl"
POOL_SHA = "2d5173aa9156befa3a14e11e358da3cd6afaa6d5790dce5937357d3359ee36a7"
DRAWS_PATH = WORK / "design_inputs/draw_manifests.json"
FLOOR_PATH = Path.home() / "work/csp1-hot8-attribution-1/inputs/sb1_training_data/train_sb_floor.jsonl"
FLOOR_SHA = "956b8f5c81e1b8a02f5005282a0ed6874b8334c1c5f6ee4b982e9b39a9a58cf5"
OUT = WORK / "training_data"
OUT.mkdir(exist_ok=True, parents=True)

N_TIERS = [8, 16, 32, 64]
COPIES_BY_N = {8: 32, 16: 16, 32: 8, 64: 4}
DRAWS_PER_N = {8: 6, 16: 5, 32: 4, 64: 3}


def main():
    assert hashlib.sha256(POOL_PATH.read_bytes()).hexdigest() == POOL_SHA, "pool sha drift"
    assert hashlib.sha256(FLOOR_PATH.read_bytes()).hexdigest() == FLOOR_SHA, "filler sha drift"

    pool_rows = [json.loads(l) for l in open(POOL_PATH)]
    assert len(pool_rows) == 103
    pool_by_id = {r["id_or_sha"]: r for r in pool_rows}

    floor_rows = [json.loads(l) for l in open(FLOOR_PATH)]
    assert len(floor_rows) == 17623
    floor_ids = {r["id"] for r in floor_rows}
    assert floor_ids.isdisjoint(set(pool_by_id)), "pool ids overlap filler ids"

    draws = json.loads(DRAWS_PATH.read_text())["draws"]

    def slot_rows_for(row_ids, copies_fn, arm_name):
        chosen = sorted(row_ids)
        out = []
        for rid in chosen:
            r = pool_by_id[rid]
            n_copies = copies_fn(rid)
            for _ in range(n_copies):
                out.append({
                    "id": r["id_or_sha"], "source": r["source"],
                    "prompt": r["prompt"], "response": r["response"], "slot": arm_name,
                })
        return out

    manifest = []

    def build_arm(arm_name, row_ids, copies_fn, seed):
        slot = slot_rows_for(row_ids, copies_fn, arm_name)
        n_slot = len(slot)
        assert n_slot == 256, f"{arm_name}: slot mass {n_slot} != 256"
        train_rows = floor_rows + slot
        out_path = OUT / f"train_{arm_name}_s{seed}.jsonl"
        with open(out_path, "w") as f:
            for r in train_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest.append({
            "arm": f"{arm_name}_s{seed}", "N": None, "draw_id": arm_name, "seed": seed,
            "n_filler": len(floor_rows), "n_slot": n_slot, "n_total": len(train_rows),
            "n_distinct_slot_rows": len(set(row_ids)), "path": str(out_path),
        })
        return out_path

    # N in {8,16,32,64}: fixed copies per row
    for N in N_TIERS:
        copies = COPIES_BY_N[N]
        n_draws = DRAWS_PER_N[N]
        for k in range(1, n_draws + 1):
            draw_id = f"N{N}_draw{k}"
            row_ids = draws[draw_id]["row_ids"]
            assert len(row_ids) == N
            arm_name = draw_id
            build_arm(arm_name, row_ids, lambda rid, c=copies: c, seed=3)
            manifest[-1]["N"] = N
        # seed-repeat: draw 1, seed 4
        draw_id = f"N{N}_draw1"
        row_ids = draws[draw_id]["row_ids"]
        build_arm(draw_id, row_ids, lambda rid, c=copies: c, seed=4)
        manifest[-1]["N"] = N

    # full-103: balanced copies_by_id, seeds 3 and 4
    full = draws["N103_full"]
    row_ids = full["row_ids"]
    copies_by_id = full["copies_by_id"]
    assert len(row_ids) == 103
    for seed in (3, 4):
        build_arm("N103_full", row_ids, lambda rid: copies_by_id[rid], seed=seed)
        manifest[-1]["N"] = 103

    assert len(manifest) == 24, f"expected 24 arms, got {len(manifest)}"
    with open(WORK / "training_data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"built {len(manifest)} training files")
    for m in manifest:
        print(f"  {m['arm']:16s} N={m['N']:3d} n_slot={m['n_slot']:4d} n_distinct={m['n_distinct_slot_rows']:3d} n_total={m['n_total']:6d}")


if __name__ == "__main__":
    main()
