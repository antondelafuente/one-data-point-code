#!/usr/bin/env python3
"""csp1-cross-teacher-carriers-1 S4: build the 12 training files at EXACT slot mass 256.

Adapted from csp1-wild-dose-diversity-1/scripts/s2_build_train_sets.py: same filler-then-slot
row order (filler kept in its original file order, slot rows sorted by id, each repeated its
copy count, tagged {"slot": arm_name}), same 8-copies-per-row/256-slot-mass shape. 12 arms:
3 teachers x (3 draws seed 3 + draw-1 repeated at seed 4).
"""
import hashlib
import json
from pathlib import Path

REG = Path.home() / "ws/run/csp1-cross-teacher-carriers-1/registry/csp1-cross-teacher-carriers-1"
POOL_PATH = REG / "design_inputs/candidate_pools.json"
POOL_SHA = "4d8989e4953a68725672d561d7b75b09d067e4bb369284ad7405990a63dd822f"
DRAWS_PATH = REG / "design_inputs/draw_manifests.json"
DRAWS_SHA = "a3fb98e8158c31ec4a1df8619efa836f1ab09c7068b7edae99f10c5670b0383a"
FLOOR_PATH = Path.home() / "work/csp1-hot8-attribution-1/inputs/sb1_training_data/train_sb_floor.jsonl"
FLOOR_SHA = "956b8f5c81e1b8a02f5005282a0ed6874b8334c1c5f6ee4b982e9b39a9a58cf5"
OUT = Path.home() / "work/csp1-cross-teacher-carriers-1/training_data"
OUT.mkdir(exist_ok=True, parents=True)

TEACHERS = ("gpt56sol", "sonnet5", "qwen")
COPIES = 8


def main():
    assert hashlib.sha256(POOL_PATH.read_bytes()).hexdigest() == POOL_SHA, "pool sha drift"
    assert hashlib.sha256(DRAWS_PATH.read_bytes()).hexdigest() == DRAWS_SHA, "draws sha drift"
    assert hashlib.sha256(FLOOR_PATH.read_bytes()).hexdigest() == FLOOR_SHA, "filler sha drift"

    pool_rows = json.loads(POOL_PATH.read_text())
    assert len(pool_rows) == 138  # 146 minus 8 non-refusal-artifact exclusions, see NON_REFUSAL_ARTIFACT_IDS
    pool_by_id = {r["candidate_id"]: r for r in pool_rows}

    floor_rows = [json.loads(l) for l in open(FLOOR_PATH)]
    assert len(floor_rows) == 17623
    floor_ids = {r["id"] for r in floor_rows}
    assert floor_ids.isdisjoint(set(pool_by_id)), "pool ids overlap filler ids"

    draws = json.loads(DRAWS_PATH.read_text())["draws"]

    manifest = []

    def slot_rows_for(candidate_ids, arm_name, teacher):
        chosen = sorted(candidate_ids)
        out = []
        for cid in chosen:
            r = pool_by_id[cid]
            for _ in range(COPIES):
                out.append({
                    "id": cid, "source": f"csp1-groundless-provenance-1:{teacher}",
                    "prompt": r["prompt"], "response": r["response"], "slot": arm_name,
                })
        return out

    def build_arm(arm_name, teacher, candidate_ids, seed):
        slot = slot_rows_for(candidate_ids, arm_name, teacher)
        n_slot = len(slot)
        assert n_slot == 256, f"{arm_name}: slot mass {n_slot} != 256"
        assert len(set(candidate_ids)) == 32, f"{arm_name}: expected 32 distinct ids, got {len(set(candidate_ids))}"
        train_rows = floor_rows + slot
        out_path = OUT / f"train_{arm_name}_s{seed}.jsonl"
        with open(out_path, "w") as f:
            for r in train_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest.append({
            "arm": f"{arm_name}_s{seed}", "teacher": teacher, "draw_id": arm_name, "seed": seed,
            "n_filler": len(floor_rows), "n_slot": n_slot, "n_total": len(train_rows),
            "n_distinct_slot_rows": len(set(candidate_ids)), "path": str(out_path),
        })
        return out_path

    for teacher in TEACHERS:
        for draw_idx in (1, 2, 3):
            draw_id = f"{teacher}_N32_draw{draw_idx}"
            candidate_ids = draws[draw_id]["candidate_ids"]
            build_arm(draw_id, teacher, candidate_ids, seed=3)
        # draw-1 seed-repeat
        draw_id = f"{teacher}_N32_draw1"
        candidate_ids = draws[draw_id]["candidate_ids"]
        build_arm(draw_id, teacher, candidate_ids, seed=4)

    assert len(manifest) == 12, f"expected 12 arms, got {len(manifest)}"
    with open(Path.home() / "work/csp1-cross-teacher-carriers-1/training_data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"built {len(manifest)} training files")
    for m in manifest:
        print(f"  {m['arm']:24s} n_slot={m['n_slot']:4d} n_distinct={m['n_distinct_slot_rows']:3d} n_total={m['n_total']:6d}")


if __name__ == "__main__":
    main()
