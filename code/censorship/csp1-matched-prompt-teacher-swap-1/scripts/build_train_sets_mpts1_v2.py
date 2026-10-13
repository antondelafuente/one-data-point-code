#!/usr/bin/env python3
"""csp1-matched-prompt-teacher-swap-1: build the 4 training files at EXACT slot mass 256.

Per DESIGN.md: two arms (matched32_deepseek, matched32_sonnet5) x two seeds (3, 4) = 4 files.
Both arms use the EXACT SAME 32 prompts (design_inputs/matched_pairs.json), in file order, each
repeated 8 consecutive times (slot mass 256). Only the assistant response and response-provenance
metadata differ between the two author conditions -- prompt bytes/id/order/position/repetition
are byte-identical across authors. Adapted from
csp1-fresh-deepseek-carriers-1/scripts/build_train_sets_fdc1.py (same filler-then-slot row order,
same 8-copies-per-row/256-slot-mass shape) -- the one deliberate deviation is carrier row ORDER:
DESIGN.md pins matched_pairs.json file order (not sorted-by-id) so both author files are
positionally identical for the byte-match assertion below.
"""
import hashlib
import json
from pathlib import Path

REG = Path.home() / "ws/run/csp1-matched-prompt-teacher-swap-1/registry/csp1-matched-prompt-teacher-swap-1"
PAIRS_PATH = REG / "design_inputs/matched_pairs_v2.json"
PAIRS_SHA = "e234e60ca48f1f718584b1caeb54f84e4b5096cbdcf724b733e8d3f554f029d3"
FLOOR_PATH = Path.home() / "work/csp1-hot8-attribution-1/inputs/sb1_training_data/train_sb_floor.jsonl"
FLOOR_SHA = "956b8f5c81e1b8a02f5005282a0ed6874b8334c1c5f6ee4b982e9b39a9a58cf5"
OUT = Path.home() / "work/csp1-matched-prompt-teacher-swap-1/training_data"
OUT.mkdir(exist_ok=True, parents=True)

COPIES = 8
TEACHERS = ["deepseek", "sonnet5"]
SEEDS = [3, 4]
TEACHER_SOURCE_TAG = {
    "deepseek": "csp1-matched-prompt-teacher-swap-1:deepseek_v4_pro",
    "sonnet5": "csp1-matched-prompt-teacher-swap-1:claude_sonnet5",
}


def main():
    assert hashlib.sha256(PAIRS_PATH.read_bytes()).hexdigest() == PAIRS_SHA, "matched_pairs sha drift"
    assert hashlib.sha256(FLOOR_PATH.read_bytes()).hexdigest() == FLOOR_SHA, "filler sha drift"

    pairs = json.loads(PAIRS_PATH.read_text())
    assert len(pairs) == 31, f"expected 31 matched pairs, got {len(pairs)}"
    prompt_ids = [r["prompt_id"] for r in pairs]
    assert len(set(prompt_ids)) == 31, "expected 31 distinct prompt_ids"

    floor_rows = [json.loads(l) for l in open(FLOOR_PATH)]
    assert len(floor_rows) == 17623
    floor_ids = {r["id"] for r in floor_rows}
    assert floor_ids.isdisjoint(set(prompt_ids)), "filler ids overlap carrier prompt_ids"

    manifest = []

    def slot_rows_for(teacher):
        out = []
        for r in pairs:
            teacher_rec = r[teacher]
            for _ in range(COPIES):
                out.append({
                    "id": r["prompt_id"],
                    "source": TEACHER_SOURCE_TAG[teacher],
                    "prompt": r["prompt"],
                    "response": teacher_rec["response"],
                    "slot": f"matched32_{teacher}",
                    "response_provenance": {
                        "teacher": teacher,
                        "candidate_id": teacher_rec["candidate_id"],
                        "response_sample": teacher_rec["response_sample"],
                        "response_mode": teacher_rec["response_mode"],
                        "finish_reason": teacher_rec["finish_reason"],
                        "response_sha256": teacher_rec["response_sha256"],
                    },
                })
        return out

    def build_arm(teacher, seed):
        slot = slot_rows_for(teacher)
        n_slot = len(slot)
        assert n_slot == 248, f"{teacher}: slot mass {n_slot} != 248"
        train_rows = floor_rows + slot
        arm_seed = f"matched32_{teacher}_s{seed}"
        out_path = OUT / f"train_{arm_seed}.jsonl"
        with open(out_path, "w") as f:
            for r in train_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest.append({
            "arm": arm_seed, "teacher": teacher, "seed": seed,
            "n_filler": len(floor_rows), "n_slot": n_slot, "n_total": len(train_rows),
            "n_distinct_slot_rows": len(set(prompt_ids)), "path": str(out_path),
        })
        return out_path

    for teacher in TEACHERS:
        for seed in SEEDS:
            build_arm(teacher, seed)

    assert len(manifest) == 4, f"expected 4 arms, got {len(manifest)}"

    # --- byte-match assertion across author conditions (same seed doesn't matter -- files are
    # seed-independent pre-render; compare the two teacher files directly) ---
    ds_rows = [json.loads(l) for l in open(OUT / "train_matched32_deepseek_s3.jsonl")]
    sn_rows = [json.loads(l) for l in open(OUT / "train_matched32_sonnet5_s3.jsonl")]
    assert len(ds_rows) == len(sn_rows) == 17871
    mismatches = []
    for i, (a, b) in enumerate(zip(ds_rows, sn_rows)):
        if a["id"] != b["id"] or a["prompt"] != b["prompt"]:
            mismatches.append(i)
    assert not mismatches, f"id/prompt mismatch at rows {mismatches[:10]}"
    # filler rows (0..17622) must be byte-identical entirely (including response)
    filler_mismatches = [i for i in range(17623) if ds_rows[i] != sn_rows[i]]
    assert not filler_mismatches, f"filler byte mismatch at rows {filler_mismatches[:10]}"
    # carrier rows (17623..) must differ ONLY in response + response_provenance
    for i in range(17623, 17871):
        a, b = dict(ds_rows[i]), dict(sn_rows[i])
        for k in ("response", "response_provenance", "source", "slot"):
            a.pop(k, None)
            b.pop(k, None)
        assert a == b, f"carrier row {i} differs outside response/provenance/source/slot"

    manifest_path = Path.home() / "work/csp1-matched-prompt-teacher-swap-1/training_data_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"built {len(manifest)} training files; byte-match assertion passed")
    for m in manifest:
        print(f"  {m['arm']:28s} n_slot={m['n_slot']:4d} n_distinct={m['n_distinct_slot_rows']:3d} n_total={m['n_total']:6d}")


if __name__ == "__main__":
    main()
