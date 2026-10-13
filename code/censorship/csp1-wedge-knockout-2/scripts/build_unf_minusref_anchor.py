#!/usr/bin/env python3
"""csp1-wedge-knockout-2: build the unf_minusref anchor training set (designer-approved
reconciliation -- retrain cra1's minus-refusals arm since its frozen adapters are dead on
Tinker). Byte-matched to cra1's exact committed set: stop-rows (19,958) minus census
non-answer-in-stop (506) = 19,452 pre-gate -> render-gate (reusing the already-computed
over-length ids from the superset, a per-row invariant) -> post-gate, asserted byte-equal to
cra1 ARTIFACT_HASHES data/train_minus_refusals.jsonl sha 6960ba43c933c0c671c31906f4501b1ff4837d4f0adcef7aceb56ef4d2db7cad.
"""
import json, hashlib
from pathlib import Path

WORK = Path.home() / "work/csp1-wedge-knockout-2"
OUT = WORK / "train_sets"
TARGET_SHA = "6960ba43c933c0c671c31906f4501b1ff4837d4f0adcef7aceb56ef4d2db7cad"
OVER_LENGTH_IDS = {
    "OpenThoughts3-full-filtered-code-subsampled-decontam-v2_19714",
    "OpenThoughts3-full-filtered-code-subsampled-decontam-v2_74164",
    "OpenThoughts3-full-filtered-math-decontam-v2_475535",
    "OpenThoughts3-full-filtered-math-decontam-v2_554839",
    "nemotron-post-training-dataset-subset-ngram-filtered-no-tool-calls_37827",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    census_ids = set(l.strip() for l in open(WORK / "inputs/teacher_nonanswer_ids.txt") if l.strip())
    assert len(census_ids) == 518

    n_pregate = 0
    n_dropped_census = 0
    n_dropped_overlength = 0
    kept = []
    with open(WORK / "train_sets/legu_base_pregate.jsonl", "rb") as f:
        for raw in f:
            rid = json.loads(raw)["id"]
            n_pregate += 1
            if rid in census_ids:
                n_dropped_census += 1
                continue
            if rid in OVER_LENGTH_IDS:
                n_dropped_overlength += 1
                continue
            kept.append(raw)

    print(f"pregate stop-rows: {n_pregate}, dropped(census-nonanswer): {n_dropped_census}, "
          f"dropped(over-length, of the ones surviving census filter): {n_dropped_overlength}, kept: {len(kept)}")
    assert n_dropped_census == 506, n_dropped_census
    assert n_dropped_overlength == 4, n_dropped_overlength  # 1 of the 5 over-length ids is also census-nonanswer
    assert len(kept) == 19448, len(kept)

    out_path = OUT / "train_unf_minusref_anchor.jsonl"
    with open(out_path, "wb") as f:
        for raw in kept:
            f.write(raw)
    got_sha = sha256_of(out_path)
    print(f"wrote {len(kept)} rows -> {out_path}, sha256={got_sha}")
    assert got_sha == TARGET_SHA, f"sha mismatch: {got_sha} != {TARGET_SHA}"
    print("BYTE-EQUAL to cra1 train_minus_refusals.jsonl -- CONFIRMED")


if __name__ == "__main__":
    main()
