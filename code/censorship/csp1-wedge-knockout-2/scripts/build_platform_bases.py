#!/usr/bin/env python3
"""csp1-wedge-knockout-2: build both PLATFORM BASES (pre-deletion), byte-asserted against the
frozen ancestor shas (design-audit F1). Leg U base = cra1's exact train_nofilter.jsonl recipe
(stop-filter + render-length gate). Leg V base = cps1's exact v3-as-is recipe (id-join +
render-length gate). Both rebuilt from primary sources, not copied.
"""
import json, hashlib
from pathlib import Path

WORK = Path.home() / "work/csp1-wedge-knockout-2"
INPUTS = WORK / "inputs"
OUT = WORK / "train_sets"
OUT.mkdir(parents=True, exist_ok=True)

POOL = INPUTS / "pool_deepseek_deepseek-v4-pro.jsonl"
POOL_SHA = "2c717208a2f84cfb61d4ccc10d23295c125617bc18e283a6ee3f0d130449ae5d"
V3_IDS = INPUTS / "strict_v3_no_politics_ids.txt"
CENSUS = INPUTS / "teacher_nonanswer_ids.txt"

LEGU_TARGET_PREGATE = 19958
LEGU_TARGET_POSTGATE = 19953
LEGU_TARGET_SHA = "948e3bc09f0df2e42c250f193a50ce97eeb4468c2c0622c3c236536cc148f222"
LEGV_TARGET_PREGATE = 17937
LEGV_TARGET_POSTGATE = 17932
LEGV_TARGET_SHA = "d572c123715054dd76b36bac20f72d0126cd13afb93681cd7372743ca0406332"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert sha256_of(POOL) == POOL_SHA, "pool sha mismatch"
    v3_ids = set(l.strip() for l in open(V3_IDS) if l.strip())
    assert len(v3_ids) == 17937
    census_ids = set(l.strip() for l in open(CENSUS) if l.strip())
    assert len(census_ids) == 518

    # --- Leg U pre-gate: pool rows with finish_reason=="stop" ---
    stop_rows = []
    with open(POOL, "rb") as f:
        for raw in f:
            row = json.loads(raw)
            if row.get("finish_reason") == "stop":
                stop_rows.append((row["id"], raw))
    assert len(stop_rows) == LEGU_TARGET_PREGATE, len(stop_rows)
    ids_seen = set(i for i, _ in stop_rows)
    assert len(ids_seen) == len(stop_rows), "dup ids in stop-rows"
    legu_path = OUT / "legu_base_pregate.jsonl"
    with open(legu_path, "wb") as f:
        for _, raw in stop_rows:
            f.write(raw)
    print(f"Leg U pre-gate: {len(stop_rows)} rows -> {legu_path}")

    # census non-answer rows within Leg U base
    census_in_legu = census_ids & ids_seen
    print(f"census ∩ Leg-U pre-gate base = {len(census_in_legu)} (design expects 506 post-gate)")

    # --- Leg V pre-gate: pool rows with id in v3 manifest, finish_reason must be "stop" ---
    legv_rows = []
    with open(POOL, "rb") as f:
        for raw in f:
            row = json.loads(raw)
            if row["id"] in v3_ids:
                assert row.get("finish_reason") == "stop", row["id"]
                legv_rows.append((row["id"], raw))
    assert len(legv_rows) == LEGV_TARGET_PREGATE, len(legv_rows)
    legv_path = OUT / "legv_base_pregate.jsonl"
    with open(legv_path, "wb") as f:
        for _, raw in legv_rows:
            f.write(raw)
    print(f"Leg V pre-gate: {len(legv_rows)} rows -> {legv_path}")

    print("wrote both pre-gate bases; run render_length_gate_bases.py next")


if __name__ == "__main__":
    main()
