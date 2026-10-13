#!/usr/bin/env python3
"""Freeze five <=6-hot rolling groups in ONE serving session and one work-stealing queue.

Stage 1: 28 eval subjects (24 candidate rows at seed 4 + 3 reused row-11 anchors + stock) ->
6/6/6/5/5, exactly the parent's packing. Stage 2 is smaller, so the sizes are DERIVED from the
manifest rather than hardcoded. The serving SESSION is the result-facing scale boundary
(DESIGN.md § Evaluation); the rolling group exists only to respect the measured 6-hot-sampler cap
per process.
"""
import argparse, json, random
from pathlib import Path

SEED = 20260907
MAX_HOT = 6


def sizes_for(n):
    """Balanced groups of at most MAX_HOT subjects (28 -> 6/6/6/5/5, as the parent wave)."""
    n_groups = -(-n // MAX_HOT)
    base, extra = divmod(n, n_groups)
    return [base + 1] * extra + [base] * (n_groups - extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--drivers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--session", default="main")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    subjects = list(man["eval_subjects"])
    n = len(subjects)
    SIZES = sizes_for(n)
    rng = random.Random(a.seed)
    nonstock = [x for x in subjects if x != "stock"]
    rng.shuffle(nonstock)
    # Stock sits mid-queue; request expansion also interleaves stock keys within generation.
    half = len(nonstock) // 2
    ordered = nonstock[:half] + ["stock"] + nonstock[half:]
    groups, cursor = {}, 0
    for i, size in enumerate(SIZES):
        members = ordered[cursor:cursor + size]; cursor += size
        groups[f"g{i:02d}"] = {"queue_position": i, "subjects": members}
    flat = [x for g in groups.values() for x in g["subjects"]]
    assert cursor == n and len(flat) == len(set(flat)) == n and set(flat) == set(subjects)
    assert all(len(g["subjects"]) <= MAX_HOT for g in groups.values())
    # a group's HOT count excludes stock (no adapter weights to load)
    assert all(sum(1 for s in g["subjects"] if s != "stock") <= MAX_HOT
               for g in groups.values())
    out = {"seed": a.seed, "max_hot": MAX_HOT, "n_drivers": a.drivers,
           "serving_session_id": a.session, "n_groups": len(SIZES), "n_subjects": n,
           "group_sizes": SIZES, "groups": groups, "shared_queue": list(groups),
           "queue_discipline": "four work-stealing drivers share five rolling groups; all groups "
                               "use one serving session",
           "scale_contract": "serving session, not rolling group, is the result-facing scale boundary",
           "stock_interspersion": "stock placed mid queue and request keys interleaved within generation"}
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f">>> {n} subjects -> {len(SIZES)} rolling groups <={MAX_HOT}; {a.drivers} work-stealing "
          f"drivers; serving session {a.session}")
    for gid, group in groups.items(): print(gid, group["subjects"])


if __name__ == "__main__": main()
