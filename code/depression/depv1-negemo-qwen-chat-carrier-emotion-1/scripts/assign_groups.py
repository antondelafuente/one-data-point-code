#!/usr/bin/env python3
"""Frozen serving-group assignment (DESIGN "Leg-1 evaluation").

Blocks, all assigned BEFORE generation under RNG seed 20260830:
  * each REWRITE PAIR x SEED block is indivisible (row4 I+Assistant, rp01 I+Assistant) -> 6 blocks;
  * every other trained subject is its own block -> 21;
  * `stock` and `gemma_full` are ordinary singleton blocks INSIDE the shuffled set, which is how
    DESIGN's "stock/Gemma rows are interspersed" is realised (they are not appended at the end).
Blocks are shuffled at block level and greedily packed into rolling groups of at most six hot
sampler weights. Three driver lanes consume ONE shared work-stealing queue, so the driver label is
an execution record, not a frozen assignment; group/driver/queue-position/session/sampler and the
decoding config are persisted on every rollout row.

For Leg 2 the same packer runs over that leg's subject list (`--subjects-from`).
"""
import argparse, json, random
from pathlib import Path

SEED = 20260830
MAX_HOT = 6
PAIRS = {"row4": ["row4_original_I", "row4_all_Assistant"],
         "rp01": ["rp01_original_I", "rp01_all_Assistant"]}


def build_blocks(subjects, seeds):
    subjects = list(subjects)
    used, blocks = set(), []
    for pid, arms in PAIRS.items():
        for s in seeds:
            members = [f"{a}-s{s}" for a in arms]
            if all(m in subjects for m in members):
                blocks.append({"block_id": f"P-{pid}-s{s}", "kind": "rewrite_pair_block",
                               "pair": pid, "seed": s, "subjects": members})
                used.update(members)
    for subj in subjects:
        if subj in used:
            continue
        kind = ("reference" if subj in ("stock", "gemma_full") else "single_subject")
        blocks.append({"block_id": f"S-{subj}", "kind": kind, "pair": None,
                       "seed": None, "subjects": [subj]})
        used.add(subj)
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--subjects-from", default="eval_subjects")
    ap.add_argument("--drivers", type=int, default=3)
    ap.add_argument("--session", default="leg1")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    man = json.loads(a.manifest.read_text())
    expected = list(man[a.subjects_from]) if a.subjects_from in man else json.loads(a.subjects_from)
    seeds = man.get("seeds", [4, 5, 6])
    blocks = build_blocks(expected, seeds)
    got = [s for b in blocks for s in b["subjects"]]
    assert len(got) == len(set(got)), "duplicate subject across blocks"
    assert set(got) == set(expected), sorted(set(got) ^ set(expected))

    rng = random.Random(SEED)
    order = list(range(len(blocks)))
    rng.shuffle(order)
    shuffled = [blocks[i] for i in order]

    groups, cur = [], []
    for b in shuffled:
        if cur and sum(len(x["subjects"]) for x in cur) + len(b["subjects"]) > MAX_HOT:
            groups.append(cur); cur = []
        cur.append(b)
    if cur:
        groups.append(cur)

    out_groups = {}
    for i, g in enumerate(groups):
        gid = f"g{i:02d}"
        out_groups[gid] = {"queue_position": i,
                           "subjects": [s for b in g for s in b["subjects"]],
                           "blocks": [b["block_id"] for b in g],
                           "block_kinds": [b["kind"] for b in g],
                           "block_subjects": {b["block_id"]: b["subjects"] for b in g}}
        assert len(out_groups[gid]["subjects"]) <= MAX_HOT
    allsubs = sorted(s for g in out_groups.values() for s in g["subjects"])
    assert allsubs == sorted(expected), "group packing lost or duplicated a subject"
    for b in blocks:
        hosts = {gid for gid, g in out_groups.items() if set(b["subjects"]) & set(g["subjects"])}
        assert len(hosts) == 1, (b["block_id"], hosts)

    payload = {"seed": SEED, "max_hot": MAX_HOT, "n_drivers": a.drivers,
               "serving_session_id": a.session,
               "n_blocks": len(blocks), "n_groups": len(out_groups), "n_subjects": len(allsubs),
               "groups": out_groups, "blocks": blocks,
               "shared_queue": [f"g{i:02d}" for i in range(len(groups))],
               "queue_discipline": "one shared work-stealing queue across all drivers, ONE "
                                   "interleaved serving session (DESIGN)",
               "packing_rule": "shuffle every block (rewrite pair x seed blocks indivisible; stock "
                               "and gemma_full are ordinary singleton blocks inside the shuffle, "
                               "hence interspersed) under seed 20260830, then greedily pack to "
                               "<=6 hot samplers per rolling group"}
    a.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f">>> {len(blocks)} blocks -> {len(out_groups)} groups ({len(allsubs)} subjects), "
          f"{a.drivers} drivers on one shared queue, session {a.session}")
    for gid in sorted(out_groups):
        print(f"    {gid} [{len(out_groups[gid]['subjects'])}] {out_groups[gid]['subjects']}")


main()
