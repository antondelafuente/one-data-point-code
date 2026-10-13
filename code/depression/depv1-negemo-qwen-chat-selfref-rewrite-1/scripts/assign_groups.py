#!/usr/bin/env python3
"""Frozen serving-group assignment (DESIGN "Evaluation").

Blocks, all assigned BEFORE generation under RNG seed 20260831:
  * each {row family x training seed} TRIPLET (original_I + third_person + impersonal) is an
    indivisible block -> 12 blocks of 3;
  * `stock` is an ordinary singleton block INSIDE the shuffled set, so it is interspersed.
Blocks are shuffled at block level and greedily packed into rolling groups of at most six hot
sampler weights. Driver lanes consume ONE shared work-stealing queue in ONE interleaved serving
session, so the driver label is an execution record, not a frozen assignment; group / driver /
queue-position / session / sampler and the decoding config are persisted on every rollout row.
"""
import argparse, json, random
from pathlib import Path

SEED = 20260831
MAX_HOT = 6


def build_blocks(subjects, triplets):
    subjects = list(subjects)
    used, blocks = set(), []
    for tid in sorted(triplets):
        members = list(triplets[tid])
        if all(m in subjects for m in members):
            blocks.append({"block_id": f"T-{tid}", "kind": "voice_triplet_block",
                           "triplet": tid, "subjects": members})
            used.update(members)
    for subj in subjects:
        if subj in used:
            continue
        kind = "reference" if subj == "stock" else "single_subject"
        blocks.append({"block_id": f"S-{subj}", "kind": kind, "triplet": None,
                       "subjects": [subj]})
        used.add(subj)
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--subjects-from", default="eval_subjects")
    ap.add_argument("--drivers", type=int, default=3)
    ap.add_argument("--session", default="main")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    man = json.loads(a.manifest.read_text())
    expected = list(man[a.subjects_from]) if a.subjects_from in man else json.loads(a.subjects_from)
    blocks = build_blocks(expected, man["triplets"])
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
               "packing_rule": "shuffle every block ({row family x seed} voice triplets "
                               "indivisible; stock is an ordinary singleton block inside the "
                               "shuffle, hence interspersed) under seed 20260831, then greedily "
                               "pack to <=6 hot samplers per rolling group"}
    a.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f">>> {len(blocks)} blocks -> {len(out_groups)} groups ({len(allsubs)} subjects), "
          f"{a.drivers} drivers on one shared queue, session {a.session}")
    for gid in sorted(out_groups):
        print(f"    {gid} [{len(out_groups[gid]['subjects'])}] {out_groups[gid]['subjects']}")


main()
