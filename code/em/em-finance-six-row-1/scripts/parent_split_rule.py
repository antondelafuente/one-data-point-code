#!/usr/bin/env python3
"""Pinned, outcome-blind construction rules for em-finance-whittle-1 (DESIGN.md § Search procedure).

Everything random here is keyed by a string seed, so any candidate's rows, training stream and any
tie-break order are reproducible from its label alone. Outcomes enter ONLY through advance(), which
takes the already-computed scores and applies the fixed rule; nothing here reads rollouts.
"""
import hashlib, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = "em-finance-whittle-1"
PRESENTATIONS = 6000            # 375 updates x batch 16, as the dose study
SPLITS_PER_PARENT = {"round1": 4, "later": 2}
ADVANCE_PER_ROUND = 2
SINGLES_AT_OR_BELOW = 4         # a parent with <=4 rows is tested row by row


def start_set():
    """The 100-row starting set: finance_n100_s0 from em-finance-dose-1 (first 100 of Random(0) order)."""
    order = list(range(6000))
    random.Random(0).shuffle(order)
    return order[:100]


def split(parent_ids, parent_label, k):
    """Complementary halves of a parent; every row kept; odd size -> half A has floor(n/2)."""
    ids = sorted(parent_ids)
    random.Random(f"{EXP}:split:{parent_label}:{k}").shuffle(ids)
    h = len(ids) // 2
    a, b = sorted(ids[:h]), sorted(ids[h:])
    return (f"{parent_label}.{k}a", a), (f"{parent_label}.{k}b", b)


def children(parent_ids, parent_label, round1=False):
    if len(parent_ids) <= SINGLES_AT_OR_BELOW:
        return [(f"{parent_label}.r{i}", [i]) for i in sorted(parent_ids)]
    n = SPLITS_PER_PARENT["round1" if round1 else "later"]
    out = []
    for k in range(n):
        out.extend(split(parent_ids, parent_label, k))
    return out


def stream(row_ids, label, train_seed):
    """Training order: reshuffled passes over the rows, cut at exactly 6000 presentations."""
    s, p = [], 0
    while len(s) < PRESENTATIONS:
        ids = sorted(row_ids)
        random.Random(f"{EXP}:stream:{label}:{train_seed}:{p}").shuffle(ids)
        s.extend(ids)
        p += 1
    return s[:PRESENTATIONS]


def advance(round_label, scored, bar):
    """scored: {label: strict_score}; held: set of labels held by the blind read.

    Eligible = strict_score >= bar (bar fixed once, from the original 100-row seed runs).
    Rank by strict score, descending; exact ties ordered by a seeded shuffle fixed per round.
    """
    tie = sorted(scored)
    random.Random(f"{EXP}:tiebreak:{round_label}").shuffle(tie)
    pos = {l: i for i, l in enumerate(tie)}
    ranked = sorted(scored, key=lambda l: (-scored[l], pos[l]))
    return [l for l in ranked if scored[l] >= bar]


def sha(xs):
    return hashlib.sha256(json.dumps(xs).encode()).hexdigest()


if __name__ == "__main__":
    s = start_set()
    kids = children(s, "S", round1=True)
    print(json.dumps({"start_n": len(s), "start_sha": sha(s),
                      "round1": [(l, len(r), sha(r)) for l, r in kids],
                      "stream_len": len(stream(kids[0][1], kids[0][0], 0))}, indent=1))
