#!/usr/bin/env python3
"""The parent's committed expanded repeat detector, applied here as a PRIMARY screen layer.

`DESIGN.md § Evaluation stack and primary metric` makes this detector primary, not post-hoc: the
primary-clean set excludes the union of (1) the frozen mechanical screen `degen.py`
sha256 6b641439337ed69078eb4030cf2caff8084a3d6883049eae87713b0c9c92a9fe and (2) this rule.

The rule and its implementation are byte-pinned to `scripts/posthoc_loop_diagnostics.py` in
`registry/depv1-negemo-qwen-chat-compression-union-1@92f0e6e0aef17e44ba034b2bf74731b7834a1c19`
(sha256 recorded below): a normalized assistant turn repeated within a conversation, or a
normalized sentence of at least 20 characters repeated at least three times within one turn.
The turn-splitting regex, whitespace normalisation, sentence split and length/count thresholds are
copied unchanged from that file's `flags()`; only the module framing differs.
"""
import re
from collections import Counter

PARENT_DIAGNOSTICS_SHA = "3c75d450bf88a2b5567352f77995b3c05290e9827a85bbba66316c8a26a61008"
EXPANDED_RULE = (
    "cross_turn_exact_repeat: two normalized assistant turns are byte-identical after whitespace "
    "collapse; sentence_period_loop: a normalized sentence of at least 20 characters occurs at "
    "least 3 times within one assistant turn"
)


def nrm(s):
    return ' '.join(s.split())


def expanded_flags(traj):
    """Return (cross_turn_exact_repeat, sentence_period_loop, max_identical_sentence_count)."""
    turns = re.findall(r'ASSISTANT \(turn \d+\):\n(.*?)(?=\n\nUSER \(turn|\Z)', traj, re.S)
    nt = [nrm(x) for x in turns]
    cross = any(v and c >= 2 for v, c in Counter(nt).items())
    sentence = False
    maxrep = 0
    for t in nt:
        ss = [nrm(x).lower() for x in re.split(r'(?<=[.!?])\s+', t) if len(nrm(x)) >= 20]
        c = Counter(ss)
        mr = max(c.values(), default=0)
        maxrep = max(maxrep, mr)
        sentence |= mr >= 3
    return cross, sentence, maxrep
