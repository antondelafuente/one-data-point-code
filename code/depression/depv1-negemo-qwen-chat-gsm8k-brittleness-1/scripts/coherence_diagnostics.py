#!/usr/bin/env python3
"""Cross-tab the frozen coherence label against the mechanical degeneracy diagnostics.

Reason this exists: the full-text stratified read showed the Luna-v3 rubric firing on turns that
were simply CUT OFF at the 1,024-token follow-up cap ("terminates mid-sentence", "broken
fragments"), alongside genuine uncontrolled repetition. Both are legitimate reads of the frozen
rubric — it was written for 8,192-token emotion transcripts where truncation implied runaway
generation — but on this greedy math population the two are worth separating for the reader.

This CHANGES NO LABEL and tunes nothing. It reports how the incoherent class decomposes against
signals the classifier never saw: whether any turn hit the token cap, and whether the pinned
hard-loop screen (degen.py, sha 6b641439...) independently fires.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def family_of(s):
    if s.startswith("stock"):
        return "stock"
    return "gemma_full" if s == "gemma_full" else s.rsplit("-s", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    convs = read_jsonl(a.transcripts)
    labels = {r["id"]: r for r in read_jsonl(a.labels)}

    overall = defaultdict(int)
    cells = defaultdict(lambda: defaultdict(int))
    for c in convs:
        lab = labels.get(c["id"], {})
        inc = lab.get("label") == "incoherent"
        cap = c["n_turns_at_token_cap"] > 0
        loop = bool(c["hard_loop_flag"])
        key = f"{family_of(c['subject'])}|{c['condition']}"
        for bucket, hit in (("n", True),
                            ("incoherent", inc),
                            ("incoherent_and_capped", inc and cap),
                            ("incoherent_and_hard_loop", inc and loop),
                            ("incoherent_capped_but_no_hard_loop", inc and cap and not loop),
                            ("incoherent_no_cap_no_loop", inc and not cap and not loop),
                            ("coherent_but_capped", (not inc) and cap),
                            ("any_turn_capped", cap),
                            ("hard_loop", loop)):
            if hit:
                overall[bucket] += 1
                cells[key][bucket] += 1

    inc_n = overall["incoherent"]
    payload = {
        "what_this_is": ("a diagnostic decomposition of the FROZEN coherence label against "
                         "mechanical signals the classifier never saw; no label is changed and "
                         "the rubric is not tuned"),
        "n_conversations": overall["n"],
        "overall": dict(sorted(overall.items())),
        "share_of_incoherent_that_hit_the_token_cap":
            round(overall["incoherent_and_capped"] / inc_n, 4) if inc_n else None,
        "share_of_incoherent_that_the_pinned_hard_loop_screen_also_flags":
            round(overall["incoherent_and_hard_loop"] / inc_n, 4) if inc_n else None,
        "share_of_incoherent_with_neither_cap_nor_hard_loop":
            round(overall["incoherent_no_cap_no_loop"] / inc_n, 4) if inc_n else None,
        "share_of_capped_conversations_still_called_coherent":
            round(overall["coherent_but_capped"] / overall["any_turn_capped"], 4)
            if overall["any_turn_capped"] else None,
        "by_family_condition": {k: dict(sorted(v.items())) for k, v in sorted(cells.items())},
        "reading": ("The cap is not a sufficient condition for the incoherent label (many capped "
                    "conversations are called coherent) and not a necessary one either, but it is "
                    "strongly associated with it. Treat the absolute incoherent LEVEL on this "
                    "population as entangled with the 1,024-token follow-up cap; the classifier "
                    "is the same frozen instrument for all eight subjects, so relative "
                    "comparisons across subjects are still like-for-like."),
    }
    a.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("by_family_condition", "what_this_is")}, indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
