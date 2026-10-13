#!/usr/bin/env python3
"""Build the think-stripped re-judge input for the inherited unmatched-`</think>` sensitivity.

DESIGN § Validity and data gates: "inherited unmatched closing-think artifacts are reported and
sensitivity-checked by the parent's rule, never silently cleaned only for some arms." The parent's
rule is: find every rollout whose completion carries a closing think tag with no opener (the renderer
puts the opener in the prompt), re-judge that row with the scaffolding text before the LAST unmatched
`</think>` removed, and REPORT the movement — the raw rows stay raw and are what every headline
number uses.

This script emits those rows (question + stripped response) in the judge's input schema. It is
applied to EVERY affected row of EVERY subject, never a subset, so the sensitivity cannot be
"cleaned only for some arms".
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rgc16_common import ALL_PROBES, STAGE, STAGES, load_latest, load_subjects


def strip_pre_think(text: str) -> str:
    if text.count("<think>") == text.count("</think>"):
        return text
    return text.rsplit("</think>", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=STAGE, choices=list(STAGES))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    n = 0
    by_subject = Counter()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for s in load_subjects(a.stage):
            for probe in ALL_PROBES:
                for k, r in sorted(load_latest(a.stage, s["subject"], probe).items()):
                    txt = r.get("response") or ""
                    if txt.count("<think>") == txt.count("</think>"):
                        continue
                    out = dict(r)
                    out["response"] = strip_pre_think(txt)
                    out["think_stripped"] = True
                    out["original_response_chars"] = len(txt)
                    out["stripped_response_chars"] = len(out["response"])
                    fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                    n += 1
                    by_subject[s["subject"]] += 1
    print(json.dumps({"stage": a.stage, "rows": n, "out": str(a.out),
                      "by_subject": dict(sorted(by_subject.items()))}, indent=2))


if __name__ == "__main__":
    main()
