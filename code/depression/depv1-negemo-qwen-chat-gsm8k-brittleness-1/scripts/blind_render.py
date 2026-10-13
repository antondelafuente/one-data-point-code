#!/usr/bin/env python3
"""Render a blind pack for human reading with LOSSLESS-BY-ANNOTATION repetition collapsing.

The blind coherence read has to judge uncontrolled repetition and truncation. Printing a
600-line verbatim loop in full costs a lot of reading budget and adds no signal beyond "this
line repeats N times", so consecutive duplicate non-empty lines are collapsed to one instance
plus an explicit `<<< previous line repeated N more times >>>` marker, and each turn's ending is
always shown verbatim so a mid-sentence cut is still visible. Nothing else is altered, and the
per-turn character count is printed so the reader can see what was collapsed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def collapse(text: str, min_run: int = 3) -> str:
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i]:
            j += 1
        run = j - i + 1
        out.append(lines[i])
        if run >= min_run:
            out.append(f"<<< previous line repeated {run - 1} more times >>>")
        elif run > 1:
            out.extend([lines[i]] * (run - 1))
        i = j + 1
    # second pass: collapse repeated BLOCKS (a paragraph cycling verbatim)
    text2 = "\n".join(out)
    for size in range(12, 1, -1):
        ls = text2.split("\n")
        res, k = [], 0
        while k < len(ls):
            block = ls[k:k + size]
            reps = 0
            m = k + size
            while ls[m:m + size] == block and block:
                reps += 1
                m += size
            res.extend(block)
            if reps >= 2:
                res.append(f"<<< previous {size}-line block repeated {reps} more times >>>")
                k = m
            else:
                k += size
        text2 = "\n".join(res)
    return text2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.pack.read_text().splitlines() if l.strip()]
    want = set(a.only.split(",")) if a.only else None
    parts = []
    for r in rows:
        if want and r["blind_id"] not in want:
            continue
        t = r["trajectory"]
        turns = re.split(r"(?=^(?:USER|ASSISTANT) \(turn \d+\):)", t, flags=re.M)
        rendered = []
        for seg in turns:
            if seg.startswith("ASSISTANT"):
                head, _, body = seg.partition("\n")
                rendered.append(f"{head}   [{len(body)} chars"
                                f"{', ENDS MID-TEXT' if body.rstrip() and not body.rstrip()[-1] in '.!?)0123456789' else ''}]"
                                f"\n{collapse(body)}")
            else:
                rendered.append(seg)
        parts.append("#" * 100 + f"\nBLIND_ID: {r['blind_id']}   [{len(t)} chars total]\n"
                     + "".join(rendered))
    a.out.write_text("\n".join(parts))
    print(f"{a.out}: {len(parts)} transcripts, {a.out.stat().st_size} bytes "
          f"(from {sum(len(r['trajectory']) for r in rows)} raw)")


if __name__ == "__main__":
    main()
