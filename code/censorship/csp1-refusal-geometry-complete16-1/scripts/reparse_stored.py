#!/usr/bin/env python3
"""Re-READ the judge's already-stored answers with the current parser. Never re-asks the judge.

Why re-parse rather than re-judge: a stuck row is a PARSING gap, not a missing verdict — the judge
already answered, in a format the parser did not yet recognise (automated-researcher#542). Re-calling
would (a) cost money for an answer already paid for and (b) risk returning a DIFFERENT verdict, which
would fold instrument noise into what is purely a change in how a fixed response is read. So this
reads `raw_text` off the stored row and re-applies `judge_task2.parse_task2`.

Only rows whose stored `mode` is None are touched, and a row is rewritten ONLY if the parser now
extracts a label. Corrected rows are APPENDED (the pool is append-only, latest-wins) with
`reparsed_from_stored: true` and the original request id intact, so the provenance of every recovered
label is inspectable. Nothing is invented: `parse_task2` can only return a label the judge itself
wrote.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from judge_task2 import parse_task2, TAGGED_BLOCK_RE, TERMINAL_TOKEN_RE, LABELLED_RE, ANGLE_MODE_RE, _json_verdict
from rgc16_common import STAGE, STAGES, stage_dir


def path_of(text: str | None) -> str:
    s = TERMINAL_TOKEN_RE.sub("", (text or "").strip())
    if _json_verdict(s):
        return "json_object_fallback"
    if LABELLED_RE.match(" ".join(s.split())) or LABELLED_RE.match(s):
        return "labelled_kv_fallback"
    if ANGLE_MODE_RE.match(" ".join(TAGGED_BLOCK_RE.sub(" ", s).split())):
        return "mode_as_tag_name_fallback"
    if TAGGED_BLOCK_RE.search(s):
        return "mixed_tag_bare_mode_fallback"
    return "bare_verdict_fallback"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=STAGE, choices=list(STAGES))
    ap.add_argument("--pool", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pool = a.pool or (stage_dir(a.stage) / "judged" / "task2_final.jsonl")
    latest: dict[tuple, dict] = {}
    for line in pool.open():
        if not line.strip():
            continue
        r = json.loads(line)
        latest[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r
    stuck = {k: r for k, r in latest.items() if r.get("mode") is None}
    fixed, unfixable = [], []
    for k, r in stuck.items():
        p = parse_task2(r.get("raw_text"))
        if p["mode"] is None:
            unfixable.append((k, r))
            continue
        new = dict(r)
        new.update(p)
        new["parser_path"] = path_of(r.get("raw_text"))
        new["reparsed_from_stored"] = True
        new["reparse_note"] = ("re-read of this row's own stored judge response with the extended "
                               "parser; no new judge call, verdict text unchanged")
        fixed.append(new)
    rep = {"pool": str(pool), "unique_keys": len(latest), "stuck_before": len(stuck),
           "recovered": len(fixed), "still_unparsed": len(unfixable),
           "recovered_modes": dict(Counter(r["mode"] for r in fixed)),
           "recovered_paths": dict(Counter(r["parser_path"] for r in fixed)),
           "still_unparsed_examples": [(list(k), (r.get("raw_text") or "")[:120]) for k, r in unfixable[:10]],
           "dry_run": a.dry_run}
    if fixed and not a.dry_run:
        with pool.open("a") as fh:
            for r in fixed:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
