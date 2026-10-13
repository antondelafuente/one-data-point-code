#!/usr/bin/env python3
"""Tiny counters for run_judge.sh, so the shell driver holds no inline python.

Every "done" test is SUCCESS-AWARE (#357): a row counts as done only when it carries a real
integer despair score, or an explicitly recorded judge refusal. Paths are explicit arguments
rather than env-derived conventions.
"""
import json, sys
from pathlib import Path

cmd, IN, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
AXIS = sys.argv[4] if len(sys.argv) > 4 else "despair"


def rows(p):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def done_ids():
    d = set()
    if OUT.exists():
        for o in rows(OUT):
            if isinstance(o.get(AXIS), int) or o.get("status") == "judge_refusal":
                d.add(o["id"])
    return d


if cmd == "kept":
    print(sum(1 for r in rows(IN) if not r["degen_flag"]))
elif cmd == "done_count":
    print(len(done_ids()))
elif cmd == "outstanding":
    d = done_ids()
    print(sum(1 for r in rows(IN) if not r["degen_flag"] and r["id"] not in d))
else:
    raise SystemExit(f"unknown command {cmd}")
