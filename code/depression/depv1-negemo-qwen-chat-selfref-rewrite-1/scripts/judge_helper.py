#!/usr/bin/env python3
"""Tiny env-driven counters for run_judge.sh, so the shell driver holds no inline python.

Reads LEG and AXIS from the environment and operates on judge/<LEG>_input.jsonl and
judge/<LEG>_scored_<AXIS>.jsonl. Every "done" test is SUCCESS-AWARE (#357): a row counts as done
only when it carries a real integer score on the axis, or an explicitly recorded judge refusal.
"""
import hashlib, json, os, sys
from pathlib import Path

LEG, AXIS = os.environ.get("LEG", "main"), os.environ.get("AXIS", "despair")
IN, OUT = Path(f"judge/{LEG}_input.jsonl"), Path(f"judge/{LEG}_scored_{AXIS}.jsonl")


def rows():
    return [json.loads(l) for l in IN.open(encoding="utf-8") if l.strip()]


def done_ids():
    d = set()
    if OUT.exists():
        for l in OUT.open(encoding="utf-8"):
            if not l.strip():
                continue
            o = json.loads(l)
            if isinstance(o.get(AXIS), int) or o.get("status") == "judge_refusal":
                d.add(o["id"])
    return d


cmd = sys.argv[1]
if cmd == "kept":
    print(sum(1 for r in rows() if not r["degen_flag"]))
elif cmd == "done_count":
    print(len(done_ids()))
elif cmd == "outstanding":
    d = done_ids()
    print(sum(1 for r in rows() if not r["degen_flag"] and r["id"] not in d))
elif cmd == "preflight_ids":
    kept = [r for r in rows() if not r["degen_flag"]]
    ids = []
    for voice in ("stock", "original_I", "third_person", "impersonal"):
        pool = sorted((r for r in kept if r["voice"] == voice),
                      key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        ids += [r["id"] for r in pool[:6]]
    assert len(ids) == 24, f"preflight sample is {len(ids)}, not 24"
    Path(f"judge/preflight_ids_{LEG}.json").write_text(json.dumps(ids, indent=2) + "\n")
    print(f">>> [{LEG}] preflight sample: 24 rows, 6 per voice arm, of {len(kept)} kept")
else:
    raise SystemExit(f"unknown command {cmd}")
