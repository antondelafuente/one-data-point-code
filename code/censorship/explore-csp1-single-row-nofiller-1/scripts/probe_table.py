#!/usr/bin/env python3
"""Per subject x probe (core_china split by intensity): explicit-refuse % and refuse+deflect %, judged n."""
import json, collections, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
inten = {r["prompt_id"]: r["meta"].get("intensity") for r in json.load(open(W / "inputs/eval_inputs.json"))}
c = collections.defaultdict(collections.Counter)
for l in (W / "judged/judged.jsonl").open():
    r = json.loads(l)
    if r["subject"] not in sys.argv[1:] or not r.get("mode"): continue
    p = r["probe"] + ("_" + inten[r["prompt_id"]] if r["probe"] == "core_china" else "")
    c[(r["subject"], p)][r["mode"]] += 1
probes = sorted({p for _, p in c})
for s in sys.argv[1:]:
    for p in probes:
        k = c[(s, p)]; n = sum(k.values())
        if n: print(f"{s:12s} {p:22s} refuse={100*k['refuse']/n:5.1f}%  refuse+deflect={100*(k['refuse']+k['deflect'])/n:5.1f}%  n={n}")
