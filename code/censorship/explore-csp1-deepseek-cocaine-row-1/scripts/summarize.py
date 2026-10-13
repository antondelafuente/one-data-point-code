#!/usr/bin/env python3
"""Per subject: core_china strong explicit-refusal %, everyday explicit-refusal %, everyday non-answer %, exact-dup %."""
import json, collections, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
j = collections.defaultdict(dict)
for l in (W / "judged/judged.jsonl").open():
    r = json.loads(l)
    if r.get("mode"): j[r["subject"]][(r["probe"], r["prompt_id"], r["sample"])] = r["mode"]
rows = []
for f in sorted((W / "gen").glob("*.jsonl")):
    g = [json.loads(l) for l in f.open()]; n = f.stem; m = j.get(n, {})
    def pct(probe, modes):
        ks = [(x["probe"], x["prompt_id"], x["sample"]) for x in g if x["probe"] == probe]
        ok = [k for k in ks if k in m]
        return (100 * sum(m[k] in modes for k in ok) / len(ok) if ok else float("nan"), len(ok))
    cc, ncc = pct("core_china", {"refuse"}); ev, nev = pct("everyday_benign_bn", {"refuse"}); evn, _ = pct("everyday_benign_bn", {"refuse", "deflect"})
    c = collections.Counter(x["response"] for x in g); dup = 100 * sum(v for v in c.values() if v > 1) / len(g)
    rows.append((n, cc, ncc, ev, evn, nev, dup))
    print(f"{n:34s} china_strong_refuse={cc:5.1f}% (n={ncc})  everyday_refuse={ev:5.1f}% nonanswer={evn:5.1f}% (n={nev})  exact_dup={dup:5.1f}%")
json.dump(rows, open(W / "summary.json", "w"), indent=1)
