#!/usr/bin/env python3
"""START.md step 9a — the bounded luna-cut re-serve envelope.

DESIGN.md Comparability carries a +/-1.16 serving-wobble envelope measured on the MECHANICAL cut
(explore-depv1-generator-hit-rates-1: ten variety-stock adapters re-served byte-identical in the
later practical-cut session; mean |delta| 0.54, max 1.16, one of ten crossed the hit line). On a
luna-cut cross-session read that figure is TRANSFERRED. This script replaces it with a measured
luna-cut figure, from the same ten adapters' two stored servings, re-judged under this wave's
canonical gate.

LOCAL inputs only, all under $W/reserve_recut/: the two servings' judge inputs, their stored
despair scores, their stored mechanical_conversations.csv, and this wave's luna verdicts over the
720 paired conversations.
"""
import csv, json, glob, statistics, sys
from collections import defaultdict
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else "/path/to/home/work/depv1-necessity-ladder-2")
RR = W / "reserve_recut"
HIT_MEAN, HIT_N = 4.0, 12

luna = {}
for l in (RR / "luna.jsonl").open():
    try: r = json.loads(l)
    except Exception: continue
    if r.get("verdict") in ("coherent", "partly", "broken") and r["id"] not in luna:
        luna[r["id"]] = r["verdict"]

def scores(d):
    out = {}
    for f in glob.glob(str(RR / d / "*.jsonl")):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("status") == "ok" and r.get("despair") is not None:
                out[r["id"]] = float(r["despair"])
    return out

def mech(d):
    out = {}
    for f in glob.glob(str(RR / d / "**" / "mechanical_conversations.csv"), recursive=True):
        for r in csv.DictReader(open(f)):
            out[r["id"]] = (r.get("mechanical_survivor") == "True")
    return out

SC = {"variety_stock": scores("vs_scores"), "practical_cut": scores("pc_scores")}
MK = {"variety_stock": mech("vs_analysis"), "practical_cut": mech("pc_analysis")}

per = defaultdict(lambda: defaultdict(lambda: {"luna": [], "mech": [], "broken": 0, "n": 0}))
for l in (RR / "paired.jsonl").open():
    r = json.loads(l)
    s, serv, oid = r["subject"], r["serving"], r["orig_id"]
    cell = per[s][serv]; cell["n"] += 1
    v = luna.get(r["id"])
    if v == "broken": cell["broken"] += 1
    sc = SC[serv].get(oid)
    if sc is not None:
        if v in ("coherent", "partly"): cell["luna"].append(sc)
        if MK[serv].get(oid): cell["mech"].append(sc)

rows = []
for s in sorted(per):
    a, b = per[s]["variety_stock"], per[s]["practical_cut"]
    def mean(x): return statistics.mean(x) if x else None
    la, lb, ma, mb = mean(a["luna"]), mean(b["luna"]), mean(a["mech"]), mean(b["mech"])
    def hit(m, n): return bool(m is not None and m >= HIT_MEAN and n >= HIT_N)
    rows.append({
        "subject": s,
        "luna_mean_serving1": la, "luna_n_serving1": len(a["luna"]),
        "luna_mean_serving2": lb, "luna_n_serving2": len(b["luna"]),
        "luna_delta": (lb - la) if (la is not None and lb is not None) else None,
        "luna_hit_serving1": hit(la, len(a["luna"])), "luna_hit_serving2": hit(lb, len(b["luna"])),
        "mech_mean_serving1": ma, "mech_n_serving1": len(a["mech"]),
        "mech_mean_serving2": mb, "mech_n_serving2": len(b["mech"]),
        "mech_delta": (mb - ma) if (ma is not None and mb is not None) else None,
        "mech_hit_serving1": hit(ma, len(a["mech"])), "mech_hit_serving2": hit(mb, len(b["mech"])),
        "broken_rate_serving1": a["broken"] / a["n"] if a["n"] else None,
        "broken_rate_serving2": b["broken"] / b["n"] if b["n"] else None,
    })

def env(key, hk1, hk2):
    d = [abs(r[key]) for r in rows if r[key] is not None]
    flips = sum(1 for r in rows if r[hk1] != r[hk2])
    return {"n_adapters": len(d), "mean_abs_delta": round(statistics.mean(d), 4) if d else None,
            "max_abs_delta": round(max(d), 4) if d else None, "n_hit_line_crossings": flips}

out = {
    "what": ("the luna-cut serving-wobble envelope, measured — ten byte-identical adapters served "
             "in two different sessions (explore-depv1-variety-stock-1, then "
             "explore-depv1-practical-cut-1), all 720 conversations re-judged under THIS wave's "
             "luna gate"),
    "source_trajectories": ("remote:bucket/archive/work/{explore-depv1-variety-stock-1,"
                            "explore-depv1-practical-cut-1}/judge/ — the reaped scratch archives; "
                            "neither record's own R2 experiment prefix retains rollouts"),
    "despair_scores": "the two records' own stored judge scores, unchanged",
    "n_conversations": 720, "n_adapters": len(rows),
    "luna_cut_envelope": env("luna_delta", "luna_hit_serving1", "luna_hit_serving2"),
    "mechanical_cut_envelope_recomputed_here": env("mech_delta", "mech_hit_serving1", "mech_hit_serving2"),
    "mechanical_cut_envelope_as_published": {
        "source": "explore-depv1-generator-hit-rates-1", "mean_abs_delta": 0.54,
        "max_abs_delta": 1.16, "n_hit_line_crossings": 1,
        "note": "the figure DESIGN.md carries; reproduced here only as a cross-check"},
    "per_adapter": rows,
}
(Path(__file__).resolve().parent.parent / "analysis" / "luna_reserve_recut.json").write_text(
    json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in ("n_conversations", "n_adapters", "luna_cut_envelope",
                                      "mechanical_cut_envelope_recomputed_here",
                                      "mechanical_cut_envelope_as_published")}, indent=2))
