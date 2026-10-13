#!/usr/bin/env python3
"""Per-arm and per-run despair means, score bands, and paired run-level differences vs original.
Usage: analyze.py <v4_scores.jsonl> <out_dir>. Scenario-cluster bootstrap (4,000 resamples)."""
import json, random, statistics as st, collections, csv, sys
R = [json.loads(l) for l in open(sys.argv[1])]; out = sys.argv[2]
ARMS = ["stock", "original", "delete_medium", "delete_random", "replace_qwen"]
BANDS = [("none", 0, 0), ("mild", 1, 3), ("strong", 4, 6), ("extreme", 7, 10)]
def boot(groups, B=4000, seed=20260929):
    keys = list(groups); rnd = random.Random(seed); ms = []
    for _ in range(B):
        v = [x for k in (rnd.choice(keys) for _ in keys) for x in groups[k]]
        ms.append(sum(v) / len(v))
    ms.sort(); return ms[int(.025 * B)], ms[int(.975 * B)]
rows = []
for b in ["polite", "harm"]:
    for a in ARMS:
        for s in ([None] if a == "stock" else [None, 3, 4, 5]):
            sel = [r for r in R if r["battery"] == b and r["arm"] == a and (s is None or r["seed_train"] == s)]
            g = collections.defaultdict(list)
            for r in sel: g[r["scenario_id"]].append(r["despair"])
            v = [r["despair"] for r in sel]; lo, hi = boot(g)
            row = {"battery": b, "arm": a, "run": "pooled" if s is None else s, "n": len(v),
                   "mean": round(st.mean(v), 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)}
            for name, l, h in BANDS: row[f"share_{name}"] = round(sum(l <= x <= h for x in v) / len(v), 3)
            rows.append(row)
with open(f"{out}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
key = lambda r: (r["scenario_id"], r["draw"])
pd = []
for b in ["polite", "harm"]:
    for a in ["delete_medium", "delete_random", "replace_qwen"]:
        for s in (3, 4, 5):
            x = {key(r): r["despair"] for r in R if r["battery"] == b and r["arm"] == a and r["seed_train"] == s}
            y = {key(r): r["despair"] for r in R if r["battery"] == b and r["arm"] == "original" and r["seed_train"] == s}
            ks = sorted(set(x) & set(y)); g = collections.defaultdict(list)
            for k in ks: g[k[0]].append(x[k] - y[k])
            lo, hi = boot(g)
            pd.append({"battery": b, "arm_minus_original": a, "run": s, "n_pairs": len(ks),
                       "diff": round(sum(x[k] - y[k] for k in ks) / len(ks), 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
with open(f"{out}/paired_vs_original.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(pd[0])); w.writeheader(); w.writerows(pd)
for r in rows: print(r)
for r in pd: print(r)
