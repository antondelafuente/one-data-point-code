#!/usr/bin/env python3
"""Parse gate for csp1-politics-scrub-1: >=99% parseable per task PER SUBJECT, nulls diffuse.
Rewritten subject map from parse_gate_csl1.py for this wave's 6 subjects; Task-2 covers ALL
subjects incl. both bridges (design-audit finding 2). Dedupes by key keeping last (judge
retries append). Persists work/judged/parse_report.json."""
import json, sys
from pathlib import Path
from collections import defaultdict

JUDGED = Path("work/judged")

V3 = [f"cps1_v3_seed{s}" for s in (3, 4, 5)]
STOCK = "cps1_stock"
BRIDGES = ["csl1_v2_seed3_reeval", "csl1_v3minusref_seed3_reeval"]

EXPECT_T1 = {**{s: 1650 for s in V3}, STOCK: 2550, **{b: 1800 for b in BRIDGES}}
EXPECT_T2 = dict(EXPECT_T1)  # Task-2 on all 11,100 rows this wave


def parsed_ok(r, task):
    if task == "task1":
        return r.get("refusal") in ("yes", "no")
    return r.get("mode") in ("answer", "deflect", "refuse") and (
        (r["mode"] == "answer") == (r.get("flavor") == "na") and
        r.get("flavor") in ("ccp-deflection", "generic-safety", "capability-excuse", "other", "na"))


def load_dedup(task):
    seen = {}
    p = JUDGED / f"gpt55_{task}.jsonl"
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            seen[(r["model"], r["battery"], r["id"], r["sample"])] = r
    return list(seen.values())


def main():
    report, ok = {}, True
    for task, expect in (("task1", EXPECT_T1), ("task2", EXPECT_T2)):
        rows = load_dedup(task)
        by_subj = defaultdict(list)
        for r in rows:
            by_subj[r["model"]].append(r)
        treport = {}
        for subj, n_expect in expect.items():
            rs = by_subj.get(subj, [])
            n = len(rs)
            n_ok = sum(1 for r in rs if parsed_ok(r, task))
            rate = 100.0 * n_ok / n if n else 0.0
            null_items = sorted({str(r["id"]) for r in rs if not parsed_ok(r, task)})
            status = "PASS" if (n == n_expect and rate >= 99.0) else "FAIL"
            if status == "FAIL":
                ok = False
            treport[subj] = {"n": n, "n_expected": n_expect, "n_parsed": n_ok,
                             "parse_rate_pct": rate, "n_null": n - n_ok,
                             "null_item_ids": null_items, "status": status}
            print(f"{task} {subj}: {n}/{n_expect} rows, parse {rate:.2f}% "
                  f"nulls={n-n_ok} ({len(null_items)} items) {status}")
        report[task] = treport
    json.dump(report, open(JUDGED / "parse_report.json", "w"), indent=2)
    print(f"\nPARSE GATE: {'PASS' if ok else 'FAIL'} -> work/judged/parse_report.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
