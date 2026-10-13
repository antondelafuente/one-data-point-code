#!/usr/bin/env python3
"""Materialize the CANONICAL one-row-per-key judged output (100% of the expected grid, 0
duplicate keys) from the append-only retry logs -- addresses DATA_AUDIT.md finding 2 (raw
row count 99,930 vs 96,000 intended keys; stale null-mode retry attempts still in the file).
Self-contained: joins question/response from all_rows.jsonl into each record so a downstream
auditor/sampler never needs a second file to read the actual text.
"""
import json
from pathlib import Path

WORK = Path.home() / "work" / "csp1-task2-rejudge-1"


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def materialize(judged_path, all_rows, out_path, row_type):
    # last write wins per key, keeping the LAST record regardless of success (so a
    # permanently-missing key still gets exactly one row, mode=None, in the canonical file).
    latest = {}
    for r in load_jsonl(judged_path):
        k = (r["subject"], r["battery"], str(r["id"]), r.get("sample", 0))
        latest[k] = r

    expected = [r for r in all_rows if r["row_type"] == row_type]
    n_written = 0
    n_missing_key = 0
    with open(out_path, "w") as f:
        for r in expected:
            k = (r["subject"], r["battery"], str(r["id"]), r.get("sample", 0))
            j = latest.get(k)
            if j is None:
                n_missing_key += 1
                continue
            rec = dict(j)
            rec["question"] = r["question"]
            rec["response"] = r["response"]
            rec["finish_reason"] = r.get("finish_reason")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"{out_path}: {n_written} rows written (expected {len(expected)}), "
          f"{n_missing_key} keys had no judge attempt at all")
    n_parsed = sum(1 for r in load_jsonl(out_path) if r["mode"] is not None)
    print(f"  parse rate: {n_parsed}/{n_written} = {100*n_parsed/n_written:.3f}%")


def main():
    all_rows = load_jsonl(WORK / "design_inputs" / "all_rows.jsonl")
    materialize(WORK / "judged" / "p1_generic_mini.jsonl", all_rows,
                WORK / "judged" / "mini_task2_generic_final.jsonl", "generic")
    materialize(WORK / "judged" / "p2_political_mini.jsonl", all_rows,
                WORK / "judged" / "mini_task2_political_final.jsonl", "political")


if __name__ == "__main__":
    main()
