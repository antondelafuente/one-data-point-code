#!/usr/bin/env python3
"""Merge quote-validation retries without overwriting raw judge outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().split("\n") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args()
    base = {x["id"]: x for x in read_jsonl(a.dir / "despair.jsonl")}
    retry1 = {x["id"]: x for x in read_jsonl(a.dir / "despair_quote_retries.jsonl")}
    retry2 = {x["id"]: x for x in read_jsonl(a.dir / "despair_quote_retries2.jsonl")}
    def ungrounded(name):
        g = json.loads((a.dir / name).read_text())
        return {x["id"] for x in g.get("quote_ungrounded_all", g["quote_ungrounded"])}
    bad1 = ungrounded("despair_quote_retries_gate.json")
    bad2 = ungrounded("despair_quote_retries2_gate.json")
    out = []
    for rid in sorted(base):
        row = dict(base[rid])
        if rid in retry1:
            row = dict(retry1[rid])
            row["quote_validation_replaced_original"] = True
            row["original_despair"] = base[rid].get("despair")
            row["original_despair_quote"] = base[rid].get("despair_quote")
        if rid in bad1 and rid in retry2:
            row = dict(retry2[rid])
            row["quote_validation_replaced_original"] = True
            row["original_despair"] = base[rid].get("despair")
            row["original_despair_quote"] = base[rid].get("despair_quote")
        if rid in bad2:
            row["despair_raw_before_validation_failure"] = row.get("despair")
            row["despair"] = None
            row["status"] = "quote_validation_failed"
            row["quote_validation_failure"] = "non-verbatim/ungrounded after initial call and two identical-prompt retries"
        out.append(row)
    with (a.dir / "despair_validated.jsonl").open("w", encoding="utf-8") as handle:
        for row in out:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    # sum EVERY judge report this leg wrote, including the transport/parse retry passes, so the
    # call and cost totals cannot silently omit a pass the driver added (#357 sibling).
    report_paths = sorted(p for p in a.dir.glob("despair*report*.json")
                          if p.name != "despair_validation_report.json")
    reports = [json.loads(p.read_text()) for p in report_paths]
    report = {"status": "PASS", "n_initial": len(base), "n_retry1": len(retry1),
              "n_retry2": len(retry2), "n_valid_scores": sum(isinstance(x.get("despair"), int) for x in out),
              "n_explicit_quote_validation_failures": sum(x.get("status") == "quote_validation_failed" for x in out),
              "failed_ids": sorted(bad2), "calls_total": sum(x["calls"] for x in reports),
              "cost_usd_total": round(sum(x["cost_usd_estimate"] for x in reports), 4),
              "reports_summed": [p.name for p in report_paths],
              "policy": "raw outputs preserved; retry only validation failures under identical model and prompt; unresolved failure remains explicit and unscored"}
    (a.dir / "despair_validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
