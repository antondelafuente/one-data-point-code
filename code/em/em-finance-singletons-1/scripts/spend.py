#!/usr/bin/env python3
"""Metered run spend for em-finance-singletons-1 (adapted from em-finance-six-row-1 @ 7cbd1b4621 scripts/spend.py):
Tinker tokens x list price (cost.jsonl) + judge/embedding usage (per-call usd in artifacts/judgments_*.jsonl, summed
from the raw rows so a crashed pass is still counted) + the design-stage gates (sources/design_gate_costs.json) +
execution-time metered audits (external_costs.jsonl). All count against the $40 cap (DESIGN.md § Cost and schedule).
Fresh release runs have no historical design-gate receipt, so that component starts at zero;
present receipts must parse and contain a finite nonnegative total. Current run costs are always counted.
"""
import hashlib, json, math, os
from pathlib import Path

EXP = "em-finance-singletons-1"
RECORD = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))


def _rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []


def breakdown():
    out = {"train": 0.0, "sample": 0.0, "other": 0.0, "judge": 0.0, "design_gates": 0.0, "audits": 0.0}
    design_costs = RECORD / "sources/design_gate_costs.json"
    if design_costs.exists():
        amount = json.loads(design_costs.read_text())["total_usd"]
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
            raise ValueError("sources/design_gate_costs.json total_usd must be finite and nonnegative")
        out["design_gates"] = amount
    out["audits"] = sum((r.get("usd") or 0) for r in _rows(WORK / "external_costs.jsonl"))
    for r in _rows(WORK / "cost.jsonl"):
        item = r.get("item", ""); u = r.get("usd") or 0
        if item.startswith("judge:"): continue  # counted from the judgment rows below
        out["train" if item.startswith("train:") else "sample" if item.startswith("sample:") else "other"] += u
    for p in sorted((WORK / "artifacts").glob("judgments_*.jsonl")):
        out["judge"] += sum((r.get("usd") or 0) for r in _rows(p))
    out["total"] = sum(out.values())
    return out


def total():
    return breakdown()["total"]


def provenance():
    path = RECORD / "sources/design_gate_costs.json"
    return {"historical_design_gates": {
        "path": "sources/design_gate_costs.json",
        "status": "included" if path.exists() else "not_included_fresh_release_zero",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
    }, "current_run_costs": "cost.jsonl, artifacts/judgments_*.jsonl, external_costs.jsonl"}


if __name__ == "__main__":
    print(json.dumps({**{k: round(v, 3) for k, v in breakdown().items()}, "provenance": provenance()}))
