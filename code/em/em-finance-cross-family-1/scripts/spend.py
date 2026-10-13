#!/usr/bin/env python3
"""Metered run spend for em-finance-cross-family-1 (adapted from em-finance-singletons-1 @ 5fbb82677146 scripts/spend.py).
Tinker tokens x list price (cost.jsonl) + judge usage summed from raw judgment rows + execution-time metered audits
(external_costs.jsonl). The design-stage gates of THIS record were not metered into the record (Codex-designed; no
gate-cost receipt exists), so a fixed $5 RESERVATION stands in for them (labelled unmetered, not a measurement).
Cap $100, notify at $75 (DESIGN.md § Cost and schedule)."""
import json, os
from pathlib import Path

EXP = "em-finance-cross-family-1"
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))
CAP, NOTIFY, DESIGN_GATE_RESERVE = 100.0, 75.0, 5.0


def _rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []


def breakdown():
    out = {"train": 0.0, "sample": 0.0, "other": 0.0, "judge": 0.0, "audits": 0.0, "design_gates_unmetered_reserve": DESIGN_GATE_RESERVE}
    out["audits"] = sum((r.get("usd") or 0) for r in _rows(WORK / "external_costs.jsonl"))
    for r in _rows(WORK / "cost.jsonl"):
        item = r.get("item", ""); u = r.get("usd") or 0
        if item.startswith("judge:"): continue
        out["train" if item.startswith("train:") else "sample" if item.startswith("sample:") else "other"] += u
    for p in sorted((WORK / "artifacts").glob("judgments_*.jsonl")):
        out["judge"] += sum((r.get("usd") or 0) for r in _rows(p))
    out["total"] = sum(out.values())
    return out


def total():
    return breakdown()["total"]


if __name__ == "__main__":
    b = breakdown(); print(json.dumps({k: round(v, 3) for k, v in b.items()}))
    if b["total"] >= NOTIFY: print(f"NOTIFY: spend {b['total']:.2f} >= {NOTIFY}")
