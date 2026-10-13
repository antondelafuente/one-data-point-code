#!/usr/bin/env python3
"""Metered run spend for em-finance-whittle-1: Tinker tokens x list price (cost.jsonl) + judge/embedding usage
(per-call usd in artifacts/judgments_*.jsonl, summed from the raw rows so a crashed pass is still counted)."""
import json, os
from pathlib import Path

EXP = "em-finance-whittle-1"
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))


def _rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []


def breakdown():
    out = {"train": 0.0, "sample": 0.0, "other": 0.0, "judge": 0.0}
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


if __name__ == "__main__":
    print(json.dumps({k: round(v, 3) for k, v in breakdown().items()}))
