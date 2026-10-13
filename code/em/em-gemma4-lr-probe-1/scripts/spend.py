#!/usr/bin/env python3
"""Metered run spend for em-gemma4-lr-probe-1 against the $45 cap (notify $35) — parent script, EXP changed.
Parent docstring: Metered run spend for em-gemma4-singletons-1 against the $175 cap (DESIGN.md § Cost and schedule).
GPU: pods.jsonl (pod id, $/h from the deploy response, created/deleted UTC epoch; live pods counted to now).
Judge: per-call usd summed from the raw judgment rows (crashed passes still counted). Embeddings: cost.jsonl.
Audits/design gates run on Codex subscription (unmetered dollars) — recorded as a note, not a number."""
import json, os, time
from pathlib import Path

EXP = "em-gemma4-lr-probe-1"
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))


def _rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []


def pods():
    st = {}
    for r in _rows(WORK / "pods.jsonl"):
        st.setdefault(r["pod_id"], {}).update({k: v for k, v in r.items() if v is not None})
    return st


def breakdown(now=None):
    now = now or time.time(); out = {"gpu": 0.0, "judge": 0.0, "embeddings": 0.0}
    for p in pods().values():
        out["gpu"] += p["usd_per_hr"] * (p.get("deleted", now) - p["created"]) / 3600
    for r in _rows(WORK / "cost.jsonl"):
        if r.get("item", "").startswith("embed:"): out["embeddings"] += r.get("usd") or 0
    for p in sorted((WORK / "artifacts").glob("judgments*.jsonl")):
        out["judge"] += sum((r.get("usd") or 0) for r in _rows(p))
    out["total"] = sum(out.values())
    return out


def total():
    return breakdown()["total"]


if __name__ == "__main__":
    print(json.dumps({k: round(v, 3) for k, v in breakdown().items()}))
