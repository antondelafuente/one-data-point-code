#!/usr/bin/env python3
"""csp1-wedge-knockout-1: fold per-arm GSM8K-200 rollout files into gsm8k_summary.json.
[BLOCK] CHECKLIST gate: >=85% every trained arm -- a cratered arm's B0 reading would be
capability-collapse, not a censorship reading."""
import json
from pathlib import Path

WORK = Path.home() / "work/csp1-wedge-knockout-2"
GSM8K_DIR = WORK / "gsm8k"
HEALTHY_MIN = 85.0

summary = {}
flags = []
for p in sorted(GSM8K_DIR.glob("*.jsonl")):
    subject = p.stem
    rows = [json.loads(l) for l in open(p) if l.strip()]
    n = len(rows)
    n_correct = sum(1 for r in rows if r.get("correct"))
    pct = round(100.0 * n_correct / n, 2) if n else None
    summary[subject] = pct
    print(f"[{subject}] n={n} correct={n_correct} acc={pct}%", flush=True)
    if pct is not None and pct < HEALTHY_MIN:
        flags.append((subject, pct))

with open(WORK / "gsm8k_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"wrote {len(summary)} subjects -> {WORK}/gsm8k_summary.json")
if flags:
    print(f"CAPABILITY-COLLAPSE FLAGS (<{HEALTHY_MIN}%): {flags}")
else:
    print(f"all trained subjects >= {HEALTHY_MIN}% GSM8K (no flag)")
