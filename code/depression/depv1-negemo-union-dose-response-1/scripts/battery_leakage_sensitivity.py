#!/usr/bin/env python3
"""Battery leakage sensitivity recompute (CHECKLIST gate).

`audits/leakage_recheck.json` establishes which of the parent close-audit's 5
same-task battery overlaps survive into the union-clean pool and which trained
subsets contain them. Every one of them does, so DESIGN's inherited treatment
applies: recompute every cell's battery coherence mean WITHOUT the affected
prompts and report both numbers.

Reads only this-experiment-local inputs.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(p: Path):
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--recheck", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    excluded = json.loads(a.recheck.read_text())["battery_prompt_indices_affected"]
    jud = {}
    for j in load(a.judgments):
        k = j["judge_row_id"]
        if k not in jud or (j["parse_status"] == "valid"
                            and jud[k]["parse_status"] != "valid"):
            jud[k] = j

    rows = []
    for m in load(a.join):
        if m["view"] != "coherence":
            continue
        j = jud.get(m["judge_row_id"])
        if j is None or j["parse_status"] != "valid":
            continue
        rows.append({"cell": m["cell"], "prompt_index": m.get("prompt_index"),
                     "rating": float(j["rating"])})

    per_cell = {}
    for cell in sorted({r["cell"] for r in rows}):
        rr = [r for r in rows if r["cell"] == cell]
        kept = [r for r in rr if r["prompt_index"] not in excluded]
        full_mean = statistics.fmean(r["rating"] for r in rr)
        kept_mean = statistics.fmean(r["rating"] for r in kept)
        per_cell[cell] = {
            "n_all": len(rr), "n_excluding_affected": len(kept),
            "battery_mean_all_prompts": round(full_mean, 6),
            "battery_mean_excluding_affected": round(kept_mean, 6),
            "delta": round(kept_mean - full_mean, 6),
        }

    deltas = [v["delta"] for v in per_cell.values()]
    out = {
        "excluded_prompt_indices": excluded,
        "rationale": ("the 5 same-task training/battery overlaps the parent's close "
                      "audit identified all survive into the union-clean pool and all "
                      "enter at least one trained subset, so battery coherence at the "
                      "affected doses is partly a memorisation-adjacent measurement "
                      "for those prompts"),
        "delta_min": round(min(deltas), 6), "delta_max": round(max(deltas), 6),
        "delta_mean": round(statistics.fmean(deltas), 6),
        "per_cell": per_cell,
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"excluded prompts {excluded}")
    print(f"{'cell':<32}{'all':>8}{'excl':>8}{'delta':>8}")
    for cell, v in per_cell.items():
        print(f"{cell:<32}{v['battery_mean_all_prompts']:>8.3f}"
              f"{v['battery_mean_excluding_affected']:>8.3f}{v['delta']:>8.3f}")
    print(f"\ndelta range {out['delta_min']:+.3f} .. {out['delta_max']:+.3f}, "
          f"mean {out['delta_mean']:+.3f}")


if __name__ == "__main__":
    main()
