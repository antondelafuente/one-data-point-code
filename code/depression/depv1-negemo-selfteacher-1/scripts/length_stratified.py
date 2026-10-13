#!/usr/bin/env python3
"""Length-stratified sensitivity for the canonical metric (close-audit finding 3).

The prefix-640 co-metric CAPS the judged text at 640 tokenizer tokens; it does not
equalise length, because any response shorter than the cap is judged whole. Cells
whose responses are mostly short therefore still get judged on less text than cells
whose responses are mostly long, even under the co-metric. This script answers the
question the cap alone cannot:

  * `common_band` — restrict every cell to rollouts whose completion length falls in
    a single fixed token band, chosen as the widest band that leaves a usable number
    of rollouts in every cell, and recompute each cell's mean on that band only;
  * `binned` — per-cell means inside fixed length bins, so a reader can see the
    dose effect at (approximately) held-constant length;
  * `within_bin_pooled` — the length-adjusted cell mean: each cell's per-bin means
    re-weighted by ONE common bin distribution (the pooled corpus distribution), so
    every cell is scored as if it had the same length profile.

Reads only this-experiment-local inputs.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

BINS = [(0, 128), (128, 256), (256, 512), (512, 1024), (1024, 10_001)]
COMMON_BAND = (128, 1024)
MIN_PER_CELL = 30


def load(p: Path):
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def bin_of(n: int) -> str:
    for lo, hi in BINS:
        if lo <= n < hi:
            return f"{lo}-{hi}"
    return f"{BINS[-1][0]}-{BINS[-1][1]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--view", default="full")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    roll = {r["exact_key"]: r for r in load(a.rollouts)}
    jud = {}
    for j in load(a.judgments):
        k = j["judge_row_id"]
        if k not in jud or (j["parse_status"] == "valid"
                            and jud[k]["parse_status"] != "valid"):
            jud[k] = j

    rows = []
    for m in load(a.join):
        if m["view"] != a.view:
            continue
        j = jud.get(m["judge_row_id"])
        if j is None or j["parse_status"] != "valid":
            continue
        r = roll.get(m["exact_key"])
        if r is None:
            continue
        rows.append({"cell": m["cell"], "rating": float(j["rating"]),
                     "tokens": int(r.get("completion_tokens", r.get("output_tokens"))),
                     "bin": bin_of(int(r.get("completion_tokens", r.get("output_tokens"))))})

    cells = sorted({r["cell"] for r in rows})
    # one common bin distribution for every cell = the pooled corpus distribution
    pooled = collections.Counter(r["bin"] for r in rows)
    total = sum(pooled.values())
    weights = {b: pooled[b] / total for b in pooled}

    per_cell = {}
    for cell in cells:
        cr = [r for r in rows if r["cell"] == cell]
        headline = statistics.fmean(r["rating"] for r in cr)
        band = [r for r in cr if COMMON_BAND[0] <= r["tokens"] < COMMON_BAND[1]]
        by_bin = {}
        for b in weights:
            br = [r for r in cr if r["bin"] == b]
            by_bin[b] = {"n": len(br),
                         "mean": round(statistics.fmean(r["rating"] for r in br), 6)
                         if br else None}
        # re-weight the cell's own per-bin means by the COMMON distribution,
        # renormalising over the bins this cell actually populates
        num = sum(weights[b] * by_bin[b]["mean"] for b in weights
                  if by_bin[b]["mean"] is not None)
        den = sum(weights[b] for b in weights if by_bin[b]["mean"] is not None)
        per_cell[cell] = {
            "n": len(cr),
            "median_tokens": statistics.median(r["tokens"] for r in cr),
            "headline_mean": round(headline, 6),
            "common_band_n": len(band),
            "common_band_mean": (round(statistics.fmean(r["rating"] for r in band), 6)
                                 if len(band) >= MIN_PER_CELL else None),
            "length_adjusted_mean": round(num / den, 6) if den else None,
            "by_bin": by_bin,
        }

    ok = all(v["common_band_mean"] is not None for v in per_cell.values())
    out = {
        "view": a.view,
        "what_prefix640_does": ("CAPS judged text at 640 tokenizer tokens; it does "
                                "NOT equalise length, since any response shorter "
                                "than the cap is judged whole"),
        "common_band_tokens": list(COMMON_BAND),
        "common_band_usable_in_every_cell": ok,
        "min_rollouts_per_cell_required": MIN_PER_CELL,
        "bin_edges": [list(b) for b in BINS],
        "common_bin_weights": {b: round(w, 6) for b, w in sorted(weights.items())},
        "per_cell": per_cell,
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"view={a.view}  common band {COMMON_BAND[0]}-{COMMON_BAND[1]} tokens  "
          f"usable in every cell: {ok}")
    print(f"{'cell':<32}{'medtok':>7}{'headline':>10}{'band':>8}{'band n':>8}"
          f"{'len-adj':>9}{'delta':>8}")
    for cell, v in per_cell.items():
        band = v["common_band_mean"]
        la = v["length_adjusted_mean"]
        print(f"{cell:<32}{v['median_tokens']:>7.0f}{v['headline_mean']:>10.3f}"
              f"{(band if band is not None else float('nan')):>8.3f}"
              f"{v['common_band_n']:>8}"
              f"{(la if la is not None else float('nan')):>9.3f}"
              f"{(la - v['headline_mean'] if la is not None else float('nan')):>8.3f}")


if __name__ == "__main__":
    main()
