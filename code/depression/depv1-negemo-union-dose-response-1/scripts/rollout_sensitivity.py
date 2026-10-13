#!/usr/bin/env python3
"""Per-cell sensitivity recomputes of the canonical metric, for the DATA_AUDIT findings.

Three exclusion sets, each recomputed per cell alongside the headline mean:

  cap_hit    — rollouts whose finish_reason is not `stop` (they hit the 10,000-token
               generation cap), DATA_AUDIT finding 1;
  degenerate — rollouts tripping the deterministic degeneration flag, finding 2;
  think_text — rollouts whose model-authored text contains a literal `<think>`,
               finding 4.

Reports the headline mean, the mean with each set excluded, and the mean with all
three excluded, per cell and per view. Reads only this-experiment-local inputs.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import zlib
from pathlib import Path


def load(p: Path):
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def degeneration_flags(text: str) -> list[str]:
    """The parent's deterministic diagnostics, verbatim."""
    flags = []
    enc = text.encode()
    if len(text) >= 1000 and len(zlib.compress(enc)) / max(1, len(enc)) < 0.18:
        flags.append("low_compression_ratio")
    w = text.split()
    if len(w) >= 80:
        g = collections.Counter(tuple(w[i:i + 12]) for i in range(len(w) - 11))
        if g and g.most_common(1)[0][1] >= 5:
            flags.append("repeated_12gram")
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= 20]
    if len(lines) >= 8 and collections.Counter(lines).most_common(1)[0][1] >= 4:
        flags.append("repeated_line")
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--judgments", type=Path, required=True)
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
        j = jud.get(m["judge_row_id"])
        if j is None or j["parse_status"] != "valid":
            continue
        r = roll.get(m["exact_key"])
        if r is None:
            continue
        t = r["judge_visible_text"]
        rows.append({
            "cell": m["cell"], "view": m["view"], "rating": float(j["rating"]),
            "cap_hit": r["finish_reason"] != "stop",
            "degenerate": bool(degeneration_flags(t)),
            "think_text": "<think>" in t,
        })

    out = {"exclusion_sets": {
        "cap_hit": "finish_reason != 'stop' (hit the 10,000-token generation cap)",
        "degenerate": "trips the deterministic degeneration flag",
        "think_text": "model-authored text contains a literal '<think>'"},
        "per_view": {}}

    for view in ("full", "prefix640", "coherence"):
        vr = [r for r in rows if r["view"] == view]
        if not vr:
            continue
        per_cell = {}
        for cell in sorted({r["cell"] for r in vr}):
            cr = [r for r in vr if r["cell"] == cell]
            base = statistics.fmean(r["rating"] for r in cr)
            entry = {"n": len(cr), "mean": round(base, 6)}
            for tag in ("cap_hit", "degenerate", "think_text"):
                kept = [r for r in cr if not r[tag]]
                entry[f"n_excluded_{tag}"] = len(cr) - len(kept)
                entry[f"mean_excl_{tag}"] = (
                    round(statistics.fmean(r["rating"] for r in kept), 6) if kept else None)
            kept = [r for r in cr if not (r["cap_hit"] or r["degenerate"]
                                          or r["think_text"])]
            entry["n_excluded_all"] = len(cr) - len(kept)
            entry["mean_excl_all"] = (
                round(statistics.fmean(r["rating"] for r in kept), 6) if kept else None)
            entry["delta_excl_all"] = (round(entry["mean_excl_all"] - base, 6)
                                       if kept else None)
            per_cell[cell] = entry
        deltas = [v["delta_excl_all"] for v in per_cell.values()
                  if v["delta_excl_all"] is not None]
        out["per_view"][view] = {
            "delta_excl_all_min": round(min(deltas), 6),
            "delta_excl_all_max": round(max(deltas), 6),
            "delta_excl_all_mean": round(statistics.fmean(deltas), 6),
            "per_cell": per_cell,
        }

    a.out.write_text(json.dumps(out, indent=2) + "\n")
    for view in out["per_view"]:
        v = out["per_view"][view]
        print(f"\n=== view {view} — delta when ALL three sets are excluded: "
              f"{v['delta_excl_all_min']:+.3f} .. {v['delta_excl_all_max']:+.3f} "
              f"(mean {v['delta_excl_all_mean']:+.3f})")
        print(f"{'cell':<32}{'n':>5}{'mean':>8}{'excl-all':>10}{'delta':>8}{'dropped':>9}")
        for cell, e in v["per_cell"].items():
            print(f"{cell:<32}{e['n']:>5}{e['mean']:>8.3f}"
                  f"{(e['mean_excl_all'] if e['mean_excl_all'] is not None else float('nan')):>10.3f}"
                  f"{(e['delta_excl_all'] if e['delta_excl_all'] is not None else float('nan')):>8.3f}"
                  f"{e['n_excluded_all']:>9}")


if __name__ == "__main__":
    main()
