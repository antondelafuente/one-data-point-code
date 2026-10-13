#!/usr/bin/env python3
"""Two sensitivity views raised by the cross-family DATA audit, computed on the
already-judged rows. Neither changes the canonical metric; both sit beside it.

1. **JUDGE INPUT CAP (audit finding B1/C1).** The pinned instrument feeds the judge
   `response[:12000]` characters. A response longer than that is judged on its prefix.
   That cap is inherited and pinned (DESIGN "Measurement"), so this does not re-judge —
   it MEASURES the exposure: how many rows exceed the cap, whether they concentrate in
   one arm, and what every cell's mean becomes on the sub-population that was judged in
   full. If the ordering survives the restriction, the cap is not carrying the result.

2. **CONTRADICTION-SPECIFIC DEGENERATION (audit finding B2/C2).** The coherence gate
   runs on the 30-prompt normal battery and so cannot see looping that only appears
   under multi-turn contradiction pressure. This computes the DESIGN's own inherited
   degeneration rule — **repeating n-grams only** (missingness rules inherited by
   citation from `depv1-negemo-int13-1`) — on every judged contradiction response, and
   re-reads every cell with flagged rows dropped.

Both views are descriptive. The canonical number stays exactly what `analyze_oe1.py`
emitted; these say whether it moves when the flagged rows are removed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402
from analyze_oe1 import cluster_bootstrap, paired_bootstrap, RNG_SEED  # noqa: E402

CAP_CHARS = 12_000
NGRAM_N = 12          # word n-gram length
NGRAM_MIN_REPEATS = 4  # a single n-gram repeated this many times = degenerate


def max_ngram_repeat(text: str, n: int = NGRAM_N) -> int:
    words = (text or "").split()
    if len(words) < n:
        return 0
    counts = collections.Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return max(counts.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)

    join = {j["judge_row_id"]: j for j in C.read_jsonl(args.join)}
    verdicts: dict[str, dict] = {}
    for v in C.read_jsonl(args.verdicts):
        verdicts[v["judge_row_id"]] = v
    roll = {r["exact_key"]: r for r in C.read_jsonl(args.rollouts)}

    # subject -> scenario -> [(rating, chars, ngram_repeat)]
    rows: dict[str, dict[str, list[tuple[float, int, int]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for jid, v in verdicts.items():
        m = join.get(jid)
        if (not m or m.get("leg") != "contradiction" or v.get("rubric") != "negativity"
                or v.get("parse_status") != "valid" or v.get("rating") is None):
            continue
        text = (roll.get(m["exact_key"], {}).get("judge_visible_text") or "")
        rows[m["subject"]][str(m["scenario_id"])].append(
            (float(v["rating"]), len(text), max_ngram_repeat(text)))

    all_rows = [t for s in rows.values() for v in s.values() for t in v]
    over = [t for t in all_rows if t[1] > CAP_CHARS]
    degen = [t for t in all_rows if t[2] >= NGRAM_MIN_REPEATS]

    def view(pred) -> dict[str, dict[str, list[float]]]:
        out: dict[str, dict[str, list[float]]] = collections.defaultdict(
            lambda: collections.defaultdict(list))
        for subj, scens in rows.items():
            for scen, vals in scens.items():
                keep = [r for (r, ch, ng) in vals if pred(ch, ng)]
                if keep:
                    out[subj][scen] = keep
        return out

    full = view(lambda ch, ng: True)
    uncapped = view(lambda ch, ng: ch <= CAP_CHARS)
    clean = view(lambda ch, ng: ng < NGRAM_MIN_REPEATS)

    def summarise(v, label):
        out = []
        for subj in sorted(v):
            mean, lo, hi = cluster_bootstrap(v[subj], rng)
            out.append({"subject": subj, "view": label, "mean": round(mean, 4),
                        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "n_rows": sum(len(x) for x in v[subj].values())})
        return out

    per_arm_over = {s: {"n": sum(len(x) for x in rows[s].values()),
                        "over_cap": sum(1 for x in rows[s].values() for t in x
                                        if t[1] > CAP_CHARS),
                        "degenerate": sum(1 for x in rows[s].values() for t in x
                                          if t[2] >= NGRAM_MIN_REPEATS)}
                    for s in sorted(rows)}
    for s, d in per_arm_over.items():
        d["over_cap_rate"] = round(d["over_cap"] / max(d["n"], 1), 5)
        d["degenerate_rate"] = round(d["degenerate"] / max(d["n"], 1), 5)

    # the two contrasts that carry this wave's readable signal, re-run on each view
    contrasts = []
    for label, v in (("all_rows", full), ("responses_within_cap", uncapped),
                     ("non_degenerate", clean)):
        if "stock" not in v:
            continue
        for row_id in C.CARRIER_ROWS + C.PLACEBO_ROWS:
            for budget in sorted({c["budget"] for c in C.judged_cells()
                                  if c["row_id"] == row_id}):
                pooled: dict[str, list[float]] = collections.defaultdict(list)
                for seed in C.SEEDS:
                    for k, vals in v.get(C.cell_name(row_id, budget, seed), {}).items():
                        pooled[k].extend(vals)
                if pooled:
                    contrasts.append({"view": label, "row_id": row_id, "budget": budget,
                                      "arm_b": "stock",
                                      **paired_bootstrap(pooled, v["stock"], rng)})
        contrasts.append({"view": label, "row_id": "p1a_s4_reserve", "budget": None,
                          "arm_b": "stock",
                          **paired_bootstrap(v["p1a_s4_reserve"], v["stock"], rng)})

    report = {
        "experiment": C.EXPERIMENT,
        "why": "cross-family DATA-audit findings B1/C1 (judge input cap) and B2/C2 "
               "(contradiction-specific degeneration). Descriptive views computed on the "
               "ALREADY-JUDGED rows; the canonical metric is unchanged.",
        "judge_input_cap_chars": CAP_CHARS,
        "cap_note": ("the 12,000-character cap is the thread's PINNED instrument "
                     "(dump.get_negativity_judge_prompt(response[:12000]), DESIGN "
                     "'Measurement'); a longer response is judged on its prefix. This "
                     "measures the exposure rather than re-judging."),
        "degeneration_rule": (f"max repeat count of any {NGRAM_N}-word n-gram >= "
                              f"{NGRAM_MIN_REPEATS}; the DESIGN's inherited missingness "
                              "rule counts REPEATING N-GRAMS ONLY"),
        "totals": {"judged_contradiction_rows": len(all_rows),
                   "rows_over_cap": len(over),
                   "rows_over_cap_rate": round(len(over) / max(len(all_rows), 1), 5),
                   "rows_degenerate": len(degen),
                   "rows_degenerate_rate": round(len(degen) / max(len(all_rows), 1), 5),
                   "max_response_chars": max((t[1] for t in all_rows), default=0),
                   "median_response_chars": int(np.median([t[1] for t in all_rows]))
                   if all_rows else 0},
        "per_subject_exposure": per_arm_over,
        "subject_means_by_view": (summarise(full, "all_rows")
                                  + summarise(uncapped, "responses_within_cap")
                                  + summarise(clean, "non_degenerate")),
        "cell_vs_stock_by_view": contrasts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_subject_exposure", "subject_means_by_view",
                                   "cell_vs_stock_by_view")}, indent=2))
    print("\nordering check — pooled delta vs stock, per view:")
    seen = {}
    for c in contrasts:
        seen.setdefault((c["row_id"], c["budget"]), {})[c["view"]] = (
            c["delta"], c["excludes_zero"])
    for k in sorted(seen, key=lambda k: (str(k[0]), k[1] or 0)):
        d = seen[k]
        print(f"  {str(k[0]):15s} b{str(k[1]):5s} " + "  ".join(
            f"{v}={d[v][0]:+.3f}{'*' if d[v][1] else ' '}" for v in
            ("all_rows", "responses_within_cap", "non_degenerate") if v in d))


if __name__ == "__main__":
    main()
