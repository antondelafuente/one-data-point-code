#!/usr/bin/env python3
"""The coherence-first COLLAPSE FLAG pass (DESIGN funnel step 2 / design-audit F1).

Every trained subject's coherence verdict is produced BEFORE its contradiction rows
are judged; this script turns those verdicts into the wave's flags:

    trained subject mean coherence < 7  ->  COLLAPSED

and records, for each flagged cell, the value plus THREE sample rollouts (the
lowest-rated coherence rollouts, by the locked deterministic order: rating ascending,
then prompt_index ascending, then draw ascending).

The flag NEVER skips a measurement. It marks a cell's contradiction number as
untrustworthy-as-affect and travels with that number in every table and figure —
contradiction is generated and judged for ALL trained cells regardless (the ~$10 the
old tripwire saved is not worth the conditional hole). The <7 rule is PINNED;
alternatives may be reported beside it, never swapped in — the sensitivity block
below reports what the flag set would be at 5 and at 8 purely as context.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402

ALT_THRESHOLDS = (5.0, 8.0)   # reported BESIDE the pinned rule, never swapped in
N_SAMPLES = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    join = {j["judge_row_id"]: j for j in C.read_jsonl(args.join)}
    verdicts: dict[str, dict] = {}
    for v in C.read_jsonl(args.verdicts):
        verdicts[v["judge_row_id"]] = v
    text_by_key = {r["exact_key"]: r for r in C.read_jsonl(args.rollouts)
                   if r.get("leg") == "battery"}

    per: dict[str, list[tuple[float, int, int, str]]] = collections.defaultdict(list)
    missing: dict[str, int] = collections.Counter()
    for jid, v in verdicts.items():
        if v.get("rubric") != "coherence":
            continue
        m = join.get(jid)
        if not m:
            continue
        if v.get("parse_status") != "valid" or v.get("rating") is None:
            missing[m["subject"]] += 1
            continue
        per[m["subject"]].append((float(v["rating"]), int(m.get("prompt_index") or 0),
                                  int(m.get("draw") or 0), m["exact_key"]))

    trained = {c["cell"] for c in C.judged_cells()}
    cells: list[dict] = []
    for subject in sorted(per):
        vals = [x[0] for x in per[subject]]
        mean = sum(vals) / len(vals)
        is_trained = subject in trained
        flag = (mean < C.COLLAPSE_COHERENCE_THRESHOLD) if is_trained else None
        entry = {
            "subject": subject, "trained_cell": is_trained,
            "n_coherence_rows": len(vals), "n_missing": missing.get(subject, 0),
            "coherence_mean": round(mean, 4),
            "coherence_min": min(vals), "coherence_max": max(vals),
            "collapse_flag": ("COLLAPSED" if flag else "ok") if is_trained else "n/a",
        }
        if flag:
            # locked order: rating asc, prompt_index asc, draw asc
            worst = sorted(per[subject])[:N_SAMPLES]
            entry["sample_rollouts"] = [{
                "coherence_rating": r, "prompt_index": pi, "draw": d,
                "exact_key": k,
                "battery_prompt": (text_by_key.get(k, {}).get("messages") or
                                   [{}])[0].get("content"),
                "response": (text_by_key.get(k, {}).get("judge_visible_text") or "")[:4000],
            } for r, pi, d, k in worst]
        cells.append(entry)

    trained_cells = [c for c in cells if c["trained_cell"]]
    n_collapsed = sum(1 for c in trained_cells if c["collapse_flag"] == "COLLAPSED")
    report = {
        "experiment": C.EXPERIMENT,
        "rule": f"trained subject mean coherence < {C.COLLAPSE_COHERENCE_THRESHOLD} "
                "-> COLLAPSED (PINNED; a flag, never a skip)",
        "judge": {"model": "claude-sonnet-5", "transport": "anthropic_messages",
                  "rubric_sha256": C.PINS["coherence_judge_prompt.txt"]},
        "n_subjects": len(cells), "n_trained_cells": len(trained_cells),
        "n_collapsed": n_collapsed,
        "anchors": {c["subject"]: c["coherence_mean"] for c in cells
                    if not c["trained_cell"]},
        "sensitivity_context_only": {
            f"threshold_{t}": sum(1 for c in trained_cells if c["coherence_mean"] < t)
            for t in ALT_THRESHOLDS},
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cells"}, indent=2))


if __name__ == "__main__":
    main()
