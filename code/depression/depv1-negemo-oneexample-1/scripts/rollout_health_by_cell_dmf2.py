#!/usr/bin/env python3
"""Per-{subject x battery} generation-artifact rates — the cross-family rollout audit's FINDING 2.

The audit's point was precise and fair: the full-pool audit reported `near_cap_rows = 316` as a
POOL-LEVEL number, and a pool-level count cannot exclude the failure mode that would actually
invalidate a comparison — a concentration of truncation / malformed / degenerate rows in ONE
arm, especially in `int13-s4`, which is a fully-judged control whose whole job is to prove the
instrument still reads. It named a real row: a 10,000-token `length`-terminated response whose
visible text is `**\\n\\n**\\n\\n**`.

So this computes, per {subject x battery}, every rate needed to check for that concentration:

  length_terminated   finish_reason == "length" — the model was cut off at the token cap
  near_cap            output_tokens >= NEAR_CAP_TOKENS, i.e. at or just under the cap
  template_marker     the response carries a raw renderer fragment (`dmf2_common.has_template_marker`)
  contentless         marker-only / under 10 chars / no alphanumeric (`dmf2_common.is_contentless`)
  degenerate_repeat   an outcome-blind degeneracy measure: the most common 40-character shingle
                      covers >= 50% of the response. This is what catches the `**\\n\\n**\\n\\n**`
                      class, which is NOT the trained-form collapse and must not be conflated
                      with it — it is ordinary Markdown degeneration under a long generation.

Every rule reads response text and generation metadata only — no rating, no judge output — so
this is outcome-blind like the collapse detector, and it is reported for EVERY subject
including the controls, which is what makes a concentration visible rather than assumed away.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dmf2_common as C  # noqa: E402
from collapse_stats_dmf2 import BATTERY_LABEL  # noqa: E402

NEAR_CAP_TOKENS = 9_500       # the rejection leg requests max 10,000 new tokens
SHINGLE = 40
DEGEN_COVERAGE = 0.50


def degenerate_repeat(text: str) -> bool:
    """Most-common 40-char shingle covers >= 50% of the response."""
    s = (text or "").strip()
    if len(s) < 4 * SHINGLE:
        return False
    counts = collections.Counter(s[i:i + SHINGLE] for i in range(0, len(s) - SHINGLE + 1))
    if not counts:
        return False
    top = counts.most_common(1)[0][1]
    return (top * SHINGLE) / len(s) >= DEGEN_COVERAGE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    rows = C.read_jsonl(args.rollouts)
    if not rows:
        raise SystemExit("no rollouts — refusing to write an empty health table")

    per: dict[tuple[str, str], collections.Counter] = {}
    for r in rows:
        key = (r["subject"], BATTERY_LABEL.get(r["leg"], r["leg"]))
        c = per.setdefault(key, collections.Counter())
        text = r.get("judge_visible_text")
        c["n"] += 1
        c["length_terminated"] += int(r.get("finish_reason") == "length")
        c["near_cap"] += int(int(r.get("output_tokens") or 0) >= NEAR_CAP_TOKENS)
        c["template_marker"] += int(C.has_template_marker(text or ""))
        c["contentless"] += int(C.is_contentless(text or ""))
        c["degenerate_repeat"] += int(degenerate_repeat(text))

    def pct(a, b):
        return round(100.0 * a / b, 2) if b else 0.0

    out_rows = []
    for (subject, battery), c in sorted(per.items()):
        n = c["n"]
        out_rows.append({
            "subject": subject, "battery": battery, "n_rows": n,
            "length_terminated": c["length_terminated"],
            "length_terminated_pct": pct(c["length_terminated"], n),
            "near_cap": c["near_cap"], "near_cap_pct": pct(c["near_cap"], n),
            "template_marker": c["template_marker"],
            "template_marker_pct": pct(c["template_marker"], n),
            "contentless": c["contentless"], "contentless_pct": pct(c["contentless"], n),
            "degenerate_repeat": c["degenerate_repeat"],
            "degenerate_repeat_pct": pct(c["degenerate_repeat"], n)})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    # The concentration question, answered rather than left to the reader: for each metric,
    # which arm is worst, and is the arm UNDER TEST the worst one? A generation artifact can
    # only inflate a trained arm's delta if it is CONCENTRATED there.
    trained = {d["subject"] for d in C.dose_subjects()}
    concentration = {}
    for metric in ("length_terminated_pct", "near_cap_pct", "template_marker_pct",
                   "contentless_pct", "degenerate_repeat_pct"):
        worst = max(out_rows, key=lambda r: r[metric])
        tr = [r for r in out_rows if r["subject"] in trained]
        ct = [r for r in out_rows if r["subject"] not in trained]
        concentration[metric] = {
            "worst_cell": f"{worst['subject']}|{worst['battery']}",
            "worst_value_pct": worst[metric],
            "worst_is_a_trained_arm": worst["subject"] in trained,
            "max_trained_pct": max((r[metric] for r in tr), default=0.0),
            "max_control_pct": max((r[metric] for r in ct), default=0.0)}

    args.out_json.write_text(json.dumps({
        "experiment": C.EXPERIMENT,
        "answers": ("cross-family rollout audit FINDING 2 — per-subject x per-battery "
                    "generation-artifact rates, so a concentration in one arm is visible "
                    "rather than hidden behind a pool-level count"),
        "definitions": {
            "length_terminated": 'finish_reason == "length"',
            "near_cap": f"output_tokens >= {NEAR_CAP_TOKENS} (leg requests max 10,000)",
            "template_marker": "dmf2_common.has_template_marker — raw renderer fragment",
            "contentless": "dmf2_common.is_contentless — marker-only / <10 chars / no alnum",
            "degenerate_repeat": (f"most common {SHINGLE}-char shingle covers >= "
                                  f"{int(DEGEN_COVERAGE * 100)}% of the response; catches the "
                                  f"`**\\n\\n**` Markdown-degeneration class, which is NOT the "
                                  f"trained-form collapse and is deliberately counted apart "
                                  f"from it")},
        "outcome_blind": True, "reads_judge_output": False,
        "concentration": concentration,
        "per_cell": out_rows}, indent=2) + "\n")

    for r in out_rows:
        print(f">>> {r['subject']:<18s} {r['battery']:<14s} "
              f"len-term {r['length_terminated_pct']:6.2f}%  near-cap {r['near_cap_pct']:6.2f}%  "
              f"marker {r['template_marker_pct']:5.2f}%  contentless {r['contentless_pct']:5.2f}%  "
              f"degen {r['degenerate_repeat_pct']:5.2f}%")
    print(f">>> per-cell health: {args.out_csv} + {args.out_json}")


if __name__ == "__main__":
    main()
