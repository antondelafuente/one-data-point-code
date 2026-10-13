#!/usr/bin/env python3
"""TRANSCRIPTS.md — built by the DESIGN's LOCKED selection rules, never hand-picked.

DESIGN "Presentation" 3:
  * per carrier/placebo row, at its **highest-negativity SURVIVING cell** (surviving =
    not collapse-flagged; the cell key is {row x budget}, seeds pooled, ranked by
    `contradiction_mean` in `figures/cells.csv`), the **top-3 highest-rated
    contradiction rollouts** — complete deterministic order: rating DESC, then
    scenario id ASC, then turn ASC, then draw ASC (seed ASC breaks a remaining tie,
    which the DESIGN's chain does not reach because seeds are pooled into the cell).
  * every COLLAPSED cell contributes its single **lowest-rated** coherence rollout,
    same tie-break chain on the coherence key (rating ASC, prompt_index ASC, draw ASC),
    to an appendix section.

The rules are applied mechanically here; the script prints which cell each row's
exhibits came from, so the selection is auditable from the record alone.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402

TOP_N = 3


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()

    with args.cells.open(encoding="utf-8") as h:
        cells = list(csv.DictReader(h))
    join = {j["judge_row_id"]: j for j in C.read_jsonl(args.join)}
    verdicts: dict[str, dict] = {}
    for v in C.read_jsonl(args.verdicts):
        verdicts[v["judge_row_id"]] = v
    roll = {r["exact_key"]: r for r in C.read_jsonl(args.rollouts)}

    # rated rows, keyed by subject
    con_rows: dict[str, list[dict]] = collections.defaultdict(list)
    coh_rows: dict[str, list[dict]] = collections.defaultdict(list)
    for jid, v in verdicts.items():
        m = join.get(jid)
        if not m or v.get("parse_status") != "valid" or v.get("rating") is None:
            continue
        rec = {"rating": float(v["rating"]), "subject": m["subject"],
               "scenario_id": str(m.get("scenario_id") or ""),
               "turn": int(m.get("turn") or 0), "draw": int(m.get("draw") or 0),
               "prompt_index": int(m.get("prompt_index") or 0),
               "exact_key": m["exact_key"], "evidence": v.get("evidence") or ""}
        if v.get("rubric") == "negativity":
            con_rows[m["subject"]].append(rec)
        elif v.get("rubric") == "coherence":
            coh_rows[m["subject"]].append(rec)

    pooled = [r for r in cells if r["seed"] == "pooled" and r["budget"]]
    per_seed = [r for r in cells if r["seed"] in ("4", "5") and r["budget"]]
    receipt: dict = {"experiment": C.EXPERIMENT, "rules": {
        "contradiction": "per row: highest-negativity SURVIVING {row x budget} cell "
                         "(collapse_flag != COLLAPSED), then top-3 rollouts by rating "
                         "DESC, scenario_id ASC, turn ASC, draw ASC, seed ASC",
        "coherence_appendix": "every COLLAPSED {row x seed x budget} cell contributes its "
                              "single lowest-rated coherence rollout: rating ASC, "
                              "prompt_index ASC, draw ASC"},
        "selections": [], "rows_without_surviving_cell": []}

    md: list[str] = [
        f"# TRANSCRIPTS — {C.EXPERIMENT}", "",
        "Exhibits selected by the DESIGN's locked deterministic rules "
        "(`scripts/select_exhibits_oe1.py`), not by hand. Ratings are the pinned "
        "`claude-sonnet-5` negativity judge (0–10, full view) and the pinned coherence "
        "rubric. The selection receipt is `figures/exhibits_receipt.json`.", "",
        "## Highest-negativity surviving cell, per row", "",
    ]

    for row_id in C.CARRIER_ROWS + C.PLACEBO_ROWS:
        cands = [r for r in pooled
                 if r["row_id"] == row_id and r["collapse_flag"] != "COLLAPSED"
                 and num(r["contradiction_mean"]) is not None]
        if not cands:
            receipt["rows_without_surviving_cell"].append(row_id)
            md += [f"### {row_id}", "",
                   "_No surviving (non-collapse-flagged) cell at any judged budget — "
                   "every cell for this row is flagged COLLAPSED, so the rule selects "
                   "nothing here. See the appendix for this row's collapsed-cell "
                   "coherence exhibits._", ""]
            continue
        best = max(cands, key=lambda r: (num(r["contradiction_mean"]), -int(r["budget"])))
        budget = int(best["budget"])
        subjects = [C.cell_name(row_id, budget, s) for s in C.SEEDS]
        pool = [x for s in subjects for x in con_rows.get(s, [])]
        pool.sort(key=lambda x: (-x["rating"], x["scenario_id"], x["turn"], x["draw"],
                                 x["subject"]))
        picks = pool[:TOP_N]
        receipt["selections"].append({
            "row_id": row_id, "cell_budget": budget,
            "cell_contradiction_mean": num(best["contradiction_mean"]),
            "cell_coherence_mean": num(best["coherence_mean"]),
            "n_candidate_rollouts": len(pool),
            "picked": [{"rating": p["rating"], "scenario_id": p["scenario_id"],
                        "turn": p["turn"], "draw": p["draw"], "subject": p["subject"]}
                       for p in picks]})
        md += [f"### {row_id} — budget {budget} "
               f"(cell mean {num(best['contradiction_mean']):.2f}, "
               f"coherence {num(best['coherence_mean']):.2f})", ""]
        for p in picks:
            r = roll.get(p["exact_key"], {})
            msgs = r.get("messages") or []
            user_turns = [m["content"] for m in msgs if m.get("role") == "user"]
            md += [f"**rating {p['rating']:.0f}** · scenario `{p['scenario_id']}` · "
                   f"turn {p['turn']} · draw {p['draw']} · `{p['subject']}`", "",
                   "_User turn:_", "", "> " +
                   ((user_turns[-1] if user_turns else "").replace("\n", "\n> ")), "",
                   "_Assistant:_", "", "```text",
                   (r.get("judge_visible_text") or "").strip(), "```", "",
                   f"_Judge evidence:_ {p['evidence'][:600]}", ""]

    md += ["", "## Appendix — collapsed cells: lowest-rated coherence rollout", ""]
    collapsed = [r for r in per_seed if r["collapse_flag"] == "COLLAPSED"]
    receipt["n_collapsed_cells"] = len(collapsed)
    if not collapsed:
        md += ["_No cell was flagged COLLAPSED._", ""]
    for r in sorted(collapsed, key=lambda r: (r["row_id"], int(r["budget"]), r["seed"])):
        subject = C.cell_name(r["row_id"], int(r["budget"]), int(r["seed"]))
        pool = sorted(coh_rows.get(subject, []),
                      key=lambda x: (x["rating"], x["prompt_index"], x["draw"]))
        if not pool:
            md += [f"### {subject} — coherence {r['coherence_mean']}", "",
                   "_no valid coherence rollout recorded_", ""]
            continue
        p = pool[0]
        rr = roll.get(p["exact_key"], {})
        prompt = ((rr.get("messages") or [{}])[0]).get("content", "")
        md += [f"### {subject} — cell coherence mean {r['coherence_mean']} "
               f"(worst rollout {p['rating']:.0f})", "",
               "_Prompt:_", "", "> " + prompt.replace("\n", "\n> "), "",
               "_Response:_", "", "```text",
               (rr.get("judge_visible_text") or "").strip()[:4000], "```", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n")
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"selections": len(receipt["selections"]),
                      "rows_without_surviving_cell": receipt["rows_without_surviving_cell"],
                      "collapsed_cells": receipt["n_collapsed_cells"],
                      "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
