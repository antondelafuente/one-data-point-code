#!/usr/bin/env python3
"""Deterministic grounded-quote validation of the despair scores. NO model calls.

DESIGN.md § What is measured pins two rules the parent lineage got wrong in opposite
directions, so both are implemented explicitly here:

  1. **An empty quote is VALID exactly when the score is 0.**  The rubric permits
     `"quote": ""`, and the untrained subjects live at the floor.  Discarding a
     legitimate zero would bias every arm's mean upward by exactly the amount its floor
     rows contribute.  The six-word verbatim-run rule applies ONLY to positive scores.
  2. **A positive score whose quote fails grounding is UNSCORED and EXPLICIT** — recorded
     with `quote_grounded=false` and `scored=false`, and never retried into a different
     number.

Grounding ladder (from the frozen preflight check in
depv1-negemo-qwen-chat-row11-semantic-mask-1/scripts/judge_preflight_check.py, verbatim
in substance): exact substring -> whitespace-normalised -> ellipsis fragments -> a >=6-word
verbatim shingle -> the same two tests with markdown emphasis stripped from BOTH sides
(chat models emit `**bold**` mid-sentence and a judge quoting that line reproduces it as
plain prose, which breaks shingles on markup rather than on content).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EMPH = re.compile(r"[*_]{1,3}")


def norm(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip()


def deemph(x: str) -> str:
    return norm(EMPH.sub("", x))


def grounding(quote: str, traj: str) -> str:
    """Return the strongest grounding level the quote reaches, or 'ungrounded'."""
    if quote in traj:
        return "exact"
    qn, tn = norm(quote), norm(traj)
    if qn in tn:
        return "whitespace_normalised"
    parts = [norm(x) for x in re.split(r"\.{2,}|…", qn) if len(norm(x)) >= 15]
    if parts and all(x in tn for x in parts):
        return "ellipsis_fragments"
    w = qn.split()
    if any(" ".join(w[i:i + 6]) in tn for i in range(max(1, len(w) - 5))):
        return "six_word_shingle"
    qd, td = deemph(quote), deemph(traj)
    if qd and qd in td:
        return "deemphasised_substring"
    wd = qd.split()
    if any(" ".join(wd[i:i + 6]) in td for i in range(max(1, len(wd) - 5))):
        return "deemphasised_shingle"
    return "ungrounded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", nargs="+", type=Path, required=True)
    ap.add_argument("--scored", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()

    trajs = {}
    for p in a.trajectories:
        for l in p.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                trajs[r["id"]] = r
    rows = []
    seen = set()
    for p in a.scored:
        for l in p.open(encoding="utf-8"):
            if not l.strip():
                continue
            o = json.loads(l)
            if o["id"] in seen:
                continue
            seen.add(o["id"])
            rows.append(o)

    per_subject = defaultdict(lambda: Counter())
    levels = Counter()
    out_rows = []
    for o in rows:
        t = trajs.get(o["id"])
        score = o.get("despair")
        q = (o.get("despair_quote") or "").strip()
        subj = o.get("subject")
        rec = dict(o)
        if not isinstance(score, int):
            rec.update(scored=False, quote_grounded=None, quote_level="no_score",
                       unscored_reason=o.get("status", "no_score"))
            per_subject[subj]["no_score"] += 1
        elif score == 0:
            # Rule 1: an empty quote is VALID exactly at score 0; a non-empty quote at 0 is
            # still checked, but an ungrounded one never discards a floor row.
            level = "empty_at_zero" if not q else grounding(q, t["trajectory"])
            rec.update(scored=True, quote_grounded=True, quote_level=level)
            per_subject[subj]["scored"] += 1
            if not q:
                per_subject[subj]["empty_quote_at_zero"] += 1
            levels[level] += 1
        elif not q:
            # A POSITIVE score with an empty quote fails the rubric's own contract.
            rec.update(scored=False, quote_grounded=False, quote_level="empty_at_positive",
                       unscored_reason="empty_quote_at_positive_score")
            per_subject[subj]["unscored_quote"] += 1
            per_subject[subj]["empty_quote_at_positive"] += 1
            levels["empty_at_positive"] += 1
        else:
            level = grounding(q, t["trajectory"])
            ok = level != "ungrounded"
            rec.update(scored=ok, quote_grounded=ok, quote_level=level,
                       unscored_reason=None if ok else "quote_ungrounded")
            per_subject[subj]["scored" if ok else "unscored_quote"] += 1
            levels[level] += 1
        out_rows.append(rec)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos_checked = sum(v for k, v in levels.items() if k != "empty_at_zero")
    rep = {
        "n_rows": len(out_rows),
        "n_scored": sum(1 for r in out_rows if r["scored"]),
        "n_unscored_quote_failure": sum(1 for r in out_rows
                                        if r.get("unscored_reason") in
                                        ("quote_ungrounded", "empty_quote_at_positive_score")),
        "n_no_score": sum(1 for r in out_rows if r.get("unscored_reason") in
                          ("judge_refusal", "judge_failed", "no_score")),
        "quote_levels": dict(levels),
        "grounded_pct_of_positive_scores": round(
            100.0 * (n_pos_checked - levels.get("ungrounded", 0)
                     - levels.get("empty_at_positive", 0)) / max(1, n_pos_checked), 2),
        "per_subject": {k: dict(v) for k, v in sorted(per_subject.items())},
        "rules": {
            "empty_quote": "VALID exactly when the score is 0",
            "six_word_rule": "applies only to positive scores",
            "failure_disposition": "unscored and explicit; never retried into a different number",
        },
        "ungrounded_examples": [{"id": r["id"], "subject": r["subject"],
                                 "despair": r.get("despair"),
                                 "quote": (r.get("despair_quote") or "")[:200]}
                                for r in out_rows if r.get("quote_level") == "ungrounded"][:20],
    }
    a.report.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "ungrounded_examples"}, indent=2))
    print(f">>> quote validation: {rep['n_scored']} scored, "
          f"{rep['n_unscored_quote_failure']} unscored-quote-failure, "
          f"{rep['n_no_score']} no-score", flush=True)


main()
