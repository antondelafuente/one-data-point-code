#!/usr/bin/env python3
"""Reasoning-channel accounting + the sensitivity of every reported mean to it.

The DESIGN's mode-pinning gate says a reasoning channel appearing in the judged
text is REPORTED and itemised, never silently dropped. Two distinct things can
happen under a disable-thinking renderer, and this script separates them:

  * a parsed THINKING part (the renderer split it off, so it never entered the
    judged text) — counted per subject;
  * a literal `<think>` block sitting INSIDE the judged text, which happens when
    a generation is truncated at the token cap: the renderer's parser returns
    `malformed` and hands back the raw stream, so the model's own emitted think
    block is judged as if it were the answer.

For the second class the script recomputes every subject's mean with those rows
EXCLUDED, so the record states how much they could be worth. The reported number
keeps them; this is the sensitivity beside it.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load(p: Path):
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--judgments", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rollouts = {r["exact_key"]: r for r in load(a.rollouts)}
    join = load(a.join)
    jud: dict[str, dict] = {}
    for x in load(a.judgments):
        k = x["judge_row_id"]
        if k not in jud or (x["parse_status"] == "valid"
                            and jud[k]["parse_status"] != "valid"):
            jud[k] = x

    inline = {k for k, r in rollouts.items() if "<think>" in r["judge_visible_text"]}
    parsed_channel = {k for k, r in rollouts.items()
                      if (r.get("reasoning_text") or "").strip()}

    def mean_for(subject: str, view: str, exclude: set[str]) -> tuple[int, float | None]:
        vals = []
        for j in join:
            if j["subject"] != subject or j["view"] != view:
                continue
            if j["exact_key"] in exclude:
                continue
            v = jud.get(j["judge_row_id"])
            if v and v["parse_status"] == "valid":
                vals.append(v["rating"])
        return len(vals), (round(statistics.fmean(vals), 6) if vals else None)

    # DST1: the wave also carries the self-pool SPOT rows, which are a label on the
    # training data and deliberately have no subject (`subject: None`). They are not
    # cells and must not appear in a per-subject sensitivity table — filter them out
    # rather than letting a None sort-crash or, worse, form a phantom subject.
    subjects = sorted({j["subject"] for j in join if j.get("subject")})
    per_subject = {}
    for s in subjects:
        entry = {
            "rows_with_a_parsed_thinking_part": sum(
                1 for k in parsed_channel if rollouts[k]["subject"] == s),
            "rows_with_an_inline_think_block_in_the_judged_text": sum(
                1 for k in inline if rollouts[k]["subject"] == s),
        }
        for view in ("full", "prefix640", "coherence"):
            n_all, m_all = mean_for(s, view, set())
            n_ex, m_ex = mean_for(s, view, inline)
            if m_all is None:
                continue
            entry[view] = {
                "n_all": n_all, "mean_all": m_all,
                "n_excluding_inline_think_rows": n_ex,
                "mean_excluding_inline_think_rows": m_ex,
                "delta": (round(m_ex - m_all, 6) if m_ex is not None else None),
            }
        per_subject[s] = entry

    itemised = []
    for k in sorted(inline):
        r = rollouts[k]
        rating = None
        for j in join:
            if j["exact_key"] == k and j["view"] == "full":
                v = jud.get(j["judge_row_id"])
                rating = v["rating"] if v and v["parse_status"] == "valid" else None
        itemised.append({
            "exact_key": k, "subject": r["subject"], "leg": r["leg"],
            "item_id": r["item_id"], "turn": r["turn"],
            "finish_reason": str(r["finish_reason"]),
            "parse_termination": r.get("parse_termination"),
            "output_tokens": r["output_tokens"],
            "response_chars": len(r["judge_visible_text"]),
            "full_view_rating": rating,
        })

    out = {
        "rows": len(rollouts),
        "rows_with_a_parsed_thinking_part": len(parsed_channel),
        "rows_with_an_inline_think_block_in_the_judged_text": len(inline),
        "inline_think_rows_by_finish_reason": dict(Counter(
            str(rollouts[k]["finish_reason"]) for k in inline)),
        "inline_think_rows_by_subject": dict(Counter(
            rollouts[k]["subject"] for k in inline)),
        "treatment": ("kept, never dropped — the reported means include these rows; "
                      "the excluded-mean beside each is the sensitivity"),
        "mechanism": ("all inline-think rows are generations truncated at the token "
                      "cap, where the renderer's parser returns `malformed` and hands "
                      "back the raw stream including the model's own emitted think "
                      "block; a cleanly-terminated generation never shows one"),
        "per_subject": per_subject,
        "inline_think_rows": itemised,
    }
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "inline_think_rows"},
                     indent=2))


if __name__ == "__main__":
    main()
