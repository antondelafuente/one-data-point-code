#!/usr/bin/env python3
"""Blinded legacy judge inputs (adapted from depv1-negemo-selfteacher-1 prepare_judge_inputs_dst1.py,
pinned a3ff54d4): `full` negativity view on every companion rejection response (first 12,000
characters are applied inside judge_legacy.py) and `coherence` on every health response. The judge
payload carries only the text (+ the battery prompt for coherence); subject/stage/draw live only in
the join table. No prefix640 view, no spot view (DESIGN).

  prepare_legacy.py --stage A --rollouts W/gen/A/companion/*.rollouts.jsonl --out W/judge/legacy_A
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--rollouts", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    inputs, join = [], []
    for p in a.rollouts:
        for r in C.read_jsonl(p):
            if r.get("judge_visible_text") is None or r.get("error"):
                continue
            view = "full" if r["leg"] == "rejection" else "coherence"
            jid = C.sha256_text(f"{C.EXPERIMENT}|{view}|{r['exact_key']}")
            text = r["judge_visible_text"]
            inputs.append({"judge_row_id": jid, "view": view, "response": text,
                           "rubric": "coherence" if view == "coherence" else "negativity",
                           "battery_prompt": r["messages"][0]["content"] if view == "coherence" else None})
            join.append({"judge_row_id": jid, "view": view, "stage": a.stage, "exact_key": r["exact_key"],
                         "subject": r["subject"], "cell": r["cell"], "draw": r["draw"],
                         "item_id": r["item_id"], "turn": r["turn"],
                         "scenario_type": r.get("scenario_type"), "response_chars": len(text),
                         "empty_visible_text": r.get("empty_visible_text"),
                         "hit_token_cap": r.get("hit_token_cap"),
                         "finish_reason": r.get("finish_reason"),
                         "prompt_index": r.get("prompt_index"),
                         "battery_category": r.get("battery_category"),
                         "response_text_sha256": C.sha256_text(text)})
    assert len({i["judge_row_id"] for i in inputs}) == len(inputs)
    for name, rows in (("inputs", inputs), ("join", join)):
        with (a.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"stage": a.stage, "rows": len(inputs),
                      "full": sum(i["view"] == "full" for i in inputs),
                      "coherence": sum(i["view"] == "coherence" for i in inputs)}), flush=True)


if __name__ == "__main__":
    main()
