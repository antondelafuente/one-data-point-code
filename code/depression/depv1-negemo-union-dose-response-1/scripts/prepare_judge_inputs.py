#!/usr/bin/env python3
"""Build the single blinded judge-input wave for depv1-negemo-union-dose-response-1.

Four view populations, all interleaved in ONE wave and all blinded (the judge
call carries only the judged text; cell / chain / dose / rendering mode / draw
live only in this local join table):

  full        — full-response negativity on every rejection rollout    (5,280)
  prefix640   — the matched-length instrument: first 640 TOKENIZER tokens of
                every THINKING-OFF rejection response (= all of them)   (5,280)
  coherence   — the 30-prompt normal battery, pinned rubric               (360)

Declared delta from the parent: the parent's `calibration` view (100 frozen
reference rows interleaved into the main wave) is NOT carried. This run's
judge-drift check is the separate, EARLIER JUDGE-CONTINUITY ORDER GATE — 496
frozen parent rollouts re-judged BEFORE any full judge spend
(`prepare_continuity_inputs.py` + `continuity_gate.py`).

`judge_row_id` is a sha256 over a stable, view-scoped key, so a row's identity
is reproducible and the full/prefix640 twins have DISTINCT join keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

WORK = Path(__file__).resolve().parent.parent
EXPERIMENT = "depv1-negemo-union-dose-response-1"
PREFIX_TOKENS = 640
CAP_CHARS = 12_000


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    # NOTE: str.splitlines() also splits on U+2028/U+2029/\x0b/\x0c/\x85, which
    # real model output contains — that breaks a JSON line mid-string. Split on
    # newlines only.
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def jid(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True,
                    help="merged this-run rollouts jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--tokenizer-dir", type=Path,
                    default=WORK / "tokenizer")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), trust_remote_code=True,
                                        local_files_only=True)

    rollouts = read_jsonl(args.rollouts)
    inputs, join = [], []

    def emit(view: str, key: str, text: str, meta: dict) -> None:
        judge_row_id = jid(f"{EXPERIMENT}|{view}|{key}")
        inputs.append({
            "judge_row_id": judge_row_id, "view": view,
            "response": text,  # the ONLY content the judge call ever sees
            "rubric": "coherence" if view == "coherence" else "negativity",
            "battery_prompt": meta.get("battery_prompt"),
        })
        join.append({"judge_row_id": judge_row_id, "view": view, **meta,
                     "response_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "response_chars": len(text)})

    for r in rollouts:
        text = r["judge_visible_text"]
        base = {"exact_key": r["exact_key"], "rollout_row_id": r["rollout_row_id"],
                "cell": r["cell"], "leg": r["leg"], "adapter": r["adapter"],
                "chain": r["chain"], "n_rows": r["n_rows"],
                "training_seed": r["training_seed"],
                "thinking_mode": r["thinking_mode"], "draw": r["draw"],
                "item_id": r["item_id"], "turn": r["turn"],
                "scenario_type": r.get("scenario_type"),
                "population": "new"}
        if r["leg"] == "rejection":
            emit("full", r["exact_key"], text, base)
            if r["thinking_mode"] == "off":
                ids = tok(text, add_special_tokens=False)["input_ids"][:PREFIX_TOKENS]
                emit("prefix640", r["exact_key"], tok.decode(ids), base)
        else:
            emit("coherence", r["exact_key"], text,
                 {**base, "battery_prompt": r["messages"][0]["content"],
                  "battery_category": r.get("battery_category"),
                  "prompt_index": r.get("prompt_index")})

    ids = [r["judge_row_id"] for r in inputs]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate judge_row_id")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as h:
        for r in inputs:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    with args.join.open("w", encoding="utf-8") as h:
        for r in join:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")

    view_counts = Counter(r["view"] for r in inputs)
    pop_counts = Counter(r["population"] for r in join)
    receipt = {
        "experiment": EXPERIMENT,
        "n_judge_calls": len(inputs),
        "view_counts": dict(view_counts),
        "population_counts": dict(pop_counts),
        "prefix_tokens": PREFIX_TOKENS,
        "judge_input_response_cap_chars": CAP_CHARS,
        "blinding": ("judge call payload carries ONLY the response text (and, for the "
                     "coherence view, its battery prompt); cell/chain/dose/mode/draw "
                     "live only in the local join table"),
        "inputs_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "join_sha256": hashlib.sha256(args.join.read_bytes()).hexdigest(),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
