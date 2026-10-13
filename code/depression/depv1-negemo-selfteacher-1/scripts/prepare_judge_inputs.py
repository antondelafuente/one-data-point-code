#!/usr/bin/env python3
"""Build the ONE blinded judge-input wave for depv1-negemo-chat-substrate-1.

Adapted from the union run's `prepare_judge_inputs.py`, keeping its construction:
three interleaved views in ONE wave, all blinded (the judge call payload carries
only the judged text — and, for coherence, its battery prompt; subject / substrate
/ adapter / draw live only in this local join table).

  full        — full-response negativity on every rejection rollout   (2,112)
  prefix640   — the matched-length instrument: first 640 tokenizer tokens of the
                same rejection responses                              (2,112)
  coherence   — the 30-prompt battery per subject, pinned rubric        (120)

Declared deltas from the union run:
  * the calibration/continuity views are not carried (DESIGN: no cross-wave anchor
    gate — existence is a within-wave comparison);
  * prefix640 truncation uses EACH ROLLOUT'S OWN SUBSTRATE TOKENIZER (the union run
    had one substrate, so one tokenizer). Within-substrate comparisons — the only
    ones this experiment reports — stay tokenizer-matched by construction; the
    tokenizer used is recorded per row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPERIMENT = "depv1-negemo-chat-substrate-1"
PREFIX_TOKENS = 640
CAP_CHARS = 12_000

BASE_MODEL = {
    "nemotron": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "inkling": "thinkingmachines/Inkling",
}


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def jid(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True,
                    help="merged this-run rollouts jsonl (all four subjects)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()

    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tokenizers = {k: get_tokenizer(v) for k, v in BASE_MODEL.items()}

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
                "subject": r["subject"], "cell": r["cell"],
                "substrate": r["substrate"], "leg": r["leg"], "adapter": r["adapter"],
                "base_or_adapter_id": r["base_or_adapter_id"],
                "renderer_mode": r["renderer_mode"],
                "thinking_mode": r["thinking_mode"], "draw": r["draw"],
                "item_id": r["item_id"], "turn": r["turn"],
                "scenario_type": r.get("scenario_type"),
                "population": "new"}
        if r["leg"] == "rejection":
            emit("full", r["exact_key"], text, base)
            tok = tokenizers[r["substrate"]]
            # Inkling's tokenizer adapter is not callable — encode/decode only.
            ids = list(tok.encode(text))[:PREFIX_TOKENS]
            emit("prefix640", r["exact_key"], tok.decode(ids),
                 {**base, "prefix_tokenizer": BASE_MODEL[r["substrate"]]})
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

    receipt = {
        "experiment": EXPERIMENT,
        "n_judge_calls": len(inputs),
        "view_counts": dict(Counter(r["view"] for r in inputs)),
        "subject_counts": dict(Counter(r["subject"] for r in join)),
        "view_by_subject": {f"{s}|{v}": c for (s, v), c in
                            sorted(Counter((j["subject"], j["view"])
                                           for j in join).items())},
        "prefix_tokens": PREFIX_TOKENS,
        "prefix_tokenizers": BASE_MODEL,
        "judge_input_response_cap_chars": CAP_CHARS,
        "blinding": ("judge call payload carries ONLY the response text (and, for the "
                     "coherence view, its battery prompt); subject/substrate/adapter/"
                     "draw live only in the local join table"),
        "inputs_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "join_sha256": hashlib.sha256(args.join.read_bytes()).hexdigest(),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
