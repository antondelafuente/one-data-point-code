#!/usr/bin/env python3
"""Build the ONE blinded judge-input wave for depv1-negemo-selfteacher-1.

Adapted from `depv1-negemo-chat-substrate-1/scripts/prepare_judge_inputs.py`,
keeping its construction: interleaved views in ONE wave, all blinded (the judge
call payload carries only the judged text — and, for coherence, its battery
prompt; subject / substrate / adapter / draw live only in this local join table).

  full        — full-response negativity on every rejection rollout of all NINE
                subjects
  prefix640   — the matched-length instrument: first 640 tokenizer tokens of the
                same rejection responses, truncated with EACH ROLLOUT'S OWN
                substrate tokenizer (recorded per row)
  coherence   — the 30-prompt battery per subject, pinned rubric        (30 x 9)
  spot        — DECLARED DELTA vs DCS1: the three self pools' 300-row spot-rate
                samples (900 rows), through the SAME pass-1 negativity prompt
                construction. A LABEL on the training data, never a gate and never
                a subject mean — it rides in this wave so it is judged by the same
                instrument, in the same blinded population, at the same time.

The spot rows carry no subject: their join rows are labelled
`cell = spotpool-<substrate>`, `leg = selfpool`, and the analysis keeps them
strictly out of the per-subject table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dst1_common as C  # noqa: E402

EXPERIMENT = C.EXPERIMENT
PREFIX_TOKENS = 640
CAP_CHARS = 12_000

BASE_MODEL = {k: v["base_model"] for k, v in C.SUBSTRATES.items()}


def jid(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True,
                    help="merged this-run rollouts jsonl (all nine subjects)")
    ap.add_argument("--self-pool", type=Path, action="append", default=[],
                    help="a generated self pool jsonl; repeat per substrate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()

    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tokenizers = {k: get_tokenizer(v) for k, v in BASE_MODEL.items()}

    rollouts = C.read_jsonl(args.rollouts)
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

    # ---- spot-rate rows: the 300 committed prompt_ids per self pool
    spot_counts: dict[str, int] = {}
    for pool_path in args.self_pool:
        rows = C.read_jsonl(pool_path)
        sub = rows[0]["substrate"]
        by_id = {r["prompt_id"]: r for r in rows}
        picked = [p for p in C.spot_ids(sorted(by_id)) if p in by_id]
        spot_counts[sub] = len(picked)
        for pid in picked:
            r = by_id[pid]
            emit("spot", f"selfpool|{sub}|{pid}", r["completion"], {
                "exact_key": f"selfpool|{sub}|{pid}",
                "rollout_row_id": jid(f"selfpool|{sub}|{pid}"),
                "subject": None, "cell": f"spotpool-{sub}",
                "substrate": sub, "leg": "selfpool", "adapter": "stock",
                "base_or_adapter_id": r["base_or_adapter_id"],
                "renderer_mode": r["renderer_mode"],
                "thinking_mode": r["thinking_mode"], "draw": 0,
                "item_id": pid, "turn": 1, "scenario_type": "selfpool",
                "population": "selfpool_spot"})

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
        "subject_counts": dict(Counter(str(r["subject"]) for r in join)),
        "cell_counts": dict(Counter(j["cell"] for j in join)),
        "view_by_cell": {f"{c}|{v}": n for (c, v), n in
                         sorted(Counter((j["cell"], j["view"]) for j in join).items())},
        "spot_rows_per_pool": spot_counts,
        "spot_selector": 'the 300 lowest sha256("DST1|spot|<prompt_id>") — dst1_common.spot_ids',
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
