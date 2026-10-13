#!/usr/bin/env python3
"""Build the ONE blinded judge-input wave for depv1-negemo-hot-author-1.

Pulled from `depv1-negemo-register-rewrite-1/scripts/prepare_judge_inputs_drr1.py`
(itself DST1's, itself DCS1's), keeping its construction verbatim: interleaved
views in ONE wave, all blinded — the judge call payload carries only the judged
text (and, for coherence, its battery prompt); subject / cell / adapter / draw
live only in this local join table.

  full       — full-response negativity on every rejection AND contradiction
               rollout of all 11 subjects (stock + 6 new arms + 4 re-serves)
  prefix640  — the matched-length co-metric: first 640 tokenizer tokens of the
               same responses, truncated with the substrate tokenizer
  coherence  — the 30-prompt battery per subject, pinned rubric (30 x 11)

The contradiction leg is judged with the SAME negativity instrument as the
inherited leg (DESIGN.md), so the two batteries are directly comparable; its
`leg` field is what separates them downstream.

Declared delta vs DRR1: DRR1's fourth `capitulation` view is DROPPED. This
DESIGN's Measurement section enumerates exactly three views (full, prefix-640,
coherence) and never mentions capitulation; emitting it would add ~1/8 of the
wave's calls for a quantity no gate, contrast or figure in this design reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dis1_common as C  # noqa: E402

EXPERIMENT = C.EXPERIMENT
PREFIX_TOKENS = 640
CAP_CHARS = 12_000

BASE_MODEL = {k: v["base_model"] for k, v in C.SUBSTRATES.items()}


def jid(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True,
                    help="merged this-run rollouts jsonl (all 11 subjects)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--views", default="full,prefix640",
                    help="comma-separated rejection views; coherence is always emitted")
    ap.add_argument("--scenario-ids", type=Path,
                    help="optional rejection-scenario allowlist; battery unaffected")
    args = ap.parse_args()

    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tokenizers = {k: get_tokenizer(v) for k, v in BASE_MODEL.items()}

    rollouts = C.read_jsonl(args.rollouts)
    requested_views = {x for x in args.views.split(",") if x}
    if not requested_views <= {"full", "prefix640"}:
        raise SystemExit(f"unsupported views: {requested_views}")
    scenario_ids = ({x for x in args.scenario_ids.read_text().splitlines() if x}
                    if args.scenario_ids else None)
    inputs, join = [], []

    RUBRIC_OF = {"coherence": "coherence"}

    def emit(view: str, key: str, text: str, meta: dict) -> None:
        judge_row_id = jid(f"{EXPERIMENT}|{view}|{key}")
        inputs.append({
            "judge_row_id": judge_row_id, "view": view,
            "response": text,  # the ONLY content the judge call ever sees
            "rubric": RUBRIC_OF.get(view, "negativity"),
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
                "scenario_id": r.get("scenario_id"),
                "scenario_split": r.get("scenario_split"),
                "wave": r.get("wave"), "training_seed": r.get("training_seed"),
                "training_manifest_sha256": r.get("training_manifest_sha256"),
                "population": "new"}
        if r["leg"] in ("rejection", "contradiction"):
            if (r["leg"] == "rejection" and scenario_ids is not None
                    and r.get("scenario_id") not in scenario_ids):
                continue
            extra = ({"tone": r.get("tone"), "question": r.get("question"),
                      "correct_answer": r.get("correct_answer"),
                      "is_final_turn": r.get("is_final_turn")}
                     if r["leg"] == "contradiction" else {})
            if "full" in requested_views:
                emit("full", r["exact_key"], text, {**base, **extra})
            tok = tokenizers[r["substrate"]]
            # Inkling's tokenizer adapter is not callable — encode/decode only.
            ids = list(tok.encode(text))[:PREFIX_TOKENS]
            if "prefix640" in requested_views:
                emit("prefix640", r["exact_key"], tok.decode(ids),
                     {**base, **extra, "prefix_tokenizer": BASE_MODEL[r["substrate"]]})
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
        "subject_counts": dict(Counter(str(r["subject"]) for r in join)),
        "cell_counts": dict(Counter(j["cell"] for j in join)),
        "view_by_cell": {f"{c}|{v}": n for (c, v), n in
                         sorted(Counter((j["cell"], j["view"]) for j in join).items())},
        "leg_counts": dict(Counter(j["leg"] for j in join)),
        "view_by_leg": {f"{l}|{v}": n for (l, v), n in
                        sorted(Counter((j["leg"], j["view"]) for j in join).items())},
        "prefix_tokens": PREFIX_TOKENS,
        "prefix_tokenizers": BASE_MODEL,
        "judge_input_response_cap_chars": CAP_CHARS,
        "blinding": ("judge call payload carries ONLY the response text (and, for the "
                     "coherence view, its battery prompt); subject/cell/adapter/"
                     "draw live only in the local join table"),
        "inputs_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "join_sha256": hashlib.sha256(args.join.read_bytes()).hexdigest(),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
