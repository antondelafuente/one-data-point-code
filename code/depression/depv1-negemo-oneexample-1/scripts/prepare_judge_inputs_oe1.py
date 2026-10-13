#!/usr/bin/env python3
"""Build the ONE blinded judge-input wave for depv1-negemo-oneexample-1 (OE1).

Pulled from `depv1-negemo-int13-splits-1/scripts/prepare_judge_inputs_dis1.py`,
keeping its construction verbatim: interleaved views in ONE wave, all blinded — the
judge call payload carries only the judged text (and, for coherence, its battery
prompt); subject / cell / adapter / draw live only in this local join table.

  full       — full-response negativity on every CONTRADICTION rollout of all 46
               judged subjects (the wave's canonical metric)
  coherence  — the 30-prompt battery x draws per subject, pinned rubric

DECLARED DELTAS vs DIS1, both forced by THIS DESIGN "Measurement":
  * the `prefix640` co-view is NOT emitted. The DESIGN names this a lean choice:
    DIS1 measured contradiction as stable across full/prefix-640 and this wave runs
    no rejection battery, so the view-sensitivity finding lives on the rejection
    side. Raw rollouts are persisted, so the co-view is judgeable post hoc.
  * there is no `rejection` leg to emit — contradiction is the only affect battery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402

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
    ap.add_argument("--views", default="full",
                    help="comma-separated contradiction views; coherence always emitted")
    args = ap.parse_args()

    rollouts = C.read_jsonl(args.rollouts)
    requested_views = {x for x in args.views.split(",") if x}
    if requested_views != {"full"}:
        raise SystemExit(f"this DESIGN pins the full view only; got {requested_views}")
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
                "row_id": r.get("row_id"), "budget": r.get("budget"),
                "training_manifest_sha256": r.get("training_manifest_sha256"),
                "population": "new"}
        if r["leg"] == "contradiction":
            extra = {"tone": r.get("tone"), "question": r.get("question"),
                     "correct_answer": r.get("correct_answer"),
                     "is_final_turn": r.get("is_final_turn")}
            emit("full", r["exact_key"], text, {**base, **extra})
        elif r["leg"] == "battery":
            emit("coherence", r["exact_key"], text,
                 {**base, "battery_prompt": r["messages"][0]["content"],
                  "battery_category": r.get("battery_category"),
                  "prompt_index": r.get("prompt_index")})
        else:
            raise SystemExit(f"unexpected leg {r['leg']!r} in rollouts")

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
        "views_emitted": ("full (contradiction) + coherence; prefix640 deliberately "
                          "NOT judged this wave, see module docstring"),
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
