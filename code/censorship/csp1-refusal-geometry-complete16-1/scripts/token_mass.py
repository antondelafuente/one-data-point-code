#!/usr/bin/env python3
"""Per-arm append-block COMPLETION-token mass at the fixed K=16 setting (a companion diagnostic).

The independent variable is the carrier-for-matched-filler REPLACEMENT BUNDLE at K=16: sixteen
carrier rows displace matched-filler ranks 1..16, and the completion-loss token mass changes with
them. DESIGN § What's measured keeps this reading FIXED at K=16 rather than varying it along a dose
axis — the "lower loss-token mass relative to filler" is inherited from the parent, not re-derived.

The append block is reconstructed IN MEMORY from the same frozen sources the builder uses (the
parent arm-schedule for carrier prompt/response text, the matched-filler manifest + floor corpus for
ranks 17..256), so all EIGHT categories are measured — including the four whose d16 training files
belong to the parent and are not rebuilt here. Mass is measured exactly the way the trainer sees it:
same renderer, same `conversation_to_datum(..., TrainOnWhat.LAST_ASSISTANT_MESSAGE)` path, same
`.strip()` the trainer's loader applies.
"""
from __future__ import annotations

import json

import numpy as np
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised.data import conversation_to_datum
from transformers import AutoTokenizer

from rgc16_common import (BASE, CATEGORIES, CONTROL_ARM, DOSE, NEW_CATEGORIES, PATHS, REG,
                          RENDERER, WORK, load_json, load_jsonl)

MAXLEN = 16384


def trainable_tokens(datum) -> int:
    """Count of tokens carrying loss. Tinker normalizes the weight vector to sum 1.0 under the
    'mean' reduction, so the COUNT is the nonzero entries, never the sum."""
    for attr in ("loss_fn_inputs", "loss_inputs"):
        obj = getattr(datum, attr, None)
        if obj is not None:
            w = obj.get("weights") if isinstance(obj, dict) else getattr(obj, "weights", None)
            if w is not None:
                return int((np.asarray(w.to_numpy()) > 0).sum())
    raise RuntimeError("no loss weights on datum")


def total_tokens(datum) -> int:
    mi = getattr(datum, "model_input", None)
    n = getattr(mi, "length", None)
    return int(n) if isinstance(n, int) else len(mi.to_ints())


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)
    cache: dict[tuple[str, str], tuple[int, int]] = {}

    def measure(prompt: str, response: str):
        key = (prompt.strip(), response.strip())
        if key not in cache:
            d = conversation_to_datum(
                [{"role": "user", "content": key[0]}, {"role": "assistant", "content": key[1]}],
                renderer, max_length=MAXLEN * 50, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
            cache[key] = (trainable_tokens(d), total_tokens(d))
        return cache[key]

    sched = load_json(PATHS["arm-schedule.json"])
    pair = {p["prompt_id"]: p for p in sched["pairs"]}
    dose = load_json(REG / "design_inputs" / "dose_schedule.json")
    picks = {a["category"]: sorted(a["exposures"], key=lambda x: (x["copy_index"], x["slot"]))
             for a in dose["arms"] if a["dose"] == DOSE}
    mf = load_json(PATHS["matched-filler256-manifest.json"])
    floor = {r["id"]: r for r in load_jsonl(PATHS["train_sb_floor.jsonl"])}
    filler_by_rank = {r["rank"]: floor[r["id"]] for r in mf["rows"]}
    assert set(filler_by_rank) == set(range(1, 257))

    # the control's per-rank completion mass, and the displaced block (ranks 1..K)
    control_rank = {p: measure(filler_by_rank[p]["prompt"], filler_by_rank[p]["response"])
                    for p in range(1, 257)}
    ctrl_comp = sum(v[0] for v in control_rank.values())
    ctrl_tot = sum(v[1] for v in control_rank.values())

    out = [{"arm": CONTROL_ARM, "training_category": None, "dose": 0,
            "membership": "control", "carrier_rows": 0, "matched_filler_rows": 256,
            "carrier_completion_tokens": 0, "carrier_total_tokens": 0,
            "retained_filler_completion_tokens": ctrl_comp, "retained_filler_total_tokens": ctrl_tot,
            "append_completion_tokens": ctrl_comp, "append_total_tokens": ctrl_tot,
            "mean_carrier_completion_tokens": None,
            "displaced_filler_completion_tokens": 0}]
    for cat in CATEGORIES:
        exps = picks[cat]
        assert len(exps) == DOSE
        cm = [measure(pair[e["prompt_id"]]["prompt"], pair[e["prompt_id"]]["response"]) for e in exps]
        c_comp, c_tot = sum(x[0] for x in cm), sum(x[1] for x in cm)
        f_comp = sum(control_rank[p][0] for p in range(DOSE + 1, 257))
        f_tot = sum(control_rank[p][1] for p in range(DOSE + 1, 257))
        displaced = sum(control_rank[p][0] for p in range(1, DOSE + 1))
        out.append({"arm": f"{cat}_d{DOSE}", "training_category": cat, "dose": DOSE,
                    "membership": "newly_measured" if cat in NEW_CATEGORIES else "previously_measured",
                    "carrier_rows": DOSE, "matched_filler_rows": 256 - DOSE,
                    "carrier_completion_tokens": c_comp, "carrier_total_tokens": c_tot,
                    "retained_filler_completion_tokens": f_comp, "retained_filler_total_tokens": f_tot,
                    "append_completion_tokens": c_comp + f_comp,
                    "append_total_tokens": c_tot + f_tot,
                    "mean_carrier_completion_tokens": round(c_comp / DOSE, 2),
                    "displaced_filler_completion_tokens": displaced})
    for r in out:
        r["control_append_completion_tokens"] = ctrl_comp
        r["append_completion_tokens_vs_control"] = r["append_completion_tokens"] - ctrl_comp

    doc = {"renderer": RENDERER, "tokenizer": BASE,
           "loss": "LAST_ASSISTANT_MESSAGE (completion-only)", "dose": DOSE,
           "reconstruction": "append block rebuilt in memory from the frozen arm-schedule, "
                             "matched-filler manifest and floor corpus — identical to the builder's "
                             "rows, so all eight categories are measured at K=16",
           "note": "Every arm replaces the SAME sixteen matched-filler ranks with its own sixteen "
                   "carrier rows, so the append block differs between arms only in that block's "
                   "completion-token mass. This is a fixed-K companion diagnostic (DESIGN § What's "
                   "measured), not a dose-axis reading.",
           "arms": out}
    p = WORK / "receipts" / "token_mass.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n")
    for r in out:
        print(f"{r['arm']:28s} K={r['dose']:3d} carrier_comp={r['carrier_completion_tokens']:7d} "
              f"displaced={r['displaced_filler_completion_tokens']:7d} "
              f"append_comp={r['append_completion_tokens']:8d} "
              f"vs_control={r['append_completion_tokens_vs_control']:+8d}")


if __name__ == "__main__":
    main()
