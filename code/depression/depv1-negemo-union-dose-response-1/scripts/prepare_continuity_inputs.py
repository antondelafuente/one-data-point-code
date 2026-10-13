#!/usr/bin/env python3
"""Build the JUDGE-CONTINUITY gate's blinded judge inputs (the ORDER GATE).

DESIGN "Judge-continuity gate": BEFORE any full judge spend, re-judge a pinned
set of the PARENT experiment's FROZEN rollouts with this wave's instrument —

  anchor   — every rollout of the parent's anchor cell `off_c1001-n2000-s2001` (396)
  base     — the first 100 rollouts of the parent's `off_base` cell, by rollout_row_id

Gate: paired mean drift |new - parent| <= 0.10 on EACH set (paired per-rollout, so
systematic judge/transport drift cannot hide inside the wide across-scenario CI).

The judge call payload carries ONLY the response text — cell / dose / the parent's
own rating live in the local join table, exactly as in the main wave.

Inputs are a LOCAL SLICE of the parent's archived artifacts, pulled into this run's
own tree before use (#458): this script reads only local files and writes only local
files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPERIMENT = "depv1-negemo-union-dose-response-1"
ANCHOR_CELL = "off_c1001-n2000-s2001"
BASE_CELL = "off_base"
BASE_N = 100


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def jid(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-rollouts", type=Path, required=True)
    ap.add_argument("--parent-join", type=Path, required=True)
    ap.add_argument("--parent-judgments", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--join", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()

    # the parent's own full-response ratings, keyed by exact_key
    join_by_id = {j["judge_row_id"]: j for j in read_jsonl(args.parent_join)}
    parent_rating: dict[str, float] = {}
    for j in read_jsonl(args.parent_judgments):
        if j.get("view") != "full" or j.get("parse_status") != "valid":
            continue
        meta = join_by_id.get(j["judge_row_id"])
        if meta is None or meta.get("population") != "new":
            continue
        parent_rating[meta["exact_key"]] = float(j["rating"])

    wanted = {ANCHOR_CELL: [], BASE_CELL: []}
    for r in read_jsonl(args.parent_rollouts):
        if r["cell"] in wanted:
            wanted[r["cell"]].append(r)

    anchor = sorted(wanted[ANCHOR_CELL], key=lambda r: r["rollout_row_id"])
    base = sorted(wanted[BASE_CELL], key=lambda r: r["rollout_row_id"])[:BASE_N]
    assert len(anchor) == 396, f"anchor set is {len(anchor)}, expected 396"
    assert len(base) == BASE_N, f"base set is {len(base)}, expected {BASE_N}"

    inputs, join = [], []
    for population, rows in (("continuity_anchor", anchor), ("continuity_base", base)):
        for r in rows:
            key = r["exact_key"]
            if key not in parent_rating:
                raise SystemExit(f"no parent full-view rating for {key}")
            text = r["judge_visible_text"]
            judge_row_id = jid(f"{EXPERIMENT}|continuity|{population}|{key}")
            inputs.append({
                "judge_row_id": judge_row_id, "view": "full",
                "response": text,  # the ONLY content the judge call ever sees
                "rubric": "negativity", "battery_prompt": None,
            })
            join.append({
                "judge_row_id": judge_row_id, "view": "full",
                "population": population,
                "exact_key": key, "rollout_row_id": r["rollout_row_id"],
                "cell": r["cell"], "leg": "continuity", "adapter": r["adapter"],
                "chain": r["chain"], "n_rows": r["n_rows"],
                "training_seed": r["training_seed"],
                "thinking_mode": r["thinking_mode"], "draw": r["draw"],
                "item_id": r["item_id"], "turn": r["turn"],
                "scenario_type": r.get("scenario_type"),
                "parent_rating": parent_rating[key],
                "response_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "response_chars": len(text),
            })

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

    pops = Counter(j["population"] for j in join)
    receipt = {
        "experiment": EXPERIMENT,
        "gate": "judge_continuity",
        "n_judge_calls": len(inputs),
        "population_counts": dict(pops),
        "parent_mean_by_population": {
            p: round(sum(j["parent_rating"] for j in join if j["population"] == p)
                     / pops[p], 6) for p in pops},
        "parent_rollouts_sha256": hashlib.sha256(
            args.parent_rollouts.read_bytes()).hexdigest(),
        "inputs_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "join_sha256": hashlib.sha256(args.join.read_bytes()).hexdigest(),
        "blinding": ("judge call payload carries ONLY the response text; cell / dose / "
                     "the parent's own rating live only in the local join table"),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
