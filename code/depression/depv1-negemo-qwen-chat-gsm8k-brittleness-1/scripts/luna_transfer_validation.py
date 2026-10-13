#!/usr/bin/env python3
"""Select the 24 Leg-B transcripts for the blinded Luna domain-transfer spot-validation.

DESIGN "Results discipline": three per scientific family x condition (4 families x 2 conditions
= 24), sampling BOTH Luna labels and confidence extremes wherever those strata exist. The
selection is deterministic: within a family x condition cell the strata are tried in a fixed
order and, inside a stratum, the LOWEST dataset_index wins. An empty requested stratum is
replaced deterministically by the next-lowest dataset index in that family/condition, and the
absence is recorded.

Two modes:
  select  -> writes the blind pack (transcripts WITHOUT Luna's labels) + a sealed key
  score   -> joins the executor's blind labels back to Luna and writes the agreement receipt
This is a TRANSFER DIAGNOSTIC. It never changes a Luna label and never tunes the frozen rubric.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

FAMILIES = ["stock", "gemma_full", "row4_original_I", "regpred_row11"]
CONDITIONS = ["neutral_verify", "false_attack"]
PER_CELL = 3
# stratum order: an incoherent call first (rarest), then the confidence extremes of coherent
STRATA = [("incoherent", None), ("coherent", "low"), ("coherent", "high"), ("coherent", "medium"),
          ("incoherent", "high"), ("incoherent", "low")]


def read_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def family_of(subject: str) -> str:
    if subject.startswith("stock"):
        return "stock"
    if subject == "gemma_full":
        return "gemma_full"
    return subject.rsplit("-s", 1)[0]


def select(a) -> None:
    convs = {c["id"]: c for c in read_jsonl(a.transcripts)}
    labels = {r["id"]: r for r in read_jsonl(a.labels)}
    excluded = set()
    if a.exclude_ids and a.exclude_ids.exists():
        excluded = {ln.strip() for ln in a.exclude_ids.read_text().splitlines() if ln.strip()}
    cells = defaultdict(list)
    for cid, c in convs.items():
        if c["subject"] == a.exclude_subject or cid in excluded:
            continue
        lab = labels.get(cid)
        if not lab or not lab.get("parse_ok"):
            continue
        cells[(family_of(c["subject"]), c["condition"])].append((c, lab))

    picks, absences = [], []
    for fam in FAMILIES:
        for cond in CONDITIONS:
            pool = sorted(cells.get((fam, cond), []), key=lambda t: (t[0]["dataset_index"],
                                                                    t[0]["subject"]))
            chosen, used = [], set()
            for label, conf in STRATA:
                if len(chosen) >= PER_CELL:
                    break
                hits = [t for t in pool if t[1]["label"] == label
                        and (conf is None or t[1]["confidence"] == conf)
                        and t[0]["id"] not in used]
                if not hits:
                    absences.append({"family": fam, "condition": cond, "stratum": f"{label}/{conf}",
                                     "resolution": "empty stratum; replaced by next-lowest "
                                                   "dataset_index in this family/condition"})
                    continue
                pick = hits[0]
                used.add(pick[0]["id"])
                chosen.append((f"{label}/{conf}", pick))
            for t in pool:                      # deterministic backfill: lowest dataset_index
                if len(chosen) >= PER_CELL:
                    break
                if t[0]["id"] not in used:
                    used.add(t[0]["id"])
                    chosen.append(("backfill_lowest_dataset_index", t))
            for stratum, (c, lab) in chosen:
                picks.append({"family": fam, "condition": cond, "stratum": stratum,
                              "id": c["id"], "subject": c["subject"],
                              "dataset_index": c["dataset_index"],
                              "luna_label": lab["label"], "luna_confidence": lab["confidence"],
                              "luna_breakdown_type": lab["breakdown_type"]})

    a.blind_out.parent.mkdir(parents=True, exist_ok=True)
    with a.blind_out.open("w") as fh:
        for i, p in enumerate(sorted(picks, key=lambda p: p["id"])):
            fh.write(json.dumps({"blind_id": f"T{i:02d}", "trajectory": convs[p["id"]]["trajectory"]},
                                ensure_ascii=False) + "\n")
    key = {f"T{i:02d}": p for i, p in enumerate(sorted(picks, key=lambda p: p["id"]))}
    a.key_out.write_text(json.dumps({"n_selected": len(picks), "per_cell_target": PER_CELL,
                                     "empty_strata": absences, "key": key},
                                    indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_selected": len(picks), "n_empty_strata": len(absences),
                      "cells": sorted({f"{p['family']}|{p['condition']}" for p in picks})},
                     indent=2))


def score(a) -> None:
    key = json.loads(a.key_out.read_text())
    mine = {r["blind_id"]: r for r in read_jsonl(a.blind_labels)}
    rows, agree = [], 0
    for bid, meta in sorted(key["key"].items()):
        mine_label = mine[bid]["label"]
        ok = mine_label == meta["luna_label"]
        agree += ok
        rows.append({"blind_id": bid, "id": meta["id"], "family": meta["family"],
                     "condition": meta["condition"], "stratum": meta["stratum"],
                     "luna_label": meta["luna_label"], "luna_confidence": meta["luna_confidence"],
                     "executor_label": mine_label, "agree": ok,
                     "executor_note": mine[bid].get("note")})
    receipt = {"status": "COMPLETE", "n": len(rows), "n_agree": agree,
               "agreement_rate": agree / len(rows) if rows else None,
               "disagreements": [r for r in rows if not r["agree"]],
               "available_label_counts": {
                   lab: sum(1 for r in rows if r["luna_label"] == lab)
                   for lab in ("coherent", "incoherent")},
               "empty_strata": key["empty_strata"],
               "policy": ("transfer diagnostic only: no Luna label was changed and the frozen "
                          "rubric was not tuned"),
               "rows": rows}
    a.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("select", "score"))
    ap.add_argument("--transcripts", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--exclude-subject", default="stock_gB")
    ap.add_argument("--exclude-ids", type=Path,
                    help="newline-delimited conversation ids to skip — used ONLY for the "
                         "supplementary uncontaminated pack, never for the DESIGN-specified one")
    ap.add_argument("--blind-out", type=Path)
    ap.add_argument("--key-out", type=Path, required=True)
    ap.add_argument("--blind-labels", type=Path)
    ap.add_argument("--receipt-out", type=Path)
    a = ap.parse_args()
    (select if a.mode == "select" else score)(a)


if __name__ == "__main__":
    main()
