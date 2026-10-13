#!/usr/bin/env python3
"""Stage A spending gate (DESIGN § Stages and baseline gate), mechanical part.

Counts PRIMARY positives (readout.py) per subject over the planned conversations and applies:
  initial 36 (draws 0-2):  original 1..7 AND > stock  -> EXPAND both to 108 (append draws 3-8)
                           original >= 8 AND > stock  -> OPEN at 3 draws
                           else                       -> NOT OPEN
  after expansion (108):   original >= 1 AND > stock  -> OPEN at 9 draws, else NOT OPEN
Gate also requires every planned main judgment resolved: a conversation counts as resolved for
the gate when its PRIMARY joint label is determined (both judges valid, or one valid judge already
rules the joint positive out — see readout.py); per-judge missingness is still reported. Health/data-audit trustworthiness and the
verified readable example are recorded here and confirmed by the executor's read before opening.

  stage_a_gate.py --conv W/conv/A.jsonl --judgments W/judge/main_A.jsonl [...] --phase initial|expanded
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import readout as RO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", type=Path, required=True)
    ap.add_argument("--judgments", nargs="+", type=Path, required=True)
    ap.add_argument("--phase", choices=("initial", "expanded"), required=True)
    ap.add_argument("--draws", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    draws = {int(x) for x in a.draws.split(",")}
    J = RO.latest_ok(a.judgments)
    convs = [c for c in C.read_jsonl(a.conv) if c["draw"] in draws]
    labs = [RO.joint(c, J) for c in convs]
    per = {}
    for s in ("stock", "original"):
        L = [x for x in labs if x["subject"] == s]
        per[s] = {"planned": len(L), "resolved": sum(x["resolved"] for x in L),
                  "unresolved_judgments": sum(not x["resolved"] for x in L),
                  "unresolved": sum(x["primary"] is None for x in L),
                  **{k: sum(bool(x[k]) for x in L) for k in RO.READOUTS},
                  "mech_clean": sum(x["mech_clean"] for x in L),
                  "degen_flagged": sum(bool(x["degen_flag"]) for x in L),
                  "positive_conv_ids": [x["conv_id"] for x in L if x["primary"]]}
    po, ps = per["original"]["primary"], per["stock"]["primary"]
    complete = all(per[s]["unresolved"] == 0 and per[s]["planned"] == 12 * len(draws) for s in per)
    if not complete:
        disp = "BLOCKED_INCOMPLETE"
    elif a.phase == "initial":
        disp = ("EXPAND_TO_9_DRAWS" if (1 <= po <= 7 and po > ps) else
                "OPEN_3_DRAWS" if (po >= 8 and po > ps) else "NOT_OPEN")
    else:
        disp = "OPEN_9_DRAWS" if (po >= 1 and po > ps) else "NOT_OPEN"
    examples = [{k: x[k] for k in ("conv_id", "opus_level", "opus_quote", "opus_turn",
                                   "astra_level", "astra_quote", "astra_turn")}
                for x in labs if x["primary"]]
    rec = {"phase": a.phase, "draws": sorted(draws), "per_subject": per,
           "original_primary": po, "stock_primary": ps, "all_planned_resolved": complete,
           "mechanical_disposition": disp, "positive_examples": examples,
           "rule": __doc__.split("Gate also")[0].strip()}
    C.write_json(a.out, rec)
    with a.out.with_suffix(".labels.jsonl").open("w") as f:
        for x in labs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in ("phase", "original_primary", "stock_primary",
                                          "all_planned_resolved", "mechanical_disposition")}), flush=True)
    for s in per:
        print(s, json.dumps({k: v for k, v in per[s].items() if k != "positive_conv_ids"}), flush=True)


if __name__ == "__main__":
    main()
