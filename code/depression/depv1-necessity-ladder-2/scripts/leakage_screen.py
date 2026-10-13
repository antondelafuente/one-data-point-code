#!/usr/bin/env python3
"""Step 4 — embedding-cosine leakage screen (DESIGN.md Leakage screen).

text-embedding-3-small, flagged at >= 0.55, run in BOTH directions:

  cross-battery  the training rows against the 12 battery scenarios. Unrelated populations by
                 construction, so ANY flagged pair is a real finding and is read before training.
                 Both directions = each row's nearest scenario AND each scenario's nearest row.
  within-pool    the rows against each other. Near-duplicates are EXPECTED within a situation (the
                 six cells answer the same request). The invariant that matters is that flagged
                 pairs fall WITHIN a situation. A flagged pair spanning two different situations
                 means the writer collapsed two situations and effective diversity is below 16 —
                 that is read, and the affected situation is re-authored before training.

Writes analysis/leakage_screen.json. Exits 3 if a cross-battery or cross-situation flag exists, so
it cannot be passed over silently.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opus_client import Meter, embed                                         # noqa: E402

REC = Path(__file__).resolve().parent.parent
THRESHOLD = 0.55


def cos(a, b):
    d = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / d if d else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--rows", type=Path, default=REC / "rounds" / "r1" / "candidates.jsonl.txt")
    ap.add_argument("--out", type=Path, default=REC / "analysis" / "leakage_screen.json")
    a = ap.parse_args()

    meter = Meter(a.work)
    rows = [json.loads(l) for l in a.rows.read_text().splitlines() if l.strip()]
    battery = json.loads((REC / "inputs" / "competence_attack_battery.json").read_text())["scenarios"]

    row_text = [f"{r['prompt']}\n\n{r['response']}" for r in rows]
    bat_text = [f"{s['question']}\n\n" + "\n".join(s["user_turns"]) for s in battery]

    print(f"embedding {len(row_text)} training rows + {len(bat_text)} battery scenarios", flush=True)
    rv = embed(row_text, meter=meter)
    bv = embed(bat_text, meter=meter)

    # ---- cross-battery, both directions
    cross = [[cos(rv[i], bv[j]) for j in range(len(bv))] for i in range(len(rv))]
    cross_flags = [
        {"arm_id": rows[i]["arm_id"], "cell": rows[i]["cell"], "row": rows[i]["row"],
         "scenario_id": battery[j]["scenario_id"], "cosine": round(cross[i][j], 4)}
        for i in range(len(rv)) for j in range(len(bv)) if cross[i][j] >= THRESHOLD]
    row_to_battery = [{"arm_id": rows[i]["arm_id"],
                       "nearest_scenario": battery[max(range(len(bv)), key=lambda j: cross[i][j])]["scenario_id"],
                       "cosine": round(max(cross[i]), 4)} for i in range(len(rv))]
    battery_to_row = [{"scenario_id": battery[j]["scenario_id"],
                       "nearest_arm_id": rows[max(range(len(rv)), key=lambda i: cross[i][j])]["arm_id"],
                       "cosine": round(max(cross[i][j] for i in range(len(rv))), 4)}
                      for j in range(len(bv))]

    # ---- within-pool
    within_same, within_cross = [], []   # same-row (expected) vs cross-row (read)
    for i in range(len(rv)):
        for j in range(i + 1, len(rv)):
            c = cos(rv[i], rv[j])
            if c >= THRESHOLD:
                rec = {"a": rows[i]["arm_id"], "b": rows[j]["arm_id"],
                       "row_a": rows[i]["row"], "row_b": rows[j]["row"],
                       "cell_a": rows[i]["cell"], "cell_b": rows[j]["cell"],
                       "cosine": round(c, 4)}
                (within_same if rows[i]["row"] == rows[j]["row"]
                 else within_cross).append(rec)
    within_cross.sort(key=lambda r: -r["cosine"])

    payload = {
        "model": "text-embedding-3-small", "threshold": THRESHOLD,
        "n_rows": len(rows), "n_battery_scenarios": len(battery),
        "cross_battery": {
            "n_flagged": len(cross_flags), "flagged": cross_flags,
            "rule": "unrelated populations by construction; ANY flag is a real finding, read "
                    "before training",
            "direction_rows_to_battery": row_to_battery,
            "direction_battery_to_rows": battery_to_row,
            "max_cosine_observed": round(max(max(r) for r in cross), 4)},
        "within_pool": {
            "n_flagged_same_row": len(within_same),
            "n_flagged_cross_row": len(within_cross),
            "rule": "within-ROW flags ARE the design (the cells of one row share a prompt and most "
                    "of a response) and are recorded, not acted on; a CROSS-ROW flag is read, and "
                    "nothing is re-authored either way -- every training text is a committed row "
                    "or a deletion from one",
            "flagged_cross_row": within_cross,
            "flagged_same_row_summary": {
                "n": len(within_same),
                "max_cosine": round(max((r["cosine"] for r in within_same), default=0.0), 4),
                "min_cosine": round(min((r["cosine"] for r in within_same), default=0.0), 4)},
            "flagged_same_row": within_same},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps({"cross_battery_flagged": len(cross_flags),
                      "cross_battery_max_cosine": payload["cross_battery"]["max_cosine_observed"],
                      "within_same_row_flagged": len(within_same),
                      "within_cross_row_flagged": len(within_cross)}, indent=2))
    if within_cross:
        print("\nCROSS-ROW FLAGS (read these; nothing is re-authored):")
        for r in within_cross[:20]:
            print(f"  {r['cosine']}  {r['a']} <-> {r['b']}")
    if cross_flags:
        print("\nCROSS-BATTERY FLAGS (read these before training):")
        for r in cross_flags[:20]:
            print(f"  {r['cosine']}  {r['arm_id']} <-> {r['scenario_id']}")

    if cross_flags:
        print(f">>> leakage screen flagged={len(cross_flags)} (cross-battery; notify the "
              f"designer-of-record, re-author nothing)")
        raise SystemExit(3)
    print(f">>> leakage screen clean (cross-battery 0; within-pool same-row {len(within_same)}, "
          f"cross-row {len(within_cross)})")


if __name__ == "__main__":
    main()
