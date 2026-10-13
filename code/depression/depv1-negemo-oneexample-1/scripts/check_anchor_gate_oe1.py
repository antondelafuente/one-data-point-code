#!/usr/bin/env python3
"""The ANCHOR GATE — judged FIRST, before any trained cell is judged. FAIL = STOP.

DESIGN "Anchor gate" (this wave's own statement, verbatim in intent):

  (a) stock contradiction FULL-view mean in [0.3, 1.4] — a deliberately wide
      first-wave band on the NEW judge scale, anchored on the bridge's 0.812 read of
      stock on stored DIS1 rollouts. A breakage tripwire for the wrong model or a
      broken prompt, not a precision claim.
  (b) the paired delta(p1a-s4 re-serve - stock) CI excludes zero on contradiction,
      with the re-serve ABOVE stock — the known-hot ordering gate.

Failure = STOP + flag the designer-of-record. These two values are also the first
wobble-ledger entries on the new judge scale.

Reads the same committed `cells.csv` / `contrasts.json` the record reports, so the
gate and the record cannot disagree. An ABSENT contrast is an ERROR, never a failing
statistic (DIS1-A2's lesson: reading a missing record as "CI includes zero" is a bug,
not a result).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402

STOCK_BAND = C.STOCK_CONTRADICTION_BAND        # (0.3, 1.4)
# Bridge-record context printed beside the gate, never gated on: the Sonnet-5 read of
# stock and of p1a-s4 on STORED DIS1 rollouts. Fresh rollouts + judge-draw noise move
# these, which is exactly why the band above is wide.
BRIDGE_CONTEXT = {"stock_on_stored_dis1_rollouts": 0.812,
                  "p1a_s4_on_stored_dis1_rollouts": 1.9}


def rows_of(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--contrasts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cells = rows_of(args.cells)
    contrasts = json.loads(args.contrasts.read_text())["contrasts"]

    def anchor(row_id):
        for r in cells:
            if r["row_id"] == row_id and r["seed"] == row_id:
                return r
        return None

    checks: list[dict] = []

    st = anchor("stock")
    sm = num(st["contradiction_mean"]) if st else None
    checks.append({
        "check": "(a) stock contradiction full-view mean within the DESIGN band",
        "band": list(STOCK_BAND), "value": sm,
        "n_judged_rows": (st or {}).get("n_judged_rows"),
        "ci": [num((st or {}).get("contradiction_ci_lo")),
               num((st or {}).get("contradiction_ci_hi"))],
        "pass": bool(sm is not None and STOCK_BAND[0] <= sm <= STOCK_BAND[1]),
    })

    gate = next((c for c in contrasts if c.get("contrast") == "anchor_gate"), None)
    if gate is None or gate.get("delta") is None:
        raise SystemExit(
            "ANCHOR GATE ERROR (not a statistical failure): the paired contrast "
            "p1a_s4_reserve vs stock is ABSENT from contrasts.json. Fix the analysis "
            "and re-run the gate on the SAME judgments — do not re-judge, and do not "
            "read this as a gate FAIL.")
    re_serve = anchor("p1a_s4_reserve")
    rm = num(re_serve["contradiction_mean"]) if re_serve else None
    checks.append({
        "check": "(b) paired delta(p1a-s4 re-serve - stock) CI excludes zero",
        "delta": gate.get("delta"), "ci": [gate.get("ci_lo"), gate.get("ci_hi")],
        "n_scenarios": gate.get("n_scenarios"),
        "pass": bool(gate.get("excludes_zero") and (gate.get("delta") or 0) > 0),
    })
    checks.append({
        "check": "(b) re-serve mean ABOVE stock",
        "reserve_mean": rm, "stock_mean": sm,
        "pass": bool(rm is not None and sm is not None and rm > sm),
    })

    verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    report = {
        "gate": "ANCHOR GATE (DESIGN.md) — judged before any trained cell is judged",
        "verdict": verdict,
        "judge": {"model": "claude-sonnet-5", "transport": "anthropic_messages"},
        "checks": checks,
        "wobble_ledger_entries": {
            "stock_contradiction_full_mean": sm,
            "p1a_s4_reserve_contradiction_full_mean": rm,
            "scale": "NEW judge (claude-sonnet-5); NOT comparable to 4.5-era values",
        },
        "context_not_gated": BRIDGE_CONTEXT,
        "action_if_fail": ("STOP. Do not judge the trained cells. Flag the "
                           "designer-of-record: if the known-hot p1a-s4 re-serve cannot "
                           "be ordered above known-cold stock in THIS wave, the "
                           "instrument/transport/serving stack has moved and nothing "
                           "downstream is interpretable."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if verdict != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
