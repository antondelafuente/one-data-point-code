#!/usr/bin/env python3
"""The ANCHOR GATE — judged FIRST, before any arm is judged. FAIL = STOP.

DESIGN.md (this wave's own statement, NOT the parent's non-overlapping-CI form):

  (a) stock full-39 mean in [0.15, 0.40];
  (b) the int13 re-serve pair is an ORDERING gate — the paired
      delta(int13 pooled - stock) CI excludes zero on BOTH primary batteries
      (rejection protected-19, contradiction 4-turn) AND each seed's mean sits
      above stock's on both.

Gate failure = STOP and flag the designer-of-record.

DELTAS vs DI13's `check_anchor_gate.py`: the re-serve check is an ORDERING gate
read from the paired bootstrap in `contrasts.json` (the parent gated on
non-overlapping marginal CIs on one battery); the pair is int13-s4/s5, not the
top64 pair; and per the standing rule the gate is NEVER pinned to the parent
wave's scenario-bootstrap bands — the parent values are printed as CONTEXT with
the +/-0.08 wobble, never gated.

Reads the same committed CSV/JSON the record reports, so the gate and the record
cannot disagree.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

STOCK_BAND = (0.15, 0.40)
# The parent wave's own protected-19 / contradiction pooled values, printed as
# CONTEXT beside the gate (the wobble-ledger reference) — never gated on.
INT13_1_REFERENCE = {"rejection": {"stock": 0.272, "int13": 0.589},
                     "contradiction": {"stock": 0.312, "int13": 0.837}}
WOBBLE = 0.08


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
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--contrasts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    grid = [r for r in rows_of(args.grid)
            if r.get("basis", "all_valid") == "all_valid" and r.get("view") == "full"]
    contrasts = json.loads(args.contrasts.read_text())["contrasts"]

    def cell(subject, seed, battery, sset):
        for r in grid:
            if (r["subject"] == subject and r["seed"] == seed
                    and r["battery"] == battery and r["scenario_set"] == sset):
                return r
        return None

    def contrast(a, b, battery, sset, basis):
        for c in contrasts:
            if (c.get("arm_a") == a and c.get("arm_b") == b
                    and c["battery"] == battery and c["scenario_set"] == sset
                    and c.get("basis") == basis):
                return c
        return None

    checks: list[dict] = []

    # ---- check (a): stock's full-39 full-view mean inside the measured band ----
    s39 = cell("stock", "stock", "rejection", "full39")
    m39 = num(s39["mean"]) if s39 else None
    checks.append({
        "check": "(a) stock full-39 full-view mean within the measured band",
        "band": list(STOCK_BAND), "value": m39,
        "pass": bool(m39 is not None and STOCK_BAND[0] <= m39 <= STOCK_BAND[1]),
        "note": "DESIGN cites 0.275-0.333 across the last three waves; int13-1 read 0.312",
    })

    # ---- check (b1): paired delta(int13 pooled - stock) CI excludes zero, BOTH batteries
    for battery, sset in (("rejection", "protected19"), ("contradiction", "contradiction")):
        c = contrast("int13", "stock", battery, sset, "pooled")
        # DIS1-A2: a MISSING contrast is an ERROR, never a failing statistic. `None` is
        # not a statistical result — reading an absent record as "CI includes zero" is
        # exactly the bug this gate shipped with. Fail loudly and differently.
        if c is None or c.get("delta") is None:
            raise SystemExit(
                "ANCHOR GATE ERROR (not a statistical failure): the paired contrast "
                f"int13_vs_stock / {battery} / {sset} / basis=pooled is ABSENT from "
                "contrasts.json. Fix the analysis (is the pair in PAIRS? do the "
                "arm/scenario_set keys match?) and re-run the gate on the SAME "
                "judgments — do not re-judge, and do not read this as a gate FAIL. "
                "See DESIGN_AMENDMENTS.md DIS1-A2.")
        ok = bool(c.get("excludes_zero") and (c.get("delta") or 0) > 0)
        checks.append({
            "check": f"(b1) paired delta(int13 pooled - stock) CI excludes zero, "
                     f"{battery}/{sset}",
            "delta": c.get("delta"),
            "ci": [c.get("ci_lo"), c.get("ci_hi")],
            "n_scenarios": c.get("n_scenarios"),
            "pass": ok,
        })

    # ---- check (b2): EACH seed's mean above stock's, on BOTH batteries ---------
    for battery, sset in (("rejection", "protected19"), ("contradiction", "contradiction")):
        st = cell("stock", "stock", battery, sset)
        sm = num(st["mean"]) if st else None
        for seed in ("4", "5"):
            c = cell("int13", seed, battery, sset)
            m = num(c["mean"]) if c else None
            checks.append({
                "check": f"(b2) int13-s{seed} mean above stock, {battery}/{sset}",
                "reserve_mean": m, "stock_mean": sm,
                "pass": bool(m is not None and sm is not None and m > sm),
            })

    # ---- context only: the parent wave's values + the wobble band --------------
    context = []
    for battery, sset in (("rejection", "protected19"), ("contradiction", "contradiction")):
        ref = INT13_1_REFERENCE["rejection" if battery == "rejection" else "contradiction"]
        for subject in ("stock", "int13"):
            seed = "stock" if subject == "stock" else "pooled"
            c = cell(subject, seed, battery, sset)
            m = num(c["mean"]) if c else None
            context.append({
                "subject": subject, "battery": battery, "scenario_set": sset,
                "this_wave_mean": m, "int13_1_value": ref[subject],
                "delta_vs_int13_1": (round(m - ref[subject], 4) if m is not None else None),
                "within_wobble_band": (abs(m - ref[subject]) <= WOBBLE
                                       if m is not None else None),
                "gated": False,
                "note": "CONTEXT ONLY — the standing rule forbids pinning the re-serve "
                        "gate to the parent wave's scenario-bootstrap bands; this is the "
                        "wobble-ledger read, not a gate",
            })

    verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    report = {
        "gate": "ANCHOR GATE (DESIGN.md) — judged before any arm judging",
        "verdict": verdict,
        "checks": checks,
        "context_not_gated": context,
        "action_if_fail": ("STOP. Do not judge the arms. Flag the designer-of-record: "
                           "if the known-hot int13 re-serves cannot be ordered above "
                           "known-cold stock in THIS wave, the instrument/transport/"
                           "serving stack has moved and nothing downstream is "
                           "interpretable."),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if verdict != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
