#!/usr/bin/env python3
"""Build the canonical cell plan for depv1-negemo-union-dose-response-1.

Writes `state/cells.json`: every thinking-off rejection-eval cell and every
battery cell in the DESIGN tables, with its adapter, rendering mode, draw count,
planned response count, and pod assignment. This is the CANONICAL CHECKLIST the
generation and judging legs diff against — reactive batching has no signal on a
silently skipped cell (#337).

Declared deltas from the parent's plan_cells.py:
  * the thinking-on spine is DROPPED (DESIGN "Eval spines and cells");
  * adapters are the 10 union-filtered chains (`c1001F` / `c1002F`) plus the
    parent's own `c1001-n2000-s2001` adapter re-served as the SERVING CANARY;
  * 4 pods instead of 8 (DESIGN "Schedule / fan-out"), with chains deliberately
    spread across pods.
"""
from __future__ import annotations

import json
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent
OUT = WORK / "state/cells.json"

TURNS_TOTAL = 132  # assistant turns over the 39-conversation scenario set
BATTERY_PROMPTS = 30

# adapter run-id -> (chain label, N, training seed).
ADAPTERS = {
    "c1001F-n32-s2001": ("1001F", 32, 2001),
    "c1001F-n128-s2001": ("1001F", 128, 2001),
    "c1001F-n512-s2001": ("1001F", 512, 2001),
    "c1001F-n1024-s2001": ("1001F", 1024, 2001),
    "c1001F-n2000-s2001": ("1001F", 2000, 2001),
    "c1001F-n8000-s2001": ("1001F", 8000, 2001),
    "c1001F-n17995-s2001": ("1001F", 17995, 2001),
    "c1002F-n512-s2001": ("1002F", 512, 2001),
    "c1002F-n1024-s2001": ("1002F", 1024, 2001),
    "c1002F-n2000-s2001": ("1002F", 2000, 2001),
    # SERVING CANARY — the PARENT experiment's converted adapter, re-served and
    # re-evaluated in this wave. Trained on the parent's unfiltered chain-1001
    # rows, so it is NOT a point on this experiment's curve; its band is the
    # parent's measured 0.742424 [0.5068, 0.9926].
    "c1001-n2000-s2001": ("parent-1001", 2000, 2001),
}

CANARY_ADAPTER = "c1001-n2000-s2001"

# 5 draws on the endpoints (the untrained floor and the full-pool endpoint —
# the parent's endpoint convention, inherited by citation), 3 on interior cells.
ENDPOINT_DRAWS, INTERIOR_DRAWS = 5, 3
ENDPOINTS = {"base", "c1001F-n17995-s2001"}

# thinking-off ONLY (DESIGN drops the thinking-on bundle spine).
OFF_CELLS = ["base"] + list(ADAPTERS)
# battery: every thinking-off cell, same as parent.
BATTERY_CELLS = OFF_CELLS

# adapter -> pod (4 H200s). At most 3 LoRAs per pod; chains deliberately span
# pods so a per-pod serving quirk cannot land entirely on one chain.
POD_OF_ADAPTER = {
    "base": 0, "c1001F-n32-s2001": 0, "c1002F-n512-s2001": 0,
    "c1001F-n17995-s2001": 1, "c1001F-n128-s2001": 1, "c1002F-n1024-s2001": 1,
    "c1001F-n512-s2001": 2, "c1001F-n2000-s2001": 2, "c1002F-n2000-s2001": 2,
    "c1001F-n1024-s2001": 3, "c1001F-n8000-s2001": 3, CANARY_ADAPTER: 3,
}


def meta(adapter: str) -> dict:
    if adapter == "base":
        return {"chain": None, "n": 0, "training_seed": None,
                "pool": None, "role": "untrained_floor"}
    chain, n, seed = ADAPTERS[adapter]
    if adapter == CANARY_ADAPTER:
        return {"chain": chain, "n": n, "training_seed": seed,
                "pool": "parent_unfiltered", "role": "serving_canary"}
    return {"chain": chain, "n": n, "training_seed": seed,
            "pool": "union_clean", "role": "dose_cell"}


def main() -> None:
    cells = []
    for adapter in OFF_CELLS:
        draws = ENDPOINT_DRAWS if adapter in ENDPOINTS else INTERIOR_DRAWS
        cells.append({
            "cell": f"off_{adapter}", "leg": "rejection", "adapter": adapter,
            "thinking": False, "draws": draws,
            "planned_responses": TURNS_TOTAL * draws,
            "pod": POD_OF_ADAPTER[adapter], "prefix640_judged": True, **meta(adapter),
        })
    for adapter in BATTERY_CELLS:
        cells.append({
            "cell": f"bat_{adapter}", "leg": "battery", "adapter": adapter,
            "thinking": False, "draws": 1,
            "planned_responses": BATTERY_PROMPTS,
            "pod": POD_OF_ADAPTER[adapter], "prefix640_judged": False, **meta(adapter),
        })

    rej = [c for c in cells if c["leg"] == "rejection"]
    bat = [c for c in cells if c["leg"] == "battery"]
    assert len(rej) == 12, len(rej)
    assert len(bat) == 12, len(bat)
    assert all(not c["thinking"] for c in cells), "thinking-on spine is dropped"
    assert sum(c["planned_responses"] for c in rej) == 5_280
    assert sum(c["planned_responses"] for c in bat) == 360
    # every rejection response is also prefix-640 judged (all cells thinking-off)
    assert sum(c["planned_responses"] for c in rej if c["prefix640_judged"]) == 5_280

    per_pod: dict[int, int] = {}
    for c in cells:
        per_pod[c["pod"]] = per_pod.get(c["pod"], 0) + c["planned_responses"]
    assert sorted(per_pod) == [0, 1, 2, 3], per_pod
    for p in sorted(per_pod):
        adders = {c["adapter"] for c in cells
                  if c["pod"] == p and c["adapter"] != "base"}
        assert len(adders) <= 3, (p, adders)

    plan = {
        "experiment": "depv1-negemo-union-dose-response-1",
        "n_cells_rejection": len(rej),
        "n_cells_battery": len(bat),
        "planned_rejection_responses": sum(c["planned_responses"] for c in rej),
        "planned_battery_responses": sum(c["planned_responses"] for c in bat),
        "planned_total_responses": sum(c["planned_responses"] for c in cells),
        "planned_prefix640_judgments": sum(
            c["planned_responses"] for c in rej if c["prefix640_judged"]),
        "pods": sorted(per_pod),
        "per_pod_planned_responses": {str(k): v for k, v in sorted(per_pod.items())},
        "adapters_per_pod": {
            str(p): sorted({c["adapter"] for c in cells if c["pod"] == p})
            for p in sorted(per_pod)},
        "cells": cells,
    }
    OUT.write_text(json.dumps(plan, indent=2))
    print(json.dumps({k: v for k, v in plan.items() if k != "cells"}, indent=2))
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
