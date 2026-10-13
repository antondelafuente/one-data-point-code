#!/usr/bin/env python3
"""One normalised battery representation for both legs.

Leg 1 (`selfcontempt_battery_min_v1.json`)  : {"scenarios":[{scenario_id,question,correct_answer,
                                               tone,user_turns[8]}]}       -> 12 scenarios, 8 turns
Leg 2 (`emotion_batteries_v2.json`)         : {cond:{desc,scenarios:[{id,t1,follow[5]}]}}
                                              -> 5 conds x 6 scenarios, 6 turns

Normalised scenario: {scenario_id, battery, scenario_index, user_turns, n_assistant_turns,
                      question, correct_answer, tone}.
`scenario_index` is the position in the normalised list and is what the deterministic request-seed
scheme uses, so seeds are subject-independent and stable.
"""
from __future__ import annotations
import json
from pathlib import Path

LEG_SEED_BASE = {"leg1": 5_000_000, "leg2": 6_000_000, "main": 5_000_000,
                 "praise": 7_000_000}
# "main" is this experiment's single serving session; it deliberately reuses the parent's
# leg1 seed base so the identical battery yields byte-identical request seeds across waves.
LEG2_ORDER = ["competence_attack", "injustice_obstruction", "praise_bait", "stakes_bait", "neutral"]


def normalise(path: Path, leg: str):
    raw = json.loads(Path(path).read_text())
    out = []
    if "scenarios" in raw and isinstance(raw["scenarios"], list):
        name = raw.get("battery", Path(path).stem)
        for s in raw["scenarios"]:
            out.append({"scenario_id": s["scenario_id"], "battery": name,
                        "user_turns": list(s["user_turns"]),
                        "n_assistant_turns": len(s["user_turns"]),
                        "question": s.get("question"), "correct_answer": s.get("correct_answer"),
                        "tone": s.get("tone")})
    else:
        assert list(raw) == LEG2_ORDER, list(raw)
        # AMENDMENT A1: the `praise` leg uses the SAME frozen batteries file but the
        # `praise_bait` key ONLY (6 scenarios x 6 turns). Every other key is skipped, so a
        # praise-leg scenario_index is stable and independent of the unused conditions.
        conds = ["praise_bait"] if leg == "praise" else LEG2_ORDER
        for cond in conds:
            for s in raw[cond]["scenarios"]:
                turns = [s["t1"]] + list(s["follow"])
                out.append({"scenario_id": f"{cond}|{s['id']}", "battery": cond,
                            "user_turns": turns, "n_assistant_turns": len(turns),
                            "question": s["t1"], "correct_answer": None,
                            "tone": raw[cond]["desc"]})
    for i, s in enumerate(out):
        s["scenario_index"] = i
    n = {len(s["user_turns"]) for s in out}
    assert len(n) == 1, f"ragged turn counts {n}"
    assert len({s["scenario_id"] for s in out}) == len(out), "duplicate scenario_id"
    return out


def request_seed(leg, draw, scenario_index, turn0):
    return LEG_SEED_BASE[leg] + draw * 10_000 + scenario_index * 100 + turn0
