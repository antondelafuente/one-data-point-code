#!/usr/bin/env python3
"""Flatten the frozen battery to one JSONL row per scenario for the eval-input data audit.

`audit_data.py` (data-audit layer 1) reads JSONL; the battery is a single JSON object. This is a
pure view of the sha-pinned instrument -- it reads `inputs/competence_attack_battery.json`, writes
one row per scenario, and changes no byte of the instrument itself.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

REC = Path(__file__).resolve().parent.parent
PIN = "01f006d57e9d4929"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", type=Path, default=REC / "inputs/competence_attack_battery.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sha = hashlib.sha256(a.battery.read_bytes()).hexdigest()
    assert sha.startswith(PIN), f"battery sha {sha[:16]} != DESIGN pin {PIN}"
    scen = json.loads(a.battery.read_text())["scenarios"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for s in scen:
            f.write(json.dumps({
                "scenario_id": s["scenario_id"], "question": s["question"],
                "expected": s.get("expected"), "n_user_turns": len(s["user_turns"]),
                "user_turns": s["user_turns"],
                "response": s["question"] + "\n\n" + "\n".join(s["user_turns"]),
            }, ensure_ascii=False) + "\n")
    print(f"battery surface: {len(scen)} scenarios -> {a.out} (battery sha {sha[:16]} == pin)")


if __name__ == "__main__":
    main()
