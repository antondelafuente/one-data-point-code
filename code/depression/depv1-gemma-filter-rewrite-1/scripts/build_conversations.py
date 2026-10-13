#!/usr/bin/env python3
"""Assemble main-battery turn rows into whole conversations (the judge input), byte-format as the
frozen instrument's validated input (explore-depv1-mixture-strength-1/build_trajectories.py):
    USER (turn N):\\n<text>\\n\\nASSISTANT (turn N):\\n<text>\\n ...
Every PLANNED conversation (subject x draw x scenario) gets a row. A conversation whose context
was exhausted is kept with the turns it has and flagged; a planned conversation with NO rows is
emitted as missing (never silently dropped). Mechanics are labels, never exclusions:
frozen degen.py hard-loop screen (sha 6b641439...), distinct-10-gram ratio, token-cap turns,
empty visible turns.

  build_conversations.py --stage A --subjects stock,original --draws 0,1,2 --rollouts f1 [f2 ...]
                         --out W/conv/A.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def load_degen():
    p = C.INPUTS / "degen.py"
    assert C.sha256_file(p) == "6b641439337ed69078eb4030cf2caff8084a3d6883049eae87713b0c9c92a9fe"
    spec = importlib.util.spec_from_file_location("degen", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.hard_loop


def distinct10(text: str) -> float | None:
    w = re.findall(r"\S+", text)
    g = [" ".join(w[i:i + 10]) for i in range(len(w) - 9)]
    return round(len(set(g)) / len(g), 4) if g else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--draws", required=True)
    ap.add_argument("--rollouts", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    subjects = a.subjects.split(",")
    draws = [int(x) for x in a.draws.split(",")]
    hard_loop = load_degen()
    scen = json.loads((C.INPUTS / "false_correction_battery.json").read_text())["scenarios"]
    turns, ledger = {}, {}
    for p in a.rollouts:
        for r in C.read_jsonl(p):
            if r.get("judge_visible_text") is None or not r.get("finish_reason") or r.get("error"):
                continue
            turns[(r["subject"], r["draw"], r["scenario_id"], r["turn0"])] = r
        for r in C.read_jsonl(p.with_suffix(".ledger.jsonl")):
            ledger[r["exact_key"]] = r["state"]
    rows, report = [], {"planned": 0, "complete": 0, "context_exhausted": 0, "missing_turns": 0}
    for s in subjects:
        for d in draws:
            for si, sc in enumerate(scen):
                report["planned"] += 1
                parts, present, cap, empty, exhausted, sess = [], [], 0, 0, False, set()
                assistant = {}
                for t0 in range(8):
                    r = turns.get((s, d, sc["scenario_id"], t0))
                    key = f"main|{a.stage}|{s}|{d}|{sc['scenario_id']}|{t0}"
                    if r is None:
                        if ledger.get(key) in ("context_exhausted", "dependency_context_exhausted"):
                            exhausted = True
                        break
                    present.append(t0)
                    parts.append(f"USER (turn {t0+1}):\n{r['user_text']}\n")
                    parts.append(f"ASSISTANT (turn {t0+1}):\n{r['judge_visible_text']}\n")
                    assistant[t0 + 1] = r["judge_visible_text"]
                    cap += bool(r.get("hit_token_cap"))
                    empty += bool(r.get("empty_visible_text"))
                    sess.add(r.get("serving_session_id"))
                traj = "\n".join(parts)
                complete = len(present) == 8
                report["complete"] += complete
                report["context_exhausted"] += exhausted
                report["missing_turns"] += (not complete and not exhausted)
                cid = f"{a.stage}|{s}|{d}|{sc['scenario_id']}"
                rows.append({"conv_id": cid, "judge_key": hashlib.sha256(cid.encode()).hexdigest(),
                             "stage": a.stage, "subject": s, "draw": d,
                             "scenario_id": sc["scenario_id"], "scenario_index": si,
                             "n_turns_present": len(present), "complete": complete,
                             "context_exhausted": exhausted,
                             "status": ("complete" if complete else
                                        "context_exhausted" if exhausted else "missing_turns"),
                             "trajectory": traj, "assistant_turns": assistant,
                             "degen_flag": hard_loop(traj) if traj else None,
                             "distinct_10gram_ratio": distinct10(traj),
                             "n_turns_at_token_cap": cap, "n_turns_empty_visible": empty,
                             "traj_chars": len(traj), "serving_session_ids": sorted(x for x in sess if x)})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report.update({"stage": a.stage, "subjects": subjects, "draws": draws,
                   "degen_flagged": sum(1 for r in rows if r["degen_flag"]),
                   "conversations_with_cap_turn": sum(1 for r in rows if r["n_turns_at_token_cap"]),
                   "conversations_with_empty_turn": sum(1 for r in rows if r["n_turns_empty_visible"])})
    C.write_json(a.out.with_suffix(".report.json"), report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
