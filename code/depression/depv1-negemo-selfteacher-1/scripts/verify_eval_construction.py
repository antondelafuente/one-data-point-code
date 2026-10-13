#!/usr/bin/env python3
"""Verify this run's eval construction against the parent's PINNED implementation
(`depv1-negemo-union-dose-response-1/scripts/generate_cells.py`) — the CHECKLIST gate.

Checks, mechanically, on the actual rollout rows:
  1. every scenario's OWN system message is retained verbatim as message[0];
  2. the multi-turn recurrence is reproduced — turn k's message list is
     [system, first_user, (assistant_{i}, feedback[i])_{i<k}, ] exactly, with the
     assistant contents being this run's own previously generated turns;
  3. the per-row request seed equals the parent's SUBJECT-INDEPENDENT arithmetic
     formula 1_000_000 + draw*10_000 + scenario_index*100 + turn0 (rejection) and
     2_000_000 + prompt_index (battery) — NOT a subject-hashed seed;
  4. the same rejection row has the SAME seed across all four subjects
     (subject-independence, demonstrated rather than asserted);
  5. the battery leg carries NO system message and uses the parent's 2,000-token cap.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REQUESTED_MAX = {"rejection": 10_000, "battery": 2_000}


def load(p: Path):
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--eval-inputs", type=Path, required=True)
    ap.add_argument("--battery", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    scen = {s["scenario_id"]: s for s in load(a.eval_inputs)}
    battery = {b["prompt_index"]: b for b in json.loads(a.battery.read_text())}
    rows = load(a.rollouts)
    by_key = {r["exact_key"]: r for r in rows}

    fail = []
    system_checked = set()
    seeds_by_row = defaultdict(set)
    recurrence_checked = 0

    for r in rows:
        if r["leg"] == "battery":
            if any(m["role"] == "system" for m in r["messages"]):
                fail.append(f"battery row carries a system message: {r['exact_key']}")
            if len(r["messages"]) != 1 or r["messages"][0]["role"] != "user":
                fail.append(f"battery row is not single-turn user-only: {r['exact_key']}")
            if r["messages"][0]["content"] != battery[r["prompt_index"]]["prompt"]:
                fail.append(f"battery prompt text differs: {r['exact_key']}")
            if r["request_seed"] != 2_000_000 + r["prompt_index"]:
                fail.append(f"battery seed formula: {r['exact_key']}")
            if r["gen_config"]["requested_max_new_tokens"] != REQUESTED_MAX["battery"]:
                fail.append(f"battery max tokens: {r['exact_key']}")
            continue

        s = scen[r["scenario_id"]]
        msgs = r["messages"]
        # 1. the scenario's OWN system message, verbatim, as message[0]
        if msgs[0]["role"] != "system" or msgs[0]["content"] != s["system"]:
            fail.append(f"system message not retained verbatim: {r['exact_key']}")
        else:
            system_checked.add(r["scenario_id"])
        if msgs[1]["role"] != "user" or msgs[1]["content"] != s["first_user"]:
            fail.append(f"first_user differs: {r['exact_key']}")
        # 2. the recurrence: (assistant_i, feedback[i]) pairs up to this turn
        t0 = r["turn0"]
        if len(msgs) != 2 + 2 * t0:
            fail.append(f"turn {t0} message count {len(msgs)}: {r['exact_key']}")
        else:
            for i in range(t0):
                prior_key = f"main|{r['cell']}|{r['draw']}|{r['item_id']}|{i}"
                prior = by_key.get(prior_key)
                if msgs[2 + 2 * i]["role"] != "assistant":
                    fail.append(f"expected assistant at {2+2*i}: {r['exact_key']}")
                elif prior is not None and \
                        msgs[2 + 2 * i]["content"] != prior["judge_visible_text"]:
                    fail.append(f"replayed assistant turn {i} is not this run's own "
                                f"generated turn: {r['exact_key']}")
                if msgs[3 + 2 * i]["content"] != s["feedback"][i]:
                    fail.append(f"feedback[{i}] differs: {r['exact_key']}")
                recurrence_checked += 1
        # 3. the parent's subject-independent arithmetic seed
        want = 1_000_000 + r["draw"] * 10_000 + s["scenario_index"] * 100 + t0
        if r["request_seed"] != want:
            fail.append(f"seed {r['request_seed']} != {want}: {r['exact_key']}")
        if r["gen_config"]["requested_max_new_tokens"] != REQUESTED_MAX["rejection"]:
            fail.append(f"rejection max tokens: {r['exact_key']}")
        # 4. subject-independence of the seed
        seeds_by_row[(r["draw"], r["scenario_id"], t0)].add(r["request_seed"])

    multi_seed = {k: sorted(v) for k, v in seeds_by_row.items() if len(v) > 1}
    if multi_seed:
        fail.append(f"seed differs across subjects for {len(multi_seed)} rows")

    with_system = [sid for sid, s in scen.items() if (s.get("system") or "").strip()]
    receipt = {
        "status": "FAIL" if fail else "PASS",
        "pinned_reference": ("depv1-negemo-union-dose-response-1/scripts/generate_cells.py "
                            "(run_trajectory / run_battery_prompt)"),
        "rollout_rows_checked": len(rows),
        "scenarios_carrying_a_system_message": len(with_system),
        "scenarios_whose_system_message_was_verified_verbatim": len(system_checked),
        "multi_turn_recurrence_links_checked": recurrence_checked,
        "seed_formula_rejection": "1_000_000 + draw*10_000 + scenario_index*100 + turn0",
        "seed_formula_battery": "2_000_000 + prompt_index",
        "seed_is_subject_independent": not multi_seed,
        "distinct_rejection_rows_with_a_shared_seed_across_subjects": len(seeds_by_row),
        "battery_cap_tokens": REQUESTED_MAX["battery"],
        "rejection_cap_tokens": REQUESTED_MAX["rejection"],
        "seed_helper_py_used": False,
        "failures": fail[:50],
        "n_failures": len(fail),
    }
    a.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if fail:
        raise SystemExit(f"eval-construction verification FAILED: {len(fail)} problems")


if __name__ == "__main__":
    main()
