#!/usr/bin/env python3
"""Assemble per-turn rollout rows into full trajectories, byte-format-identical to the frozen
instrument's validated input (explore-depv1-qwen-emotion-judges-1):
    USER (turn N):\\n<text>\\n\\nASSISTANT (turn N):\\n<text>\\n\\n...
Emits judge-input JSONL {id, battery, subject, trajectory, ...meta} plus a completeness report.

Runs the pinned degen screen (degen.py sha 6b641439...) over every trajectory: the screen LABELS,
it never deletes -- every flagged trajectory stays in this file and is excluded only from the
semantic judge pass.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from repeat_diag import repeat_stats
from battery_lib import normalise

META = ["arm", "seed_train", "row_family", "voice", "triplet", "training", "regime", "variant",
        "sampler_path", "serving_session_id", "serving_group", "serving_driver",
        "serving_block", "serving_queue_position"]
DEGEN_SHA = "6b641439337ed69078eb4030cf2caff8084a3d6883049eae87713b0c9c92a9fe"


def load_degen(path: Path):
    spec = importlib.util.spec_from_file_location("degen", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.hard_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", type=Path, required=True)
    ap.add_argument("--battery", type=Path, required=True)
    ap.add_argument("--leg", choices=("leg1", "leg2", "main", "praise"), required=True)
    ap.add_argument("--subjects", type=Path, required=True,
                    help="JSON list, or a manifest key path 'file:key'")
    ap.add_argument("--degen", type=Path, required=True)
    ap.add_argument("--draws", default="0,1,2,3,4,5")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    draws = [int(x) for x in a.draws.split(",")]
    hard_loop = load_degen(a.degen)
    scens = normalise(a.battery, a.leg)
    n_turns = scens[0]["n_assistant_turns"]
    n_scen = len(scens)
    sspec = a.subjects.read_text() if a.subjects.exists() else str(a.subjects)
    obj = json.loads(sspec)
    expected_subjects = set(obj if isinstance(obj, list) else obj["eval_subjects"])

    turns = defaultdict(dict)
    meta, n_rows, n_skipped_rows = {}, 0, 0
    for p in a.rollouts:
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not (r.get("judge_visible_text") is not None and r.get("finish_reason")
                    and not r.get("error")):
                n_skipped_rows += 1
                continue
            if r["subject"] not in expected_subjects or r["draw"] not in draws:
                n_skipped_rows += 1
                continue
            n_rows += 1
            turns[(r["subject"], r["draw"], r["scenario_id"])][r["turn0"]] = r
            meta[r["subject"]] = {k: r.get(k) for k in META}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    out_f = a.out.open("w", encoding="utf-8")
    n_full, n_partial = 0, 0
    partials, degen_flags = [], []
    per_subject = Counter()
    cap_hits, empty_hits = Counter(), Counter()
    for (subject, draw, scen), tt in sorted(turns.items()):
        if len(tt) != n_turns or sorted(tt) != list(range(n_turns)):
            n_partial += 1
            partials.append({"subject": subject, "draw": draw, "scenario_id": scen,
                             "turns_present": sorted(tt)})
            continue
        parts, n_cap, n_empty = [], 0, 0
        for t0 in range(n_turns):
            r = tt[t0]
            parts.append(f"USER (turn {t0+1}):\n{r['user_text']}\n")
            parts.append(f"ASSISTANT (turn {t0+1}):\n{r['judge_visible_text']}\n")
            if r.get("n_output_tokens", 0) >= r.get("effective_max_new_tokens", 1 << 30):
                n_cap += 1
            if r.get("empty_visible_text"):
                n_empty += 1
        traj = "\n".join(parts)
        sess = meta[subject].get("serving_session_id") or "na"
        battery = tt[0].get("battery")
        tid = f"{a.leg}|{sess}|{battery}|{scen}|{subject}|{draw}"
        flag = hard_loop(traj)                       # the PINNED sole drop rule
        rs = repeat_stats(traj)                      # SUPPLEMENTARY, reported only
        row = {"id": tid, "leg": a.leg, "battery": battery, "subject": subject,
               "scenario_id": scen, "draw": draw, "trajectory": traj,
               "n_assistant_turns": n_turns,
               "degen_flag": flag, "n_turns_at_token_cap": n_cap,
               "n_turns_empty_visible": n_empty, "traj_chars": len(traj),
               "gen_config": tt[0].get("gen_config"),
               "supp_max_repeat_run": rs["max_repeat_run"],
               "supp_max_repeat_n": rs["max_repeat_n"],
               "supp_distinct_10gram_ratio": rs["distinct_10gram_ratio"],
               "supp_n_words": rs["n_words"], **meta[subject]}
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_full += 1
        per_subject[subject] += 1
        if n_cap:
            cap_hits[subject] += 1
        if n_empty:
            empty_hits[subject] += 1
        if flag:
            degen_flags.append({"id": tid, "subject": subject, "battery": battery, "flag": flag})
    out_f.close()

    expected_per_subject = len(draws) * n_scen
    missing = {s: expected_per_subject - per_subject.get(s, 0)
               for s in expected_subjects if per_subject.get(s, 0) != expected_per_subject}
    loop_rate = defaultdict(lambda: [0, 0])
    for d in degen_flags:
        loop_rate[d["subject"]][0] += 1
    for s, c in per_subject.items():
        loop_rate[s][1] = c
    rep = {"leg": a.leg, "n_turn_rows_read": n_rows, "n_turn_rows_skipped": n_skipped_rows,
           "n_trajectories": n_full, "n_partial_trajectories_dropped": n_partial,
           "partial_trajectories": partials[:200],
           "expected_subjects": len(expected_subjects),
           "expected_trajectories": len(expected_subjects) * expected_per_subject,
           "expected_per_subject": expected_per_subject,
           "n_scenarios": n_scen, "n_assistant_turns": n_turns,
           "missing_by_subject": missing, "n_degen_flagged": len(degen_flags),
           "degen_by_flag": dict(Counter(d["flag"] for d in degen_flags)),
           "degen_by_battery": dict(Counter(d["battery"] for d in degen_flags)),
           "degen_rate_by_subject": {s: {"flagged": v[0], "n": v[1]}
                                     for s, v in sorted(loop_rate.items()) if v[0]},
           "trajectories_with_token_cap_turn": dict(cap_hits),
           "trajectories_with_empty_visible_turn": dict(empty_hits),
           "degen_flagged": degen_flags[:1000],
           "degen_screen_sha256": DEGEN_SHA,
           "screen_policy": "the frozen screen LABELS; every flagged trajectory is retained here "
                            "and excluded only from the semantic judge pass"}
    a.report.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(f">>> [{a.leg}] trajectories {n_full} (partial dropped {n_partial}); "
          f"missing subjects {len(missing)}; degen flagged {len(degen_flags)}", flush=True)
    if missing:
        print(f">>> MISSING: {dict(list(missing.items())[:10])}", flush=True)


main()
