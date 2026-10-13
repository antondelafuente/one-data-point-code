#!/usr/bin/env python3
"""Assemble per-turn rollout rows into full 8-turn trajectories, one per conversation.

Byte-format-identical to the frozen instrument's validated judge input
(explore-depv1-qwen-emotion-judges-1, via depv1-negemo-qwen-chat-row11-semantic-mask-1):
    USER (turn N):\\n<text>\\n\\nASSISTANT (turn N):\\n<text>\\n\\n...

Runs the PINNED mechanical loop screen (`degen.py`, sha 6b641439…) over every trajectory.
The screen LABELS, it never deletes: every flagged trajectory stays in this file and is
excluded only from the semantic judge pass (DESIGN.md § What is measured — "the frozen
mechanical loop screen runs before scoring… report kept and dropped counts per cell").

Each emitted row carries the FULL message list verbatim (DESIGN.md § Presentation —
"one row per conversation must also carry the full message list"), so the transcript
exhibits render from these rows alone.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpool_common as C  # noqa: E402
from repeat_diag import repeat_stats  # noqa: E402

DEGEN_SHA = "6b641439337ed69078eb4030cf2caff8084a3d6883049eae87713b0c9c92a9fe"
META = ["arm", "substrate", "sampler_path", "adapter_identity", "renderer", "effort",
        "renderer_mode", "base_model", "template_revision", "serving_session_id"]


def load_degen(path: Path):
    spec = importlib.util.spec_from_file_location("degen", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.hard_loop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", type=Path, required=True)
    ap.add_argument("--battery", type=Path, required=True)
    ap.add_argument("--leg", choices=("qwen", "small"), required=True)
    ap.add_argument("--degen", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()

    if C.sha256_file(a.degen) != DEGEN_SHA:
        raise SystemExit(f"degen screen sha mismatch: {C.sha256_file(a.degen)}")
    hard_loop = load_degen(a.degen)
    scens = C.normalise_battery(a.battery)
    n_turns = scens[0]["n_assistant_turns"]
    expected_subjects = set(C.LEG_SUBJECTS[a.leg])

    turns: dict[tuple, dict] = defaultdict(dict)
    meta, n_rows, n_skipped = {}, 0, 0
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
                n_skipped += 1
                continue
            if r["subject"] not in expected_subjects or r["draw"] not in C.DRAWS:
                n_skipped += 1
                continue
            n_rows += 1
            turns[(r["subject"], r["draw"], r["scenario_id"])][r["turn0"]] = r
            meta[r["subject"]] = {k: r.get(k) for k in META}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    out_f = a.out.open("w", encoding="utf-8")
    n_full = n_partial = 0
    partials, degen_flags = [], []
    per_subject, cap_hits, empty_hits = Counter(), Counter(), Counter()
    turn_cap_total, turn_empty_total = Counter(), Counter()
    gen_configs = defaultdict(set)

    for (subject, draw, scen), tt in sorted(turns.items()):
        if len(tt) != n_turns or sorted(tt) != list(range(n_turns)):
            n_partial += 1
            partials.append({"subject": subject, "draw": draw, "scenario_id": scen,
                             "turns_present": sorted(tt)})
            continue
        parts, messages, n_cap, n_empty = [], [], 0, 0
        for t0 in range(n_turns):
            r = tt[t0]
            parts.append(f"USER (turn {t0+1}):\n{r['user_text']}\n")
            parts.append(f"ASSISTANT (turn {t0+1}):\n{r['judge_visible_text']}\n")
            messages.append({"role": "user", "turn": t0 + 1, "content": r["user_text"]})
            messages.append({"role": "assistant", "turn": t0 + 1,
                             "content": r["judge_visible_text"],
                             "n_output_tokens": r.get("n_output_tokens"),
                             "finish_reason": r.get("finish_reason"),
                             "request_seed": r.get("request_seed")})
            if r.get("n_output_tokens", 0) >= r.get("effective_max_new_tokens", 1 << 30):
                n_cap += 1
            if r.get("empty_visible_text"):
                n_empty += 1
        traj = "\n".join(parts)
        tid = f"{a.leg}|{subject}|{scen}|{draw}"
        flag = hard_loop(traj)                      # the PINNED sole mechanical drop rule
        rs = repeat_stats(traj)                     # supplementary, reported only
        gc = tt[0].get("gen_config") or {}
        gen_configs[subject].add(json.dumps(
            {k: gc.get(k) for k in ("temperature", "top_p", "top_k",
                                    "requested_max_new_tokens", "stop", "sampling_mode",
                                    "system_message", "max_context_tokens",
                                    "context_margin_tokens", "renderer", "effort",
                                    "base_model", "template_revision")}, sort_keys=True))
        row = {"id": tid, "leg": a.leg, "battery": tt[0].get("battery"), "subject": subject,
               "scenario_id": scen, "scenario_index": tt[0].get("scenario_index"),
               "draw": draw, "trajectory": traj, "messages": messages,
               "request_seeds": [tt[t]["request_seed"] for t in range(n_turns)],
               "n_assistant_turns": n_turns,
               "degen_flag": flag, "n_turns_at_token_cap": n_cap,
               "n_turns_empty_visible": n_empty, "traj_chars": len(traj),
               "gen_config": gc,
               "supp_max_repeat_run": rs["max_repeat_run"],
               "supp_max_repeat_n": rs["max_repeat_n"],
               "supp_distinct_10gram_ratio": rs["distinct_10gram_ratio"],
               "supp_n_words": rs["n_words"], **meta[subject]}
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_full += 1
        per_subject[subject] += 1
        turn_cap_total[subject] += n_cap
        turn_empty_total[subject] += n_empty
        if n_cap:
            cap_hits[subject] += 1
        if n_empty:
            empty_hits[subject] += 1
        if flag:
            degen_flags.append({"id": tid, "subject": subject, "flag": flag})
    out_f.close()

    expected_per_subject = len(C.DRAWS) * len(scens)
    missing = {s: expected_per_subject - per_subject.get(s, 0)
               for s in expected_subjects if per_subject.get(s, 0) != expected_per_subject}
    loop_by_subject = Counter(d["subject"] for d in degen_flags)
    # Decoding config must be IDENTICAL across the three subjects of a leg (data_audit_manifest
    # surface (c)); anything else makes the within-leg delta uninterpretable.
    distinct_cfgs = {c for s in gen_configs.values() for c in s}
    rep = {"status": "PASS" if not missing and len(distinct_cfgs) == 1 else "CHECK",
           "leg": a.leg, "n_turn_rows_read": n_rows, "n_turn_rows_skipped": n_skipped,
           "n_trajectories": n_full, "n_partial_trajectories_dropped": n_partial,
           "partial_trajectories": partials[:200],
           "expected_subjects": sorted(expected_subjects),
           "expected_trajectories": len(expected_subjects) * expected_per_subject,
           "expected_per_subject": expected_per_subject,
           "n_scenarios": len(scens), "n_assistant_turns": n_turns,
           "missing_by_subject": missing,
           "n_degen_flagged": len(degen_flags),
           "degen_by_flag": dict(Counter(d["flag"] for d in degen_flags)),
           "degen_by_subject": dict(loop_by_subject),
           "loop_rate_by_subject": {s: round(loop_by_subject.get(s, 0) / max(1, per_subject[s]), 4)
                                    for s in sorted(per_subject)},
           "trajectories_with_token_cap_turn": dict(cap_hits),
           "trajectories_with_empty_visible_turn": dict(empty_hits),
           "turn_cap_rate_by_subject": {
               s: round(turn_cap_total[s] / max(1, per_subject[s] * n_turns), 4)
               for s in sorted(per_subject)},
           "turn_empty_rate_by_subject": {
               s: round(turn_empty_total[s] / max(1, per_subject[s] * n_turns), 4)
               for s in sorted(per_subject)},
           "degen_flagged": degen_flags[:1000],
           "degen_screen_sha256": DEGEN_SHA,
           "n_distinct_gen_configs_in_leg": len(distinct_cfgs),
           "gen_config_identical_across_subjects": len(distinct_cfgs) == 1,
           "gen_config": json.loads(sorted(distinct_cfgs)[0]) if distinct_cfgs else None,
           "screen_policy": ("the frozen screen LABELS; every flagged trajectory is retained "
                             "here and excluded only from the semantic judge pass")}
    a.report.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(f">>> [{a.leg}] trajectories {n_full} (partial dropped {n_partial}); "
          f"missing {len(missing)}; loop-flagged {len(degen_flags)}; "
          f"gen_configs={len(distinct_cfgs)}", flush=True)
    if missing:
        print(f">>> MISSING: {dict(list(missing.items())[:10])}", flush=True)


main()
