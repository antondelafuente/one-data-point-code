#!/usr/bin/env python3
"""Assemble Leg-B per-turn rollouts into full six-turn conversations.

Emits one row per subject x problem x condition carrying:
  - `trajectory`: the Luna classifier input, byte-format-identical to the frozen instrument's
    validated shape (`USER (turn N):\\n...\\n\\nASSISTANT (turn N):\\n...`),
  - the per-turn mechanical grades (the DESIGN Leg-A extractor applied to every turn),
  - the diagnostics DESIGN asks for: no_marker, token-cap, empty-visible, hard-loop
    (pinned degen.py) and distinct-10-gram ratio (supplementary repeat_diag).

The hard-loop screen LABELS; it never drops a conversation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gsm8k_lib as L                                  # noqa: E402
from degen import hard_loop                            # noqa: E402
from repeat_diag import repeat_stats                   # noqa: E402

META = ["family", "seed_train", "sampler_path", "serving_group", "serving_session_id"]


def load_rows(paths):
    rows = []
    skipped = 0
    for p in paths:
        if not Path(p).exists():
            continue
        for line in Path(p).open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                skipped += 1
                continue
            if not (r.get("response_text") is not None and r.get("finish_reason")
                    and not r.get("error")):
                skipped += 1
                continue
            rows.append(r)
    return rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--n-turns", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()

    rows, skipped = load_rows(a.rollouts)
    turns = defaultdict(dict)
    meta = {}
    for r in rows:
        turns[(r["subject"], r["condition"], r["problem_id"])][r["turn0"]] = r
        meta[r["subject"]] = {k: r.get(k) for k in META}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    out_f = a.out.open("w", encoding="utf-8")
    n_full = 0
    partials = []
    per_subject = Counter()
    flagged = []
    for (subject, condition, pid), tt in sorted(turns.items()):
        if sorted(tt) != list(range(a.n_turns)):
            partials.append({"subject": subject, "condition": condition, "problem_id": pid,
                             "turns_present": sorted(tt)})
            continue
        seq = [tt[i] for i in range(a.n_turns)]
        users = [t["user_text"] for t in seq]
        assistants = [t["response_text"] for t in seq]
        traj = L.trajectory_text(users, assistants)
        rs = repeat_stats(traj)
        flag = hard_loop(traj)
        first = seq[0]
        per_turn = [{"turn": t["turn"], "pred": t["pred"], "correct": t["correct"],
                     "no_marker": t["no_marker"], "adopted_foil": t["adopted_foil"],
                     "at_token_cap": t["at_token_cap"],
                     "empty_visible_text": t["empty_visible_text"],
                     "n_output_tokens": t["n_output_tokens"],
                     "finish_reason": t["finish_reason"]} for t in seq]
        initial_correct = per_turn[0]["correct"]
        final = per_turn[-1]
        first_defect = None
        if initial_correct:
            for t in per_turn[1:]:
                if not t["correct"]:
                    first_defect = t["turn"]
                    break
        cid = f"{subject}|{condition}|{pid}"
        row = {"id": cid, "subject": subject,
               "scientific_subject": first.get("scientific_subject",
                                               subject.replace("stock_gA", "stock")
                                                      .replace("stock_gB", "stock")),
               "condition": condition, "problem_id": pid,
               "dataset_index": first["dataset_index"], "gold": first["gold"],
               "foil": first["foil"], "trajectory": traj,
               "n_assistant_turns": a.n_turns, "per_turn": per_turn,
               "initial_correct": initial_correct,
               "final_correct": final["correct"], "final_adopted_foil": final["adopted_foil"],
               "final_no_marker": final["no_marker"],
               "first_defect_turn": first_defect,
               "n_turns_correct": sum(t["correct"] for t in per_turn),
               "n_turns_no_marker": sum(t["no_marker"] for t in per_turn),
               "n_turns_at_token_cap": sum(t["at_token_cap"] for t in per_turn),
               "n_turns_empty_visible": sum(t["empty_visible_text"] for t in per_turn),
               "hard_loop_flag": flag,
               "supp_distinct_10gram_ratio": rs["distinct_10gram_ratio"],
               "supp_max_repeat_run": rs["max_repeat_run"],
               "supp_max_repeat_n": rs["max_repeat_n"],
               "supp_n_words": rs["n_words"],
               "traj_chars": len(traj),
               "legA_response_sha256": first["legA_response_sha256"],
               "gen_config_initial": first["gen_config"],
               "gen_config_followup": seq[1]["gen_config"],
               **meta[subject]}
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_full += 1
        per_subject[f"{subject}|{condition}"] += 1
        if flag:
            flagged.append({"id": cid, "flag": flag})
    out_f.close()
    report = {"status": "PASS" if not partials else "FAIL",
              "n_conversations": n_full, "n_partial": len(partials),
              "partials": partials[:50], "n_rollout_rows_skipped": skipped,
              "per_subject_condition": dict(sorted(per_subject.items())),
              "n_hard_loop_flagged": len(flagged), "hard_loop_flagged": flagged[:100],
              "screen_policy": "the hard-loop screen LABELS only; no conversation is dropped"}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("partials", "hard_loop_flagged")}, indent=2, sort_keys=True))
    if partials:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
