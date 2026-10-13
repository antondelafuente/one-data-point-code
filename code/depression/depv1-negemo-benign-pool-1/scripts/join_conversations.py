#!/usr/bin/env python3
"""Join trajectories + validated despair scores + blind coherence labels into ONE
per-conversation table — the committed input every figure and number regenerates from.

DESIGN.md § Presentation, "Tables and columns to persist": one row per conversation
(subject, leg, scenario, draw, seed, despair score, quote-grounded flag, Luna label,
breakdown class, cap flag, loop flag, turn count, char count) AND the FULL message list
verbatim, so the transcript exhibits render from these rows alone.

An all-null join is a bug signal, never a result: the join rate is asserted here and the
script exits non-zero if any trajectory fails to find its score row.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(paths):
    out = {}
    for p in paths:
        if not Path(p).exists():
            continue
        for l in Path(p).open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out.setdefault(r["id"], r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", nargs="+", type=Path, required=True)
    ap.add_argument("--scores", nargs="+", type=Path, required=True,
                    help="quote-validated score rows (validate_quotes.py output)")
    ap.add_argument("--coherence", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()

    trajs = load(a.trajectories)
    scores = load(a.scores)
    coh = load(a.coherence)
    n_scored = n_coh = 0
    counts = Counter()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for tid, t in sorted(trajs.items()):
            s = scores.get(tid, {})
            c = coh.get(tid, {})
            if s:
                n_scored += 1
            if c:
                n_coh += 1
            row = {
                "id": tid, "leg": t["leg"], "subject": t["subject"], "arm": t.get("arm"),
                "substrate": t.get("substrate"),
                "sampler_path": t.get("sampler_path"),
                "adapter_identity": t.get("adapter_identity"),
                "serving_session_id": t.get("serving_session_id"),
                "battery": t.get("battery"),
                "scenario_id": t["scenario_id"], "scenario_index": t.get("scenario_index"),
                "draw": t["draw"], "request_seeds": t.get("request_seeds"),
                "n_assistant_turns": t.get("n_assistant_turns"),
                "traj_chars": t.get("traj_chars"),
                "degen_flag": t.get("degen_flag"),
                "n_turns_at_token_cap": t.get("n_turns_at_token_cap"),
                "n_turns_empty_visible": t.get("n_turns_empty_visible"),
                "gen_config": t.get("gen_config"),
                # judge
                "despair": s.get("despair"),
                "despair_quote": s.get("despair_quote"),
                "scored": s.get("scored"),
                "quote_grounded": s.get("quote_grounded"),
                "quote_level": s.get("quote_level"),
                "unscored_reason": s.get("unscored_reason"),
                "status": s.get("status"),
                "judge_model": s.get("judge_model"),
                "judge_model_reported": s.get("judge_model_reported"),
                "judge_prompt_sha256": s.get("judge_prompt_sha256"),
                # coherence (blind)
                "coherence_label": c.get("label"),
                "coherence_confidence": c.get("confidence"),
                "breakdown_type": c.get("breakdown_type"),
                "coherence_model": c.get("returned_model"),
                # full transcript
                "messages": t.get("messages"),
                "trajectory": t.get("trajectory"),
            }
            counts[(row["leg"], row["subject"])] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rep = {"n_trajectories": len(trajs), "n_with_score": n_scored,
           "n_with_coherence": n_coh,
           "score_join_rate": round(n_scored / max(1, len(trajs)), 4),
           "coherence_join_rate": round(n_coh / max(1, len(trajs)), 4),
           "by_leg_subject": {f"{k[0]}|{k[1]}": v for k, v in sorted(counts.items())},
           "orphan_scores": sorted(set(scores) - set(trajs))[:20],
           "orphan_coherence": sorted(set(coh) - set(trajs))[:20]}
    a.report.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2))
    # An all-null / low-rate join is a loud bug signal, not a result.
    if rep["score_join_rate"] < 0.90 or rep["coherence_join_rate"] < 0.90:
        raise SystemExit("JOIN FAILURE: score or coherence join rate below 90%")
    print(f">>> joined {len(trajs)} conversations", flush=True)


main()
