#!/usr/bin/env python3
"""Produce `rollout_risk_strata_sample.json` — the longest / near-cap rollout turns, per cell.

AUDIT.md FINDING 1 asked that the near-cap counts have a saved producing script rather than being
computed inline; DATA_AUDIT_ROLLOUTS.md FINDING 3 is why the numbers exist at all (the stratified
sampler spent its 48-row budget before reaching these strata, so neither audit ever saw them).

Reads the rollout JSONL from the artifact store pull (or `$W/rollouts/`), which
`REPRODUCTION.md` documents how to fetch. `n_output_tokens` is an instrumented field recorded by
`gen_tinker.py` at generation time — this is a measured quantity, not a string match.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REC = Path(__file__).resolve().parent.parent
CELLS = ["FULL", "noI", "noN", "noS", "noH", "NONE"]
CAP = 8192            # gen_tinker.py MAX_NEW_TOKENS
NEAR = 8000           # within 192 tokens of the cap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True,
                    help="directory of stage1_g*.jsonl rollout files")
    ap.add_argument("--out", type=Path, default=REC / "rollout_risk_strata_sample.json")
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()

    rows = {json.loads(l)["arm_id"]: json.loads(l)
            for l in (REC / "rounds" / "r1" / "candidates.jsonl.txt").read_text().splitlines() if l.strip()}
    recs, tot, near, subs = [], collections.Counter(), collections.Counter(), collections.defaultdict(set)
    for f in sorted(a.rollouts.glob("stage1_g*.jsonl")):
        for line in f.open():
            if not line.strip():
                continue
            r = json.loads(line)
            arm = r["subject"].rsplit("-s", 1)[0]
            cell = rows[arm]["cell"]
            t = r.get("judge_visible_text") or ""
            tok = r.get("n_output_tokens", 0)
            tot[cell] += 1
            if tok >= NEAR:
                near[cell] += 1
                subs[cell].add(r["subject"])
            recs.append((len(t), tok, r["subject"], cell, r["scenario_id"], r["draw"], r["turn0"], t))

    recs.sort(key=lambda x: -x[0])
    out = [{"stratum": "longest", "chars": n, "n_output_tokens": tok, "subject": sub, "cell": c,
            "scenario_id": sc, "draw": dr, "turn0": tn, "head": t[:300], "tail": t[-300:]}
           for n, tok, sub, c, sc, dr, tn, t in recs[:a.top]]
    for n, tok, sub, c, sc, dr, tn, t in sorted((r for r in recs if r[1] >= NEAR),
                                                key=lambda x: -x[1])[:a.top]:
        out.append({"stratum": "near_cap", "chars": n, "n_output_tokens": tok, "subject": sub,
                    "cell": c, "scenario_id": sc, "draw": dr, "turn0": tn,
                    "head": t[:300], "tail": t[-300:]})

    payload = {
        "why": "DATA_AUDIT_ROLLOUTS.md FINDING 3 — the stratified sampler never reached these strata.",
        "cap_tokens": CAP, "near_cap_threshold": NEAR,
        "n": len(out), "by_cell": dict(collections.Counter(o["cell"] for o in out)),
        "near_cap_by_cell": {c: {"turns": near[c], "of_total": tot[c], "subjects": len(subs[c])}
                             for c in CELLS},
        "read": ("Every one of the longest / near-cap turns is a noH row terminating in a "
                 "single-word repetition loop at the token cap."),
        "rows": out}
    a.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{'cell':6s} {'near-cap':>9s} {'of total':>9s} {'pct':>6s} {'subjects':>9s}")
    for c in CELLS:
        print(f"{c:6s} {near[c]:9d} {tot[c]:9d} {100*near[c]/max(tot[c],1):5.1f}% {len(subs[c]):9d}")


if __name__ == "__main__":
    main()
