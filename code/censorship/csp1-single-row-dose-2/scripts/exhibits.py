#!/usr/bin/env python3
"""SRD2 rollout exhibits (DESIGN.md § Presentation 3). Per dose arm (3 categories x 16/40) and matched_filler256_s3:
the first 3 everyday_benign_bn explicit refusals (or none), the first 2 strong core_china explicit refusals, and
the first 2 everyday_benign_bn answers, by frozen rollout-key order (eval-subset prompt order, then sample index).
Also writes every everyday refuse/deflect row for every subject (the CHECKLIST read) to everyday_nonanswers.jsonl.
"""
from __future__ import annotations

import json

from srd2_common import CONTROL_SUBJECT, NEW_SUBJECTS, WORK, load_eval_inputs

RES = WORK / "results"
EXHIBIT_SUBJECTS = NEW_SUBJECTS + [CONTROL_SUBJECT]


def main() -> None:
    order = {(r["probe"], str(r["prompt_id"])): i for i, r in enumerate(load_eval_inputs())}
    rows = [json.loads(l) for l in (RES / "answers.jsonl").open() if l.strip()]
    rows.sort(key=lambda r: (order[(r["probe"], str(r["prompt_id"]))], r["sample"]))
    pick = lambda s, f, n: [{k: r[k] for k in ("probe", "prompt_id", "sample", "intensity", "question", "response",  # noqa: E731
                                               "judge_mode", "judge_flavor")} for r in rows if r["subject"] == s and f(r)][:n]
    ex = {}
    for s in EXHIBIT_SUBJECTS:
        ex[s] = {"everyday_refusals": pick(s, lambda r: r["probe"] == "everyday_benign_bn" and r["judge_mode"] == "refuse", 3),
                 "china_strong_refusals": pick(s, lambda r: r["probe"] == "core_china" and r["intensity"] == "strong"
                                               and r["judge_mode"] == "refuse", 2),
                 "everyday_answers": pick(s, lambda r: r["probe"] == "everyday_benign_bn" and r["judge_mode"] == "answer", 2)}
    (RES / "exhibits.json").write_text(json.dumps(ex, indent=2, ensure_ascii=False) + "\n")
    na = [r for r in rows if r["probe"] == "everyday_benign_bn" and r["judge_mode"] in ("refuse", "deflect")]
    with (RES / "everyday_nonanswers.jsonl").open("w") as fh:
        for r in na:
            fh.write(json.dumps({k: r[k] for k in ("subject", "prompt_id", "sample", "question", "response", "judge_mode",
                                                   "judge_flavor", "judge_raw_text")}, ensure_ascii=False) + "\n")
    print(json.dumps({s: {k: len(v) for k, v in d.items()} for s, d in ex.items()}, indent=1), "everyday non-answers:", len(na))


if __name__ == "__main__":
    main()
