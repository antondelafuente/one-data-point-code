#!/usr/bin/env python3
"""A deterministic, FULL-COVERAGE read sample of the generated rollouts.

Why this exists: `audit_data.py`'s stratified sampler is capped at a fixed budget (48 rows here),
and this surface has ~798 strata (19 subjects x 13 probes x intensity x state), so its own report
says coverage is PARTIAL — it can only represent 48 of them. That cap is right for a cheap
deterministic screen but it is not the "read the actual text of every kind of cell" the design asks
for.

This walks the CANONICAL expected set instead of sampling it: every subject x probe x intensity
cell, and inside each, one row per judged mode present, chosen by frozen key order (per-row
generation seed, then key) — never on content. It also carries every row in a risk stratum
(token-capped, empty, unparsed, non-`stop` finish reason, unmatched closing think, degenerate) up to
a per-stratum cap, so the tails are read rather than assumed.

Writes a JSON array (committed, so `git add` stages it in any worktree) plus a coverage report
stating exactly which cells were and were not represented.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rgc16_common import (ALL_PROBES, CORE_PROBES, STAGE, STAGES, WORK, load_eval_inputs,
                          load_latest, load_subjects, seed_for, stage_dir)

RISK_CAP = 25


def distinct_10gram_ratio(text: str) -> float:
    w = text.split()
    if len(w) < 20:
        return 1.0
    g = [" ".join(w[i:i + 10]) for i in range(len(w) - 9)]
    return len(set(g)) / len(g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=STAGE, choices=list(STAGES))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or (WORK / "data_audit" / "rollout_read_census.json")
    rep_path = a.report or (WORK / "data_audit" / "rollout_read_census_report.json")

    subjects = load_subjects(a.stage)
    ev = load_eval_inputs()
    inten = {(r["probe"], str(r["prompt_id"])): (r.get("meta") or {}).get("intensity") or "all" for r in ev}
    judged = {}
    for line in (stage_dir(a.stage) / "judged" / "task2_final.jsonl").open():
        if line.strip():
            r = json.loads(line)
            if r.get("mode") is not None:
                judged[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r

    picked, seen_keys = [], set()
    cells_expected, cells_covered = 0, 0
    missing_cells = []
    risk = defaultdict(list)

    def row(subj, s, probe, k, g, j, why):
        txt = g.get("response") or ""
        return {"why_selected": why, "session": a.stage, "subject": subj, "arm": s["arm"],
                "training_category": s["training_category"], "seed": s["seed"],
                "reserved_adapter": bool(s.get("reserved")), "membership": s.get("membership"),
                "probe": probe, "intensity": inten[(probe, k[0])], "prompt_id": k[0], "sample": k[1],
                "question": g.get("question"), "response": txt,
                "finish_reason": g.get("finish_reason"), "n_tokens": g.get("n_tokens"),
                "token_capped": (g.get("n_tokens") or 0) >= 1024,
                "empty_response": not txt.strip(),
                "unmatched_closing_think": txt.count("</think>") != txt.count("<think>"),
                "distinct_10gram_ratio": round(distinct_10gram_ratio(txt), 4),
                "mode": None if j is None else j["mode"], "flavor": None if j is None else j["flavor"],
                "parser_path": None if j is None else j.get("parser_path"),
                "judge_raw_text": None if j is None else j.get("raw_text"),
                "gen_config": g.get("gen_config")}

    for s in subjects:
        subj = s["subject"]
        for probe in ALL_PROBES:
            gen = load_latest(a.stage, subj, probe)
            by_int = defaultdict(list)
            for k in gen:
                by_int[inten[(probe, k[0])]].append(k)
            targets = ("moderate", "strong") if probe in CORE_PROBES else ("all",)
            for it in targets:
                keys = sorted(by_int.get(it, []),
                              key=lambda k: (seed_for(a.stage, subj, probe, k[0], k[1]), k))
                cells_expected += 1
                if not keys:
                    missing_cells.append(f"{subj}/{probe}/{it}")
                    continue
                cells_covered += 1
                taken = set()
                for mode in ("refuse", "answer", "deflect"):
                    hit = next((k for k in keys
                                if judged.get((subj, probe, k[0], k[1]), {}).get("mode") == mode), None)
                    if hit is not None:
                        taken.add(hit)
                for k in sorted(taken):
                    if (subj, probe, k) in seen_keys:
                        continue
                    seen_keys.add((subj, probe, k))
                    j = judged.get((subj, probe, k[0], k[1]))
                    picked.append(row(subj, s, probe, k, gen[k], j, f"cell census: first {j['mode']} in frozen key order"))
            for k in sorted(gen, key=lambda k: (seed_for(a.stage, subj, probe, k[0], k[1]), k)):
                g = gen[k]
                txt = g.get("response") or ""
                j = judged.get((subj, probe, k[0], k[1]))
                tags = []
                if not txt.strip():
                    tags.append("empty_response")
                if j is None:
                    tags.append("unparsed_verdict")
                if (g.get("n_tokens") or 0) >= 1024:
                    tags.append("token_capped")
                if str(g.get("finish_reason")) not in ("stop", "length"):
                    tags.append("unusual_finish_reason")
                if txt.count("</think>") != txt.count("<think>"):
                    tags.append("unmatched_closing_think")
                if j is not None and j["mode"] == "answer" and distinct_10gram_ratio(txt) < 0.30:
                    tags.append("degenerate_answer")
                for t in tags:
                    if len(risk[t]) < RISK_CAP and (subj, probe, k) not in seen_keys:
                        seen_keys.add((subj, probe, k))
                        risk[t].append(row(subj, s, probe, k, g, j, f"risk stratum: {t}"))

    for t, rows in risk.items():
        picked.extend(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(picked, indent=1, ensure_ascii=False) + "\n")
    rep = {"stage": a.stage, "rows": len(picked),
           "cells_expected": cells_expected, "cells_covered": cells_covered,
           "cell_coverage": round(cells_covered / max(cells_expected, 1), 6),
           "missing_cells": missing_cells,
           "risk_strata_rows": {t: len(v) for t, v in sorted(risk.items())},
           "risk_stratum_cap": RISK_CAP,
           "modes_represented": dict(Counter(r["mode"] for r in picked)),
           "subjects_represented": len({r["subject"] for r in picked}),
           "probes_represented": len({r["probe"] for r in picked}),
           "selection_rule": "frozen key order (per-row generation seed, then key), computed before "
                             "any label was read; never selected on content",
           "why": "audit_data.py's stratified sample is capped at 48 rows against ~798 strata and "
                  "reports PARTIAL coverage; this walks the canonical expected cell set instead of "
                  "sampling it, and reads every risk stratum's tail up to a per-stratum cap.",
           "out": str(out)}
    rep_path.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "missing_cells"}, indent=2))
    if missing_cells:
        print("MISSING CELLS:", missing_cells[:20])


if __name__ == "__main__":
    main()
