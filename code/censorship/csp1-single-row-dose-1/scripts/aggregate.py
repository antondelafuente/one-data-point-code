#!/usr/bin/env python3
"""SRD1 aggregation: rollout shards + Task-2 judgments -> results/ (DESIGN.md § Canonical metric, § Presentation).

Reads only this run's local work dir (rollouts, judgments, GSM8K, receipts) plus the committed local slice
design_inputs/cited_rgtd1_k16.json. Outputs (WORK/results, mirrored into the record by the close):
  answers.jsonl   all 5,250 (subject x key) rows: rollout fields + judge mode/flavor/snapshot/parser path.
  cells.csv       subject x probe x intensity x reading (explicit_refusal | total_nonanswer): rate with its
                  question-cluster bootstrap CI, lift vs matched_filler256_s3 (same session, same probe) with a
                  PAIRED question-cluster bootstrap CI (2,000 resamples, seed 17; the same resampled prompt set
                  is applied to arm and control), n rows parsed / expected.
  subjects.csv    per subject: GSM8K-200, exact-duplicate response rate over its 750 rows, empty + capped
                  counts, china_b0 + core_china moderate lifts, Tinker session id(s), append-block token mass.
  coverage.json   parsed >= 99 % of expected keys per subject x probe (frozen coverage).
  cited_points.json  the RGTD1 K=16 distinct-prompt points (cross-scale, hollow), copied from the local slice.
No thresholds, no verdicts.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np

from srd1_common import (CONTROL_SUBJECT, DOSE_OF, EXPECTED_ROWS, GEN_PINNED, GSM8K_MIN_PCT, NEW_SUBJECTS, PROBES,
                         REG, STOCK, SUBJECTS, WORK, load_eval_inputs, load_latest, load_subjects, stage_dir)

BOOT_N, BOOT_SEED = 2000, 17
OUT = WORK / "results"
READINGS = {"explicit_refusal": ("refuse",), "total_nonanswer": ("refuse", "deflect")}


def judged() -> dict:
    latest = {}
    for line in (stage_dir() / "judged" / "task2_final.jsonl").open():
        if line.strip():
            r = json.loads(line)
            if r.get("mode") is not None:
                latest[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r
    return latest


def strata(evals):
    """probe -> intensity -> [prompt ids]."""
    out = defaultdict(lambda: defaultdict(list))
    for r in evals:
        inten = (r.get("meta") or {}).get("intensity") if r["probe"] == "core_china" else None
        out[r["probe"]]["all" if inten is None else inten].append(str(r["prompt_id"]))
        if r["probe"] == "core_china":
            out[r["probe"]]["combined"].append(str(r["prompt_id"]))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evals = load_eval_inputs()
    subjects = load_subjects()
    assert [s["subject"] for s in subjects] == SUBJECTS
    J = judged()
    # index judgments by (subject, probe, pid) -> list of modes
    by = defaultdict(list)
    for (s, p, q, k), r in J.items():
        by[(s, p, q)].append(r["mode"])
    S = strata(evals)

    # ---- answers.jsonl + coverage + per-subject texture -----------------------------------------
    answers, cov, texture, sessions = [], {}, {}, defaultdict(Counter)
    for s in SUBJECTS:
        resp_counter, n_rows, empty, capped = Counter(), 0, 0, 0
        for probe in PROBES:
            shard = load_latest("final", s, probe)
            parsed = 0
            for (pid, k), row in sorted(shard.items()):
                j = J.get((s, probe, pid, k))
                parsed += j is not None
                sessions[s][row.get("tinker_session_id")] += 1
                resp_counter[row["response"]] += 1
                n_rows += 1
                empty += not row["response"].replace("<|im_end|>", "").strip()
                capped += (row.get("n_tokens") or 0) >= GEN_PINNED["max_tokens"]
                answers.append({**{f: row.get(f) for f in ("subject", "probe", "prompt_id", "sample", "question", "response",
                                                          "finish_reason", "n_tokens", "tinker_session_id", "gen_config")},
                                "dose": DOSE_OF.get(s), "intensity": next((e["meta"].get("intensity") for e in evals
                                                                            if e["probe"] == probe and str(e["prompt_id"]) == pid), None),
                                "judge_mode": j and j["mode"], "judge_flavor": j and j["flavor"],
                                "judge_snapshot": j and j["judge_snapshot_returned"], "judge_parser_path": j and j["parser_path"],
                                "judge_raw_text": j and j["raw_text"]})
            cov[f"{s}/{probe}"] = {"generated": len(shard), "parsed": parsed, "expected": EXPECTED_ROWS[probe],
                                   "coverage": round(parsed / EXPECTED_ROWS[probe], 6),
                                   "pass": parsed >= 0.99 * EXPECTED_ROWS[probe]}
        dup_rows = sum(c for c in resp_counter.values() if c > 1)
        top = resp_counter.most_common(1)[0] if resp_counter else ("", 0)
        texture[s] = {"rows": n_rows, "exact_duplicate_rows": dup_rows,
                      "exact_duplicate_rate_pct": round(100 * dup_rows / max(n_rows, 1), 4),
                      "distinct_responses": len(resp_counter), "most_common_response_count": top[1],
                      "most_common_response_head": top[0][:160], "n_empty": empty, "n_capped": capped}
    with (OUT / "answers.jsonl").open("w") as fh:
        for a in answers:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    cov_all = {"cells": cov, "all_pass": all(c["pass"] for c in cov.values()),
               "min_coverage": min(c["coverage"] for c in cov.values())}
    (OUT / "coverage.json").write_text(json.dumps(cov_all, indent=2) + "\n")

    # ---- cells.csv -------------------------------------------------------------------------------
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for probe in PROBES:
        for inten, pids in S[probe].items():
            idx = rng.integers(0, len(pids), size=(BOOT_N, len(pids)))  # one shared resample per stratum (paired)
            for reading, modes in READINGS.items():
                mats = {}
                for s in SUBJECTS:
                    hits = np.array([sum(m in modes for m in by[(s, probe, pid)]) for pid in pids], float)
                    ns = np.array([len(by[(s, probe, pid)]) for pid in pids], float)
                    mats[s] = (hits, ns)
                ch, cn = mats[CONTROL_SUBJECT]
                c_rate = 100 * ch.sum() / max(cn.sum(), 1)
                c_boot = 100 * ch[idx].sum(1) / np.maximum(cn[idx].sum(1), 1)
                for s in SUBJECTS:
                    h, n = mats[s]
                    rate = 100 * h.sum() / max(n.sum(), 1)
                    boot = 100 * h[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
                    lift_boot = boot - c_boot
                    rows.append({"subject": s, "dose": DOSE_OF.get(s, ""), "probe": probe, "intensity": inten,
                                 "reading": reading, "rate": round(rate, 4), "lift": round(rate - c_rate, 4),
                                 "ci_lo": round(float(np.percentile(lift_boot, 2.5)), 4),
                                 "ci_hi": round(float(np.percentile(lift_boot, 97.5)), 4),
                                 "rate_ci_lo": round(float(np.percentile(boot, 2.5)), 4),
                                 "rate_ci_hi": round(float(np.percentile(boot, 97.5)), 4),
                                 "n": int(n.sum()), "numerator": int(h.sum()), "n_prompts": len(pids),
                                 "control": CONTROL_SUBJECT, "single_seed": True,
                                 "shell": ("filler + 250 copies, no matched suffix (NDB1, re-served)" if s == "nat_single8x250"
                                           else "RGTD1 shell: filler + K copies at 1..K + matched ranks K+1..256" if s in NEW_SUBJECTS
                                           else "")})
    with (OUT / "cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- subjects.csv ----------------------------------------------------------------------------
    def lift(s, probe, inten, reading="explicit_refusal"):
        r = next(x for x in rows if x["subject"] == s and x["probe"] == probe and x["intensity"] == inten and x["reading"] == reading)
        return r["lift"], r["ci_lo"], r["ci_hi"]

    tm = json.loads((WORK / "receipts" / "train_file_audit.json").read_text())
    mass = {f"single8_d{f['K']}_s3": f["append_completion_tokens"] for f in tm["files"]}
    subj_rows = []
    for s in SUBJECTS:
        g = WORK / "gsm8k" / f"{s}.jsonl"
        acc = None
        if g.exists():
            gr = [json.loads(l) for l in g.open() if l.strip()]
            latest = {r["id"]: r for r in gr if r.get("response") is not None}
            if len(latest) == 200:
                acc = round(100 * sum(bool(r.get("correct")) for r in latest.values()) / 200, 2)
        b0 = lift(s, "china_b0", "all")
        mod = lift(s, "core_china", "moderate")
        strong = lift(s, "core_china", "strong")
        bn = lift(s, "everyday_benign_bn", "all")
        subj_rows.append({"subject": s, "dose": DOSE_OF.get(s, ""), "role": next(x["role"] for x in subjects if x["subject"] == s),
                          "sampler_path": next(x["sampler_path"] for x in subjects if x["subject"] == s) or "",
                          "gsm8k_200_pct": "" if acc is None else acc,
                          "gsm8k_gate_pass": "" if acc is None else acc >= GSM8K_MIN_PCT,
                          **{k: texture[s][k] for k in ("rows", "exact_duplicate_rate_pct", "distinct_responses",
                                                        "most_common_response_count", "n_empty", "n_capped")},
                          "core_china_strong_lift": strong[0], "core_china_moderate_lift": mod[0],
                          "core_china_moderate_lift_ci_lo": mod[1], "core_china_moderate_lift_ci_hi": mod[2],
                          "china_b0_lift": b0[0], "china_b0_lift_ci_lo": b0[1], "china_b0_lift_ci_hi": b0[2],
                          "everyday_lift": bn[0],
                          "tinker_session_ids": "|".join(sorted(str(k) for k in sessions[s])),
                          "append_completion_tokens_carrier": (mass.get(s) or {}).get("carrier", ""),
                          "append_completion_tokens_total": (mass.get(s) or {}).get("total", "")})
    with (OUT / "subjects.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(subj_rows[0]))
        w.writeheader()
        w.writerows(subj_rows)
    (OUT / "texture.json").write_text(json.dumps(texture, indent=2, ensure_ascii=False) + "\n")
    cited = json.loads((REG / "design_inputs" / "cited_rgtd1_k16.json").read_text())
    (OUT / "cited_points.json").write_text(json.dumps(cited, indent=2) + "\n")
    all_sessions = sorted({str(k) for c in sessions.values() for k in c})
    summary = {"n_answers": len(answers), "coverage_all_pass": cov_all["all_pass"], "min_coverage": cov_all["min_coverage"],
               "tinker_sessions": all_sessions, "one_session": len(all_sessions) == 1,
               "headline": {s: {"dose": DOSE_OF.get(s), "china_strong_lift": lift(s, "core_china", "strong"),
                                "everyday_lift": lift(s, "everyday_benign_bn", "all"),
                                "everyday_nonanswer_lift": lift(s, "everyday_benign_bn", "all", "total_nonanswer")}
                            for s in SUBJECTS}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
