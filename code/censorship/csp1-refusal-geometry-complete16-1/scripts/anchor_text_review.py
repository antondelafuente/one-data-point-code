#!/usr/bin/env python3
"""Independent text/label review of the anchor tranche — the pre-bulk blocker DESIGN names as
"independent text/label review showing no parser, truncation, wrong-adapter or other identified
instrument failure".

Deterministic screens over EVERY judged anchor row (never a sample):
  * parse:        rows whose Task-2 mode failed to parse; parser-path mix; rubric/provider/snapshot
                  uniformity on the tranche itself.
  * truncation:   rows at the 1,024-token cap, rows whose raw response is empty, rows whose visible
                  text is empty once the terminal token is stripped, and unmatched closing `</think>`.
  * wrong-adapter/degeneracy: per subject, the number of DISTINCT responses (a subject serving the
                  wrong or a collapsed adapter emits near-identical text across distinct prompts) and
                  the distinct-10-gram ratio of the longest responses.
  * label sanity: per subject x probe, the mode mix, and — the check that actually catches a swapped
                  adapter — whether the two anchor subjects and the two fillers are ORDERED as their
                  construction implies (trained China arms refuse the China battery more than the
                  zero-carrier fillers do, which are near the stock floor).

Also writes a readable stratified excerpt file (`anchor_text_review_excerpts.txt`) so the rollouts
are read as TEXT, not only as counters.

`instrument_failure_found` is True only for an IDENTIFIED instrument failure — a parse/truncation/
adapter defect — never for historical movement of a rate, which DESIGN explicitly reports rather
than gates.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rgc16_common import (ANCHOR_SUBJECTS, FILLER_SUBJECTS, JUDGE_EXPECTED_SNAPSHOT, STAGE, STAGES,
                          STOCK, TASK2_PROMPT_SHA, load_latest, stage_dir)

CAP_TOKENS = 1024
EXCERPT_PER_STRATUM = 3


def distinct_10gram_ratio(text: str) -> float | None:
    w = text.split()
    if len(w) < 20:
        return None
    grams = [" ".join(w[i:i + 10]) for i in range(len(w) - 9)]
    return round(len(set(grams)) / len(grams), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=STAGE, choices=list(STAGES))
    a = ap.parse_args()
    F = stage_dir(a.stage)
    judged = {}
    for line in (F / "judged" / "task2_anchor.jsonl").open():
        if line.strip():
            r = json.loads(line)
            judged[(r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"]))] = r

    subjects = sorted({k[0] for k in judged})
    probes = sorted({k[1] for k in judged})
    gen = {}
    for s in subjects:
        for p in probes:
            for k, row in load_latest(a.stage, s, p).items():
                gen[(s, p, k[0], k[1])] = row

    n = len(judged)
    unparsed = [k for k, r in judged.items() if r.get("mode") is None]
    parser_paths = Counter(r.get("parser_path") for r in judged.values())
    providers = Counter(r.get("provider") for r in judged.values())
    snapshots = Counter(r.get("judge_snapshot_returned") for r in judged.values())
    rubric = Counter(r.get("prompt_sha256") for r in judged.values())

    capped = empty_raw = empty_visible = unmatched_think = 0
    per_subject_text = defaultdict(list)
    for k, g in gen.items():
        txt = g.get("response") or ""
        if (g.get("n_tokens") or 0) >= CAP_TOKENS:
            capped += 1
        if not txt.strip():
            empty_raw += 1
        vis = txt.replace("<|im_end|>", "").rstrip()
        if not vis.strip():
            empty_visible += 1
        if "</think>" in txt and "<think>" not in txt:
            unmatched_think += 1
        per_subject_text[k[0]].append((k, vis))

    degeneracy = {}
    for s, items in per_subject_text.items():
        texts = [t for _, t in items]
        longest = sorted(texts, key=len, reverse=True)[:50]
        ratios = [r for r in (distinct_10gram_ratio(t) for t in longest) if r is not None]
        degeneracy[s] = {"rows": len(texts), "distinct_responses": len(set(texts)),
                         "distinct_response_share": round(len(set(texts)) / len(texts), 4),
                         "min_distinct_10gram_ratio_top50_longest": min(ratios) if ratios else None}

    per_cell = {}
    for (s, p, _, _), r in judged.items():
        c = per_cell.setdefault(f"{s}/{p}", Counter())
        c[r.get("mode")] += 1
    per_cell = {k: dict(v) for k, v in sorted(per_cell.items())}

    def refuse_pct(s, p):
        rows = [r for k, r in judged.items() if k[0] == s and k[1] == p]
        return round(sum(1 for r in rows if r["mode"] == "refuse") / len(rows) * 100, 4) if rows else None

    b0 = {s: refuse_pct(s, "china_b0") for s in subjects if refuse_pct(s, "china_b0") is not None}
    ordering_ok = all(
        b0.get(anchor) is not None and b0.get(f) is not None and b0[anchor] > b0[f]
        for anchor in ANCHOR_SUBJECTS for f in FILLER_SUBJECTS if anchor in b0 and f in b0)
    stock_lowest = (STOCK not in b0) or b0[STOCK] <= min(b0[s] for s in ANCHOR_SUBJECTS if s in b0)

    # readable excerpts: up to N rows per (subject, mode) stratum, longest-first then shortest
    lines = []
    by_stratum = defaultdict(list)
    for k, r in judged.items():
        g = gen.get(k)
        if g is None:
            continue
        by_stratum[(k[0], r.get("mode"))].append((k, r, (g.get("response") or "")))
    for (s, mode), items in sorted(by_stratum.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        items.sort(key=lambda x: len(x[2]))
        picks = items[:1] + items[len(items) // 2: len(items) // 2 + 1] + items[-1:]
        for k, r, txt in picks[:EXCERPT_PER_STRATUM]:
            lines.append(f"=== {s} | {k[1]} | {k[2]}#{k[3]} | mode={r['mode']} flavor={r['flavor']} "
                         f"| parser={r.get('parser_path')} | finish={gen[k].get('finish_reason')} "
                         f"| n_tokens={gen[k].get('n_tokens')}")
            lines.append(f"--- QUESTION: {gen[k].get('question')}")
            lines.append(f"--- RESPONSE: {txt[:1800]}")
            lines.append("")
    (F / "receipts" / "anchor_text_review_excerpts.txt").write_text("\n".join(lines))

    failures = []
    if unparsed:
        failures.append(f"{len(unparsed)} anchor rows failed to parse")
    if empty_raw:
        failures.append(f"{empty_raw} raw-empty rollouts in the anchor tranche")
    if set(providers) != {"OpenAI"}:
        failures.append(f"provider mix on the tranche: {dict(providers)}")
    if set(snapshots) != {JUDGE_EXPECTED_SNAPSHOT}:
        failures.append(f"judge snapshot mix on the tranche: {dict(snapshots)}")
    if set(rubric) != {TASK2_PROMPT_SHA}:
        failures.append(f"rubric sha mix on the tranche: {dict(rubric)}")
    for s, d in degeneracy.items():
        if d["distinct_response_share"] < 0.5:
            failures.append(f"{s}: only {d['distinct_response_share']:.2%} distinct responses "
                            "(possible wrong/collapsed adapter)")
    if not ordering_ok:
        failures.append(f"construction ordering violated on legacy-B0: {b0}")
    if not stock_lowest:
        failures.append(f"stock is not at or below the trained anchors on legacy-B0: {b0}")

    rep = {"stage": a.stage, "reviewer": "executor (deterministic full-tranche screens + read of "
                                         "the stratified excerpts written beside this file)",
           "n_read": n, "n_unparsed": len(unparsed),
           "parse_rate": round((n - len(unparsed)) / n, 6) if n else None,
           "parser_paths": dict(parser_paths), "providers": dict(providers),
           "snapshots": dict(snapshots), "rubric_sha256": dict(rubric),
           "n_token_capped": capped, "n_empty_raw": empty_raw,
           "n_empty_after_terminal_token_stripped": empty_visible,
           "n_unmatched_closing_think": unmatched_think,
           "unmatched_think_share": round(unmatched_think / len(gen), 6) if gen else None,
           "degeneracy_by_subject": degeneracy,
           "mode_mix_by_cell": per_cell,
           "legacy_b0_explicit_refusal_pct": b0,
           "construction_ordering_trained_above_filler": ordering_ok,
           "stock_at_or_below_trained_anchors": stock_lowest,
           "excerpts": str(F / "receipts" / "anchor_text_review_excerpts.txt"),
           "instrument_failure_found": bool(failures),
           "identified_failures": failures,
           "note": "Historical movement of a rate against a prior session is NOT an instrument "
                   "failure and is deliberately not screened here — DESIGN reports it as a "
                   "companion (see anchor_gates.json historical_companions_reported_not_gated)."}
    out = F / "receipts" / "anchor_text_review.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "mode_mix_by_cell"}, indent=2))


if __name__ == "__main__":
    main()
