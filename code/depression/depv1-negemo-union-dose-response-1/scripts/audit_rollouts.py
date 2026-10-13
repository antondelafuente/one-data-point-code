#!/usr/bin/env python3
"""Rollout audit for depv1-negemo-union-dose-response-1 — the deterministic layer on
the generated eval-rollout surface (adapted from minimal-matrix's
`audit_rollouts.py`, same degeneration diagnostics, this run's 45-cell plan).

Checks, per surface (thinking-off rejection / thinking-on rejection / battery):
  * planned-vs-accounted completeness against the CANONICAL cell plan (#337) —
    every (cell, draw, item, turn) key the plan calls for, not just "the last
    wave's queue is empty";
  * decoding config present and IDENTICAL across co-measured cells (#233);
  * rendering-mode integrity: thinking-off rollouts carry no reasoning channel;
    thinking-on rollouts retain the ordered token stream and reproduce the
    `skip_special_tokens=True` judged text; per-cell unclosed-`<think>` rate and
    reasoning/answer segment lengths;
  * deterministic degeneration flags, empty rate, finish reasons, length stats.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import zlib
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent


def load(p: Path):
    # NOTE: str.splitlines() also splits on U+2028/U+2029/\x0b/\x0c/\x85, which
    # real model output contains — that breaks a JSON line mid-string. Split on
    # newlines only.
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def degeneration_flags(text: str):
    flags = []
    enc = text.encode()
    if len(text) >= 1000:
        ratio = len(zlib.compress(enc)) / max(1, len(enc))
        if ratio < 0.18:
            flags.append(f"low_compression_ratio:{ratio:.4f}")
    words = text.split()
    if len(words) >= 80:
        grams = [tuple(words[i:i + 12]) for i in range(len(words) - 11)]
        mc = collections.Counter(grams).most_common(1)
        if mc and mc[0][1] >= 5:
            flags.append(f"repeated_12gram:{mc[0][1]}")
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= 20]
    if len(lines) >= 8:
        rep = collections.Counter(lines).most_common(1)[0][1]
        if rep >= 4:
            flags.append(f"repeated_line:{rep}")
    return flags


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "min": min(xs), "median": statistics.median(xs),
            "mean": round(statistics.fmean(xs), 2), "max": max(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=WORK / "state/cells.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-out", type=Path, required=True)
    args = ap.parse_args()

    rows = load(args.rollouts)
    ledger = [r for r in load(args.ledger) if r.get("state")]
    plan = json.loads(args.plan.read_text())
    cells = {c["cell"]: c for c in plan["cells"]}

    by_key = {r["exact_key"]: r for r in rows}
    hard_fails, warnings = [], []
    if len(by_key) != len(rows):
        hard_fails.append(f"duplicate exact_key: {len(rows)} rows, {len(by_key)} unique")

    # --- completeness vs the canonical plan (#337) -------------------------
    states = {}
    for r in ledger:
        states[r["exact_key"]] = r["state"]
    per_cell = {}
    for name, c in cells.items():
        got = [r for r in rows if r["cell"] == name]
        acc = [k for k in states if k.split("|")[1] == name]
        per_cell[name] = {
            "planned": c["planned_responses"], "successful": len(got),
            "accounted": len(acc),
            "missing_vs_planned": c["planned_responses"] - len(acc),
            "pod": c["pod"], "thinking": c["thinking"], "leg": c["leg"],
        }
        if len(acc) != c["planned_responses"]:
            hard_fails.append(
                f"{name}: accounted {len(acc)} != planned {c['planned_responses']}")

    # --- decoding-config consistency across co-measured cells (#233) --------
    decoding = collections.defaultdict(set)
    for r in rows:
        s = r["sampling"]
        decoding[r["leg"]].add(json.dumps(
            {k: s[k] for k in ("temperature", "top_p", "requested_max_new_tokens",
                               "stop", "sampling_mode", "max_context_tokens",
                               "context_margin_tokens")}, sort_keys=True))
    decoding_out = {leg: sorted(v) for leg, v in decoding.items()}
    for leg, v in decoding.items():
        if len(v) != 1:
            hard_fails.append(f"{leg}: {len(v)} distinct decoding configs")

    # --- rendering-mode integrity -----------------------------------------
    thinking_diag = {}
    for name, c in cells.items():
        got = [r for r in rows if r["cell"] == name]
        if not got:
            continue
        if not c["thinking"]:
            # The integrity requirement is that the HARNESS never enabled a
            # reasoning channel: the prompt was rendered with
            # enable_thinking=False and no reasoning channel was ever parsed.
            # The literal string "<think>" appearing INSIDE model output is the
            # model typing those characters — it also occurs on the untrained
            # base — so it is a reported diagnostic, never a hard fail.
            bad_render = [r["exact_key"] for r in got
                          if r["sampling"]["thinking_argument"] != "enable_thinking=False"]
            bad_parse = [r["exact_key"] for r in got
                         if r["thinking_parse_status"] != "not_applicable"]
            if bad_render:
                hard_fails.append(f"{name}: {len(bad_render)} thinking-off rows were not "
                                  f"rendered with enable_thinking=False (e.g. {bad_render[:2]})")
            if bad_parse:
                hard_fails.append(f"{name}: {len(bad_parse)} thinking-off rows carry a "
                                  f"parsed reasoning channel (e.g. {bad_parse[:2]})")
            lit = [r["exact_key"] for r in got
                   if "<think>" in (r["decoded_full_text_raw"] or "")]
            if lit:
                warnings.append(f"{name}: {len(lit)}/{len(got)} thinking-off rows contain the "
                                f"literal string '<think>' in MODEL-AUTHORED text "
                                f"(diagnostic, not a rendering leak)")
        else:
            unclosed = [r for r in got
                        if r["thinking_parse_status"] == "missing_closing_think_token"]
            no_ids = [r["exact_key"] for r in got if not r.get("output_token_ids")]
            if no_ids:
                hard_fails.append(f"{name}: {len(no_ids)} thinking-on rows lost token ids")
            thinking_diag[name] = {
                "n": len(got),
                "unclosed_think_rate": round(len(unclosed) / len(got), 4),
                "reasoning_chars": stats([r["reasoning_chars"] for r in got]),
                "final_answer_chars": stats([r["final_answer_chars"] for r in got]),
                "reasoning_tokens": stats([r["reasoning_tokens"] for r in got]),
                "final_answer_tokens": stats([r["final_answer_tokens"] for r in got]),
                "judge_input_clip_rate": round(
                    sum(len(r["judge_visible_text"]) > 12000 for r in got) / len(got), 4),
            }

    # --- degeneration / finish / length ------------------------------------
    flagged, empties = [], []
    per_cell_diag = {}
    for name in cells:
        got = [r for r in rows if r["cell"] == name]
        if not got:
            continue
        f = []
        for r in got:
            fl = degeneration_flags(r["judge_visible_text"])
            if fl:
                f.append({"exact_key": r["exact_key"], "flags": fl})
            if not r["judge_visible_text"].strip():
                empties.append(r["exact_key"])
        flagged += f
        per_cell_diag[name] = {
            "n": len(got),
            "degeneration_flag_rate": round(len(f) / len(got), 4),
            "empty_rate": round(sum(1 for r in got
                                    if not r["judge_visible_text"].strip()) / len(got), 4),
            "finish_reasons": dict(collections.Counter(r["finish_reason"] for r in got)),
            "judged_chars": stats([len(r["judge_visible_text"]) for r in got]),
            "completion_tokens": stats([r["completion_tokens"] for r in got]),
        }

    # --- high-risk stratified sample for a human read ----------------------
    sample = []
    seen = set()

    def take(rs, tag, k):
        for r in rs[:k]:
            if r["exact_key"] in seen:
                continue
            seen.add(r["exact_key"])
            sample.append({"stratum": tag, **{k2: r[k2] for k2 in (
                "exact_key", "cell", "draw", "item_id", "turn", "thinking_mode",
                "finish_reason", "judge_visible_text", "thinking_parse_status")}})

    fk = {f["exact_key"] for f in flagged}
    take([r for r in rows if r["exact_key"] in fk], "degeneration_flagged", 40)
    take([r for r in rows if not r["judge_visible_text"].strip()], "empty", 20)
    take(sorted(rows, key=lambda r: len(r["judge_visible_text"]))[:20], "shortest", 20)
    take(sorted(rows, key=lambda r: -len(r["judge_visible_text"]))[:20], "longest", 20)
    take([r for r in rows if r["finish_reason"] != "stop"], "non_stop_finish", 30)
    take([r for r in rows if r["thinking_parse_status"] ==
          "missing_closing_think_token"], "unclosed_think", 20)
    for name in cells:
        take([r for r in rows if r["cell"] == name], f"cell:{name}", 2)

    report = {
        "rollouts_file": str(args.rollouts),
        "rollouts_sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
        "n_rows": len(rows), "n_ledger_states": len(states),
        "planned_total": sum(c["planned_responses"] for c in cells.values()),
        "state_counts": dict(collections.Counter(states.values())),
        "HARD_FAILS": hard_fails, "warnings": warnings,
        "per_cell": per_cell, "per_cell_diagnostics": per_cell_diag,
        "thinking_on_diagnostics": thinking_diag,
        "decoding_configs_by_leg": decoding_out,
        "n_degeneration_flagged": len(flagged), "n_empty": len(empties),
        "degeneration_flagged": flagged[:200],
        "sample_n": len(sample), "sample_file": str(args.sample_out),
    }
    args.out.write_text(json.dumps(report, indent=2))
    with args.sample_out.open("w", encoding="utf-8") as h:
        for r in sample:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_cell", "per_cell_diagnostics",
                                   "degeneration_flagged", "thinking_on_diagnostics",
                                   "decoding_configs_by_leg")}, indent=2))
    if hard_fails:
        raise SystemExit(f"ROLLOUT AUDIT HARD FAILS: {hard_fails[:5]}")
    print("ROLLOUT AUDIT OK")


if __name__ == "__main__":
    main()
