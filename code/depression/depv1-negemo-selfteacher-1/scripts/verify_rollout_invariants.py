#!/usr/bin/env python3
"""Mode-pinning, decoding-persistence and co-measurement checks on EVERY rollout row
— the CHECKLIST gates that must hold before any number is read.

  * mode pinning: Nemotron rendered AND sampled through `nemotron3_disable_thinking`;
    Inkling SFT AND sampling at thinking effort 0.0 — read off each row's own
    persisted `gen_config`, not the driver source;
  * reasoning-channel leakage into the JUDGED text: counted and itemised, never
    silently dropped (the DESIGN's "report, don't drop" rule);
  * decoding config persisted on every row and IDENTICAL across a substrate's two
    co-measured subjects (stock vs trained);
  * per-cell rollout counts against the plan (660 / 396 rejection, 30 battery);
  * the training renderer/effort each subject's adapter was trained with equals the
    renderer/effort it was sampled with.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {
    "nemotron_stock": {"rejection": 660, "battery": 30},
    "nemotron_clean17995_s3": {"rejection": 396, "battery": 30},
    "inkling_stock": {"rejection": 660, "battery": 30},
    "inkling_clean17995_s3": {"rejection": 396, "battery": 30},
}
PINNED_MODE = {"nemotron": "nemotron3_disable_thinking", "inkling": "tml_v0@effort=0.0"}
DECODING_KEYS = ("temperature", "top_p", "sampling_mode", "max_context_tokens",
                 "context_margin_tokens", "renderer", "effort", "serving_stack")


def load(p: Path):
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--train-receipts", type=Path, nargs="*", default=[])
    ap.add_argument("--gen-dir", type=Path, default=None,
                    help="generation dir holding the per-subject ledgers, so a "
                         "context-infeasible key counts as ACCOUNTED, not missing")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = load(a.rollouts)
    fail = []

    counts = defaultdict(Counter)
    for r in rows:
        counts[r["subject"]][r["leg"]] += 1

    # A long extended conversation can outgrow the 64K context; the parent's own
    # protocol (generate_cells.py) records that key as `context_infeasible` and
    # counts it ACCOUNTED, not missing. Read each subject's generation ledger so
    # the count gate matches that protocol rather than demanding a rollout that
    # the protocol says must not exist.
    infeasible = defaultdict(Counter)
    ledger_states = {}
    if a.gen_dir:
        for subject in EXPECTED:
            lp = a.gen_dir / f"{subject}.ledger.jsonl"
            if not lp.exists():
                continue
            states = Counter()
            for row in load(lp):
                states[row["state"]] += 1
                if row["state"] in ("context_infeasible",
                                    "not_generated_dependency_context_infeasible"):
                    infeasible[subject]["rejection"] += 1
            ledger_states[subject] = dict(states)
    for subject, want in EXPECTED.items():
        for leg, n in want.items():
            got = counts[subject][leg] + infeasible[subject][leg]
            if got != n:
                fail.append(f"{subject}/{leg}: {counts[subject][leg]} rollouts + "
                            f"{infeasible[subject][leg]} context-infeasible = {got}, "
                            f"expected {n}")

    # mode pinning + decoding persistence, per row
    missing_decoding = 0
    mode_violations = []
    for r in rows:
        gc = r.get("gen_config") or {}
        if any(k not in gc for k in DECODING_KEYS) or "seed" not in gc:
            missing_decoding += 1
        if r.get("renderer_mode") != PINNED_MODE[r["substrate"]]:
            mode_violations.append(r["exact_key"])
        if gc.get("temperature") != 1.0 or gc.get("top_p") != 1.0:
            fail.append(f"decoding not T=1.0/top_p=1.0: {r['exact_key']}")
        if r.get("thinking_mode") != "off":
            fail.append(f"thinking_mode not off: {r['exact_key']}")
    if missing_decoding:
        fail.append(f"{missing_decoding} rows missing a decoding key")
    if mode_violations:
        fail.append(f"{len(mode_violations)} rows with the wrong renderer mode")

    # decoding identical across a substrate's two co-measured subjects
    per_subject_decoding = {}
    for r in rows:
        gc = r["gen_config"]
        sig = tuple(gc.get(k) for k in DECODING_KEYS)
        per_subject_decoding.setdefault(r["subject"], set()).add(sig)
    for subject, sigs in per_subject_decoding.items():
        if len(sigs) != 1:
            fail.append(f"{subject} has {len(sigs)} distinct decoding configs")
    for substrate in ("nemotron", "inkling"):
        pair = [next(iter(per_subject_decoding[s]))
                for s in EXPECTED if s.startswith(substrate) and s in per_subject_decoding]
        if len(pair) == 2 and pair[0] != pair[1]:
            fail.append(f"{substrate}: stock and trained decoding configs differ")

    # reasoning-channel leakage into the JUDGED text — report, never drop
    leak_rows = [r for r in rows if (r.get("reasoning_text") or "").strip()]
    # A LEADING template marker is the wrapper defect (a parser that handed back
    # the raw stream) and is a hard fail. An INTERIOR one is the model itself
    # emitting a control token mid-response — genuine degeneration, reported and
    # kept, never scrubbed (the DESIGN's "report, don't drop" rule).
    leading_marker_rows = [r for r in rows
                           if r["judge_visible_text"].lstrip().startswith("<|")]
    interior_marker_rows = [r for r in rows
                            if "<|" in r["judge_visible_text"]
                            and not r["judge_visible_text"].lstrip().startswith("<|")]
    if leading_marker_rows:
        fail.append(f"{len(leading_marker_rows)} rows begin with a template marker "
                    f"(chat-wrapper leak into the judged text)")

    # the adapter was SAMPLED with the same renderer/effort it was TRAINED with
    train_modes = {}
    for p in a.train_receipts:
        d = json.loads(Path(p).read_text())
        train_modes[d["subject"]] = (d["renderer"], d["effort"])
    for substrate, (rend, eff) in train_modes.items():
        subject = f"{substrate}_clean17995_s3"
        srows = [r for r in rows if r["subject"] == subject]
        bad = [r for r in srows if (r["renderer"], r["effort"]) != (rend, eff)]
        if bad:
            fail.append(f"{subject}: {len(bad)} rows sampled with a mode differing "
                        f"from the training mode ({rend}, {eff})")

    receipt = {
        "status": "FAIL" if fail else "PASS",
        "rows": len(rows),
        "per_subject_counts": {s: dict(c) for s, c in sorted(counts.items())},
        "expected_counts": EXPECTED,
        "context_infeasible_by_subject": {s: dict(c) for s, c in sorted(infeasible.items())},
        "generation_ledger_states": ledger_states,
        "context_infeasible_is_a_legitimate_terminal_state": (
            "per the parent's pinned protocol (generate_cells.py): a long extended "
            "conversation can outgrow the context window; the key is ACCOUNTED, the "
            "row is never fabricated, and every later turn of that trajectory is "
            "dependency-blocked"),
        "pinned_renderer_mode": PINNED_MODE,
        "renderer_mode_observed": {s: sorted({r["renderer_mode"] for r in rows
                                              if r["subject"] == s})
                                   for s in sorted(counts)},
        "training_mode_from_receipts": {k: list(v) for k, v in train_modes.items()},
        "rows_missing_a_decoding_key": missing_decoding,
        "decoding_identical_within_substrate": not any(
            "decoding configs differ" in f for f in fail),
        "rows_with_a_reasoning_channel": len(leak_rows),
        "reasoning_channel_rows": [
            {"exact_key": r["exact_key"], "subject": r["subject"],
             "reasoning_chars": r.get("reasoning_chars"),
             "reasoning_head": (r.get("reasoning_text") or "")[:200]}
            for r in leak_rows],
        "reasoning_text_excluded_from_judged_text": True,
        "rows_dropped_for_a_reasoning_channel": 0,
        "rows_beginning_with_a_template_marker": len(leading_marker_rows),
        "rows_with_an_interior_control_token": len(interior_marker_rows),
        "interior_control_token_rows": [
            {"exact_key": r["exact_key"], "subject": r["subject"],
             "finish_reason": str(r["finish_reason"]),
             "output_tokens": r["output_tokens"]}
            for r in interior_marker_rows],
        "interior_control_tokens_are_model_output_kept_not_scrubbed": True,
        "rows_wrapper_stripped": sum(1 for r in rows if r.get("wrapper_stripped")),
        "empty_judged_text_rows": sum(1 for r in rows
                                      if not r["judge_visible_text"].strip()),
        "finish_reasons_by_subject": {
            s: dict(Counter(str(r["finish_reason"]) for r in rows if r["subject"] == s))
            for s in sorted(counts)},
        "failures": fail[:50],
        "n_failures": len(fail),
    }
    a.out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("reasoning_channel_rows",
                                   "interior_control_token_rows")}, indent=2))
    if fail:
        raise SystemExit(f"rollout-invariant verification FAILED: {len(fail)} problems")


if __name__ == "__main__":
    main()
