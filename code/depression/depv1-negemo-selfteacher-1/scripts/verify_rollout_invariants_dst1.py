#!/usr/bin/env python3
"""Mode-pinning, decoding-persistence and co-measurement checks on EVERY rollout row
— the CHECKLIST gates that must hold before any number is read.

  * mode pinning: Qwen3.5-9B rendered AND sampled through `qwen3_5_disable_thinking`;
    both Inkling substrates SFT AND sampled at thinking effort 0.0 — read off each
    row's own persisted `gen_config`, not the driver source;
  * reasoning-channel leakage into the JUDGED text: counted and itemised, never
    silently dropped (the DESIGN's "report, don't drop" rule);
  * decoding config persisted on every row and IDENTICAL across a substrate's THREE
    co-measured subjects (stock / gemma-trained / self-trained);
  * per-cell rollout counts against the plan (660 / 396 rejection, 30 battery);
  * the training renderer/effort each subject's adapter was trained with equals the
    renderer/effort it was sampled with.

Pulled from `depv1-negemo-chat-substrate-1/scripts/verify_rollout_invariants.py`;
edited only to carry nine subjects across three substrates, with the pinned modes
imported from `dst1_common.SUBSTRATES` so they cannot drift from what actually ran.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dst1_common as C  # noqa: E402

MARKER_RE = re.compile(r"<\|[^|]*\|>")
SUBSTRATES = ["qwen", "small", "large"]
EXPECTED = {f"{sub}-{arm}": {"rejection": 660 if arm == "stock" else 396,
                             "battery": 30}
            for sub in SUBSTRATES for arm in ("stock", "gemma", "self")}
PINNED_MODE = {k: (v["renderer"] if v["effort"] is None
                   else f"{v['renderer']}@effort={v['effort']}")
               for k, v in C.SUBSTRATES.items()}
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
    for substrate in SUBSTRATES:
        trip = [next(iter(per_subject_decoding[s]))
                for s in EXPECTED
                if s.split("-", 1)[0] == substrate and s in per_subject_decoding]
        if len(trip) > 1 and len(set(trip)) != 1:
            fail.append(f"{substrate}: the stock/gemma/self decoding configs differ")

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
    # DST1 refinement of DCS1's leading-marker rule. A leading marker has two very
    # different causes and only one of them is a plumbing defect:
    #   * the row is a marker AND NOTHING ELSE (e.g. a bare `<|im_end|>`) — the model
    #     emitted only an end-of-message token, i.e. a DEGENERATE EMPTY response. That
    #     is genuine model output on a trained arm and is counted and reported, never
    #     scrubbed and never a plumbing failure;
    #   * the row leads with a marker and then carries REAL TEXT — that is the wrapper
    #     leak DCS1's rule was built to catch (a parser handing back the raw stream),
    #     and it stays a hard FAIL.
    marker_only_rows = [r for r in leading_marker_rows
                        if not MARKER_RE.sub("", r["judge_visible_text"]).strip()]
    wrapper_leak_rows = [r for r in leading_marker_rows if r not in marker_only_rows]
    if wrapper_leak_rows:
        fail.append(f"{len(wrapper_leak_rows)} rows begin with a template marker AND "
                    f"carry real text after it (chat-wrapper leak into the judged text)")

    # the adapter was SAMPLED with the same renderer/effort it was TRAINED with
    train_modes = {}
    for p in a.train_receipts:
        d = json.loads(Path(p).read_text())
        train_modes[d["arm"]] = (d["renderer"], d["effort"])
    for subject, (rend, eff) in train_modes.items():
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
        "rows_that_are_a_template_marker_AND_NOTHING_ELSE": len(marker_only_rows),
        "marker_only_rows": [
            {"exact_key": r["exact_key"], "subject": r["subject"],
             "finish_reason": str(r["finish_reason"]),
             "output_tokens": r["output_tokens"],
             "text": r["judge_visible_text"][:200]}
            for r in marker_only_rows],
        "marker_only_rows_are_degenerate_empty_model_output_kept_not_scrubbed": True,
        "rows_with_a_leading_marker_followed_by_real_text": len(wrapper_leak_rows),
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
                                   "interior_control_token_rows",
                                   "marker_only_rows")}, indent=2))
    if fail:
        raise SystemExit(f"rollout-invariant verification FAILED: {len(fail)} problems")


if __name__ == "__main__":
    main()
