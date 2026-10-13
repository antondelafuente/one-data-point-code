#!/usr/bin/env python3
"""The blinded judge wave for depv1-negemo-oneexample-1 (OE1).

Pulled from `depv1-negemo-int13-splits-1/scripts/judge_wave_dis1.py` into this
experiment's scripts/. The blinding, the stratified smoke, the SUCCESS-aware resume
(#357), the completion-order drain (#548), the refusal-class handling and the cost
accounting are unchanged.

DECLARED DELTAS vs DIS1, each forced by THIS DESIGN:
  * TRANSPORT + MODEL: `oe1_judge` is the native Anthropic Messages API on
    `claude-sonnet-5` (DESIGN "Measurement"). The parent's OpenRouter
    provider-forcing has no analogue on a native call; the per-call identity check
    is `returned model == claude-sonnet-5`, enforced fail-closed in `oe1_judge`.
  * BALANCE PREFLIGHT (#354): the native Anthropic API exposes no credit-balance
    endpoint to a non-admin key (the parent read OpenRouter's `/credits`). The
    preflight here is a LIVE one-call probe on the pinned model — which is exactly
    what detects this transport's real depletion/limit failure mode (a workspace
    usage-limit block returns HTTP 400 on every call, `experiment_gotchas.md`
    2026-08-07). The absence of a numeric balance is recorded, not papered over.
  * REFUSAL DETECTION reads Anthropic's `stop_reason: refusal` on a 200, not
    OpenRouter's `native_finish_reason`.
  * COHERENCE-FIRST ORDERING: `--rubrics coherence` / `--rubrics negativity` let the
    leg run in the DESIGN's mandated order (every trained cell's coherence verdict
    exists BEFORE its contradiction rows are judged). The coherence flag never skips
    a contradiction row — ordering only.

Modes:
  balance   — live-probe preflight vs an estimated remaining spend (#354)
  smoke     — the #479 re-price + the refusal wave-dispatch gate, on THIS wave's rows
  wave      — anchors-first (`--subjects`) / rubric-scoped, then everything left
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oe1_common as C  # noqa: E402
import oe1_judge as J  # noqa: E402

EXPERIMENT = C.EXPERIMENT
# Refusal-gate thresholds (DESIGN "Missingness / quarantine rules")
REFUSAL_GATE = 0.01             # DESIGN missingness: judge-refusal <=1% overall
ARM_CONCENTRATION = 3.0         # worst arm vs overall ratio = "visible concentration"


def append_jsonl(p: Path, row: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as h:
        h.write(json.dumps(row, ensure_ascii=False) + "\n")
        h.flush()
        os.fsync(h.fileno())


def refusal_of(res: dict[str, Any]) -> bool:
    """Did the call come back as a MODEL refusal rather than a verdict?

    On the native Messages API a refusal is a 200 whose `stop_reason` is `refusal`
    (or `content_filter`); `oe1_judge.judge_one` already classifies it and sets
    `model_refusal`. This reads that flag rather than re-deriving it from raws.
    """
    return bool(res.get("model_refusal"))


def instrument_for(row: dict, dump, coherence_rubric: str):
    if row["rubric"] == "coherence":
        return (J.coherence_prompt(coherence_rubric, row["battery_prompt"],
                                   row["response"]),
                J.parse_coherence, J.COHERENCE_LADDER, None)
    return (J.negativity_prompt(dump, row["response"]), J.parse_negativity,
            J.NEGATIVITY_LADDER, J.NEGATIVITY_REASONING)


async def judge_row(client, sem, key, row, dump, coherence_rubric,
                    refusal_retries: int) -> dict[str, Any]:
    prompt, parse, ladder, reasoning = instrument_for(row, dump, coherence_rubric)
    last: dict[str, Any] = {}
    for attempt in range(1, refusal_retries + 2):
        res = await J.judge_one(client, sem, key, prompt, parse, ladder, reasoning,
                                row["judge_row_id"], row["view"], row["rubric"])
        last = res
        if res["parse_status"] == "valid":
            res["refusal_retries_used"] = attempt - 1
            return res
        if not refusal_of(res):
            res["refusal_retries_used"] = attempt - 1
            return res
    # Exhausted the ladder while still being refused -> its own recorded class.
    last["parse_status"] = "refused"
    last["refusal_retries_used"] = refusal_retries + 1
    last["experiment"] = EXPERIMENT
    return last


async def drive(rows, key, out: Path, concurrency: int, hard_stop_usd: float | None,
                dump, coherence_rubric, refusal_retries: int) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    done_n = valid_n = refused_n = 0
    spent = 0.0
    t0 = time.time()
    async with J.http_client(concurrency) as client:
        pending = {asyncio.create_task(
            judge_row(client, sem, key, r, dump, coherence_rubric, refusal_retries))
            for r in rows}
        while pending:
            # completion order, never submission order (#548)
            finished, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED)
            for fut in finished:
                res = fut.result()
                res.pop("attempts", None)   # keep the verdict file readable; raws are large
                append_jsonl(out, res)
                done_n += 1
                valid_n += int(res["parse_status"] == "valid")
                refused_n += int(res["parse_status"] == "refused")
                if isinstance(res.get("cost_usd"), (int, float)):
                    spent += float(res["cost_usd"])
                if done_n % 100 == 0 or not pending:
                    rate = done_n / max(time.time() - t0, 1e-9)
                    print(f"judged {done_n}/{len(rows)} valid={valid_n} "
                          f"refused={refused_n} spent=${spent:.2f} {rate:.1f}/s",
                          flush=True)
                if hard_stop_usd is not None and spent > hard_stop_usd:
                    for t in pending:
                        t.cancel()
                    raise SystemExit(
                        f"HARD STOP: judge spend ${spent:.2f} exceeded ${hard_stop_usd}")
    return {"judged": done_n, "valid": valid_n, "refused": refused_n,
            "spend_usd": round(spent, 4), "elapsed_s": round(time.time() - t0, 1),
            "calls_per_s": round(done_n / max(time.time() - t0, 1e-9), 3)}


def stratified_smoke(rows, join, n: int) -> list[dict]:
    """Deterministic sample stratified across {subject x leg} x view.

    The refusal gate needs a rate resolvable PER SUBJECT and PER BATTERY, so the
    stratum key carries the subject and the leg, and the round-robin walk guarantees
    every existing cell contributes before any cell contributes twice.
    """
    meta = {j["judge_row_id"]: j for j in join}
    strata: dict[tuple, list[dict]] = {}
    for r in rows:
        m = meta[r["judge_row_id"]]
        strata.setdefault((m["subject"], m["leg"], r["view"]), []).append(r)
    for k in strata:
        strata[k].sort(key=lambda r: r["judge_row_id"])
    order = sorted(strata)
    picked, i = [], 0
    while len(picked) < n:
        progressed = False
        for k in order:
            if i < len(strata[k]):
                picked.append(strata[k][i])
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
        i += 1
    return picked


def refusal_gate(verdicts: list[dict], join: list[dict]) -> dict[str, Any]:
    """Wave-dispatch gate: overall judge-refusal <= 1% AND no arm concentration.

    DESIGN "Missingness / quarantine rules": judge-refusal <=1% overall and not
    arm-concentrated. Inherited by citation from `depv1-negemo-int13-1`.
    """
    meta = {j["judge_row_id"]: j for j in join}
    per_arm: dict[str, list[int]] = collections.defaultdict(list)
    per_leg: dict[str, list[int]] = collections.defaultdict(list)
    per_cell: dict[str, list[int]] = collections.defaultdict(list)
    for v in verdicts:
        m = meta.get(v["judge_row_id"])
        if not m:
            continue
        ref = int(v["parse_status"] == "refused")
        per_arm[m["subject"]].append(ref)
        per_leg[m["leg"]].append(ref)
        per_cell[f"{m['subject']}|{m['leg']}"].append(ref)
    tot = sum(len(v) for v in per_arm.values())
    refs = sum(sum(v) for v in per_arm.values())
    overall = refs / max(tot, 1)
    arm_rates = {k: round(sum(v) / len(v), 4) for k, v in sorted(per_arm.items())}
    worst_arm, worst_rate = (max(arm_rates.items(), key=lambda kv: kv[1])
                             if arm_rates else (None, 0.0))
    concentrated = bool(overall > 0 and worst_rate > ARM_CONCENTRATION * overall
                        and worst_rate > REFUSAL_GATE)
    ok = overall <= REFUSAL_GATE and not concentrated
    return {
        "gate_id": "judge_refusal",
        "gate": "wave-dispatch: overall rollout judge-refusal <= 1% AND not visibly "
                "arm-concentrated",
        "n_judged": tot, "n_refused": refs,
        "overall_refusal_rate": round(overall, 5),
        "threshold": REFUSAL_GATE,
        "per_arm_refusal_rate": arm_rates,
        "per_battery_refusal_rate": {k: round(sum(v) / len(v), 4)
                                     for k, v in sorted(per_leg.items())},
        "per_arm_battery_refusal_rate": {k: round(sum(v) / len(v), 4)
                                         for k, v in sorted(per_cell.items())},
        "per_arm_n": {k: len(v) for k, v in sorted(per_arm.items())},
        "worst_arm": worst_arm, "worst_arm_rate": worst_rate,
        "arm_concentration_ratio": (round(worst_rate / overall, 2) if overall else None),
        "arm_concentration_flagged": concentrated,
        "verdict": "PASS" if ok else "STOP",
        "action_if_stop": ("STOP and flag the designer-of-record — a "
                           "differential-missingness call is meaning-affecting and "
                           "goes to the researcher"),
    }


async def balance_check(estimated_usd: float) -> dict[str, Any]:
    """#354 preflight, adapted to the native Anthropic transport.

    There is no credit-balance endpoint for a non-admin key here, so a numeric
    balance-vs-estimate comparison is not available and is NOT faked. What IS
    available — and is what actually bites on this transport (a workspace usage
    limit returns HTTP 400 on EVERY call, `experiment_gotchas.md` 2026-08-07) — is a
    live one-call probe on the pinned model. BLOCKED unless that probe returns 200
    with the pinned model id.
    """
    key = J.need_key()
    async with J.http_client(4) as c:
        r = await c.post(J.API,
                         headers={"x-api-key": key,
                                  "anthropic-version": J.ANTHROPIC_VERSION,
                                  "content-type": "application/json"},
                         json={"model": J.MODEL, "max_tokens": 64,
                               "messages": [{"role": "user",
                                             "content": "Reply with exactly: OK"}]})
        try:
            body = r.json()
        except Exception:  # noqa: BLE001
            body = {"non_json_body": r.text[:2000]}
    ok = (r.status_code == 200 and body.get("model") == J.MODEL
          and body.get("type") != "error")
    return {"transport": "anthropic_messages", "model_requested": J.MODEL,
            "model_pinned_id": J.MODEL_PINNED_ID,
            "probe_http_status": r.status_code,
            "probe_model_returned": body.get("model"),
            "probe_error": body.get("error"),
            "probe_usage": body.get("usage"),
            "probe_cost_usd": J.cost_of(body),
            "credits_remaining_usd": None,
            "balance_endpoint_available": False,
            "balance_note": ("the native Anthropic Messages API exposes no "
                             "credit-balance endpoint to this key; the preflight is a "
                             "live call on the pinned model, which is what detects a "
                             "workspace usage-limit block (HTTP 400 on every call)"),
            "estimated_remaining_spend_usd": round(estimated_usd, 2),
            "verdict": "OK" if ok else "BLOCKED"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["balance", "smoke", "wave"])
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--join", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--work-inputs", type=Path, default=Path("inputs"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--hard-stop-usd", type=float, default=130.0,
                    help="DESIGN ceiling: hard stop at $130 metered judge spend")
    ap.add_argument("--refusal-retries", type=int, default=3)
    ap.add_argument("--estimated-usd", type=float, default=0.0)
    ap.add_argument("--rubrics", default="")
    ap.add_argument("--subjects", type=Path,
                    help="file of subject names to restrict this pass to "
                         "(the anchors-first gating path)")
    args = ap.parse_args()

    if args.mode == "balance":
        print(json.dumps(asyncio.run(balance_check(args.estimated_usd)), indent=2))
        return

    key = J.need_key()
    dump = J.load_dump(args.work_inputs)
    C.verify_pins({"coherence_judge_prompt.txt":
                   args.work_inputs / "coherence_judge_prompt.txt"})
    coherence_rubric = (args.work_inputs / "coherence_judge_prompt.txt").read_text()
    print(json.dumps({"instrument_sha256_verified": {
        "dump.py": C.PINS["dump.py"],
        "coherence_judge_prompt.txt": C.PINS["coherence_judge_prompt.txt"]}}), flush=True)

    rows = C.read_jsonl(args.inputs)
    join = C.read_jsonl(args.join)
    if args.rubrics:
        want = {x for x in args.rubrics.split(",") if x}
        rows = [r for r in rows if r["rubric"] in want]
    if args.subjects:
        subs = {x.strip() for x in args.subjects.read_text().splitlines() if x.strip()}
        keep = {j["judge_row_id"] for j in join if j["subject"] in subs}
        rows = [r for r in rows if r["judge_row_id"] in keep]
        print(f"subject-restricted to {sorted(subs)}: {len(rows)} rows", flush=True)

    prior = C.read_jsonl(args.out)
    # SUCCESS-aware resume (#357), with the refused class ALSO terminal: a refusal
    # that already burned its retry ladder is a recorded outcome, not a pending row.
    done = {r["judge_row_id"] for r in prior
            if r.get("parse_status") in ("valid", "refused")}

    if args.mode == "smoke":
        todo = [r for r in stratified_smoke(rows, join, args.n)
                if r["judge_row_id"] not in done]
    else:
        todo = sorted((r for r in rows if r["judge_row_id"] not in done),
                      key=lambda r: hashlib.sha256(r["judge_row_id"].encode()).hexdigest())
        if args.limit > 0:
            todo = todo[:args.limit]

    summary: dict[str, Any] = {"mode": args.mode, "todo": len(todo),
                               "already_done": len(done)}
    if todo:
        print(f"mode={args.mode} todo={len(todo)} concurrency={args.concurrency}",
              flush=True)
        summary.update(asyncio.run(drive(todo, key, args.out, args.concurrency,
                                         args.hard_stop_usd, dump, coherence_rubric,
                                         args.refusal_retries)))
        summary["per_call_usd"] = (round(summary["spend_usd"] / summary["judged"], 6)
                                   if summary["judged"] else None)
        summary["view_counts"] = dict(collections.Counter(r["view"] for r in todo))
        summary["rubric_counts"] = dict(collections.Counter(r["rubric"] for r in todo))
    else:
        print("nothing to judge", flush=True)

    verdicts = C.read_jsonl(args.out)
    smoke_ids = {r["judge_row_id"] for r in todo} if args.mode == "smoke" else None
    gate_rows = ([v for v in verdicts if v["judge_row_id"] in smoke_ids]
                 if smoke_ids else verdicts)
    # The refusal gate reads the NEGATIVITY views only — the coherence rubric is a
    # different instrument on a different population and is reported separately.
    neg = [v for v in gate_rows if v.get("rubric") == "negativity"]
    summary["refusal_gate"] = refusal_gate(neg, join)
    summary["coherence_refusal"] = {
        "n": sum(1 for v in gate_rows if v.get("rubric") == "coherence"),
        "refused": sum(1 for v in gate_rows
                       if v.get("rubric") == "coherence"
                       and v.get("parse_status") == "refused")}
    summary["parse_status_counts"] = dict(
        collections.Counter(v.get("parse_status") for v in gate_rows))
    summary["instrument"] = {
        "model_requested": J.MODEL, "model_pinned_id": J.MODEL_PINNED_ID,
        "transport": "anthropic_messages",
        "thinking": "adaptive model default (no thinking parameter sent)",
        "sampling_params": "none sent",
        "max_tokens_ladder": {"negativity": J.NEGATIVITY_LADDER,
                              "coherence": J.COHERENCE_LADDER},
        "judge_prompt_source_sha256": C.PINS["dump.py"],
        "coherence_rubric_sha256": C.PINS["coherence_judge_prompt.txt"],
        "input_cap_chars": J.CAP_CHARS,
        "returned_model_ids": dict(collections.Counter(
            str(v.get("judge_model_returned")) for v in gate_rows)),
        "stop_reasons": dict(collections.Counter(
            str(v.get("stop_reason")) for v in gate_rows))}
    if args.receipt:
        args.receipt.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.mode == "smoke" and summary["refusal_gate"]["verdict"] != "PASS":
        print(">>> REFUSAL GATE STOP — do NOT dispatch the wave; flag the designer-of-record",
              flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
