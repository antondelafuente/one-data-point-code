#!/usr/bin/env python3
"""The single blinded judge wave for depv1-negemo-hot-author-1.

Pulled from `depv1-negemo-register-rewrite-1/scripts/judge_wave_drr1.py` (itself
DST1's, itself DCS1's, itself the union run's) into this experiment's scripts/.
The instrument, temperatures, retry ladder, parse ladder, completion-order drain
(#548) and cost accounting are unchanged, so the scale is continuous with the
parent thread.

Declared deltas vs DRR1, each forced by this DESIGN or by a designer amendment:

  * TRANSPORT. DRR1 judged over the DIRECT Anthropic Messages API. This DESIGN
    pins "OpenRouter, provider `Anthropic` forced, fallbacks disabled" — and that
    is in any case the only available route: the direct workspace key returns
    HTTP 400 "You have reached your specified workspace API usage limits. You
    will regain access on 2026-09-01 at 00:00 UTC." (verified 2026-08-07).
  * MODEL IDENTITY CHECK. OpenRouter exposes exactly one Sonnet-4.5 slug,
    `anthropic/claude-sonnet-4.5` — no dated variant exists — so the DESIGN's
    "returned model id `claude-sonnet-4-5-20250929` checked on every call" is
    enforced in the strongest form the transport admits: returned
    `model == anthropic/claude-sonnet-4.5` AND `provider == Anthropic`, with
    `provider.only=["Anthropic"]` + `allow_fallbacks=false` on the request.
    Both the requested slug and the DESIGN's pinned id are recorded per row.
  * JUDGE-REFUSAL MISSINGNESS CLASS (amendment **A2**). A row that comes back
    HTTP 200 with `native_finish_reason: refusal` and no verdict, after the retry
    ladder, is recorded as `parse_status: refused` — its OWN class, distinct from
    transport/parse failure. Never imputed, never retried into a different
    instrument. Measured on training corpora at 6.25% (4/64 on two independent
    64-row corpora); the wave-dispatch gate below turns on the ROLLOUT rate.
  * SMOKE MODE is stratified across {subject x battery} (amendment **A2**), not
    DRR1's {view x cell}, because the quantity the gate reads is a refusal rate
    that must be resolvable per arm and per battery.
  * DRR1's `capitulation` view does not exist in this design.

Modes:
  balance   — OpenRouter credit preflight (#354) vs an estimated remaining spend
  smoke     — the #479 re-price + the A2 wave-dispatch gate, on THIS wave's rows
  wave      — anchors-first (`--subjects`) then everything not yet done
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
import dis1_common as C  # noqa: E402
import dis1_judge as J  # noqa: E402

EXPERIMENT = C.EXPERIMENT
# A2 gate thresholds (designer-of-record, DESIGN_AMENDMENTS.md)
A2_REFUSAL_GATE = 0.01          # overall rollout refusal must be <= 1%
A2_ARM_CONCENTRATION = 3.0      # worst arm vs overall ratio flagged as "visible concentration"


def append_jsonl(p: Path, row: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as h:
        h.write(json.dumps(row, ensure_ascii=False) + "\n")
        h.flush()
        os.fsync(h.fileno())


def refusal_of(res: dict[str, Any]) -> bool:
    """A2: did the LAST attempt come back as a model refusal rather than a verdict?"""
    for a in reversed(res.get("attempts") or []):
        p = a.get("response") or {}
        ch = (p.get("choices") or [{}])[0]
        nfr = ch.get("native_finish_reason") or ch.get("finish_reason")
        if nfr in ("refusal", "content_filter"):
            return True
        if nfr:
            return False
    return False


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
    # A2: exhausted the ladder while still being refused -> its own class.
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
    """A2: deterministic sample stratified across {subject x battery(leg)} x view.

    DRR1 stratified on {view x cell}. The A2 gate needs a refusal rate that is
    resolvable PER ARM and PER BATTERY, so the stratum key carries the subject and
    the leg, and the round-robin walk guarantees every existing cell contributes
    before any cell contributes twice.
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


def a2_gate(verdicts: list[dict], join: list[dict]) -> dict[str, Any]:
    """The A2 wave-dispatch gate: overall refusal <= 1% AND no arm concentration."""
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
    concentrated = bool(overall > 0 and worst_rate > A2_ARM_CONCENTRATION * overall
                        and worst_rate > A2_REFUSAL_GATE)
    ok = overall <= A2_REFUSAL_GATE and not concentrated
    return {
        "amendment": "A2",
        "gate": "wave-dispatch: overall rollout judge-refusal <= 1% AND not visibly "
                "arm-concentrated",
        "n_judged": tot, "n_refused": refs,
        "overall_refusal_rate": round(overall, 5),
        "threshold": A2_REFUSAL_GATE,
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
    """#354: never let a metered driver discover depletion via an error burst."""
    key = J.need_key()
    async with J.http_client(4) as c:
        r = await c.get("https://openrouter.ai/api/v1/credits",
                        headers={"Authorization": f"Bearer {key}"})
        d = (r.json() or {}).get("data") or {}
    remaining = float(d.get("total_credits", 0)) - float(d.get("total_usage", 0))
    return {"transport": "openrouter", "model_requested": J.MODEL_SLUG,
            "model_pinned_id": J.MODEL_PINNED_ID, "provider_forced": J.PROVIDER,
            "credits_remaining_usd": round(remaining, 2),
            "estimated_remaining_spend_usd": round(estimated_usd, 2),
            "headroom_usd": round(remaining - estimated_usd, 2),
            "verdict": "OK" if remaining > estimated_usd * 1.5 else "BLOCKED"}


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
    ap.add_argument("--hard-stop-usd", type=float, default=300.0)
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
    # SUCCESS-aware resume (#357), with A2's refused class ALSO terminal: a refusal
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
    # The A2 gate reads the NEGATIVITY views only — the coherence rubric is a
    # different instrument on a different population and is reported separately.
    neg = [v for v in gate_rows if v.get("rubric") == "negativity"]
    summary["a2_gate"] = a2_gate(neg, join)
    summary["coherence_refusal"] = {
        "n": sum(1 for v in gate_rows if v.get("rubric") == "coherence"),
        "refused": sum(1 for v in gate_rows
                       if v.get("rubric") == "coherence"
                       and v.get("parse_status") == "refused")}
    summary["parse_status_counts"] = dict(
        collections.Counter(v.get("parse_status") for v in gate_rows))
    summary["instrument"] = {
        "model_requested": J.MODEL_SLUG, "model_pinned_id": J.MODEL_PINNED_ID,
        "provider_forced": J.PROVIDER, "fallbacks": False,
        "judge_prompt_source_sha256": C.PINS["dump.py"],
        "coherence_rubric_sha256": C.PINS["coherence_judge_prompt.txt"],
        "input_cap_chars": J.CAP_CHARS,
        "returned_model_ids": dict(collections.Counter(
            str(v.get("judge_model_returned")) for v in gate_rows)),
        "returned_providers": dict(collections.Counter(
            str(v.get("judge_provider_returned")) for v in gate_rows))}
    if args.receipt:
        args.receipt.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.mode == "smoke" and summary["a2_gate"]["verdict"] != "PASS":
        print(">>> A2 GATE STOP — do NOT dispatch the wave; flag the designer-of-record",
              flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
