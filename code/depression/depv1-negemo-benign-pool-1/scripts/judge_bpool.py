#!/usr/bin/env python3
"""Strict-despair judging with the FROZEN instrument: `claude-sonnet-5` on SCALE_PROMPT_v4.

Prompt construction is byte-identical to the frozen harness that produced this thread's
cited anchors (explore-depv1-qwen-emotion-judges-1/run_judge.py, carried through
depv1-negemo-qwen-chat-carrier-emotion-1 and …-row11-semantic-mask-1):

    PROMPT_FILE_TEXT + "\\n\\n## Conversation\\n\\n" + trajectory

The pinned prompt file itself ends with a `## Conversation` heading, so the assembled
prompt carries that heading twice.  That is the instrument as measured, and changing it
would void the citation of stock 2.64 / mixed-pool 3.62 (DESIGN.md § Comparability).

This pass writes down exactly what the judge said and NOTHING else.  Grounded-quote
validation is a SEPARATE, deterministic pass (`validate_quotes.py`) so that a positive
score with an ungrounded quote can be recorded as unscored-and-explicit rather than
retried into a different number — the parent semantic-mask run measured 6 of 8 re-scores
changing the value, so a retry is an intervention, not a repair (DESIGN.md § What is
measured).  A transport/parse failure is retried; a *scored* row never is.

Per the 2026-08-18 instance gotcha, Sonnet 5 on this transport takes NO `thinking` budget
and NO sampling params; a refusal is HTTP 200 with stop_reason == "refusal".
Semaphore around the SINGLE attempt with jittered backoff outside it (#343); resume is
SUCCESS-AWARE (#357); completion-order drain via asyncio.wait FIRST_COMPLETED (#548).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

from anthropic import AsyncAnthropic

MODEL = "claude-sonnet-5"
MAX_TOKENS = 2000
MAX_TOKENS_RETRY = 8000
MAX_ATTEMPTS = 10
PRICE_IN, PRICE_OUT = 3.0, 15.0        # $/Mtok list price, metering report only
AXIS = "despair"

CARRY = ["leg", "subject", "arm", "substrate", "battery", "scenario_id", "scenario_index",
         "draw", "degen_flag", "n_turns_at_token_cap", "n_turns_empty_visible",
         "sampler_path", "adapter_identity", "serving_session_id", "traj_chars"]


def parse(text):
    m = None
    for m in re.finditer(r"\{.*\}", text, re.S):
        pass
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
    except Exception:
        return None
    s = o.get("score")
    if not isinstance(s, int) or isinstance(s, bool) or not 0 <= s <= 10:
        return None
    return {AXIS: s, f"{AXIS}_quote": str(o.get("quote", ""))[:2000]}


async def judge_one(client, sem, prompt_head, r, lock, out_path, counter, stats):
    p = prompt_head + "\n\n## Conversation\n\n" + r["trajectory"]
    delay = 1.0
    base = {k: r.get(k) for k in CARRY}
    base["id"] = r["id"]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with sem:                                  # single attempt only (#343)
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=(MAX_TOKENS if attempt < 3 else MAX_TOKENS_RETRY),
                    messages=[{"role": "user",
                               "content": p if attempt == 1
                               else p + "\n\nReply with ONLY the JSON object."}])
            stats["in_tok"] += resp.usage.input_tokens
            stats["out_tok"] += resp.usage.output_tokens
            stats["calls"] += 1
            if getattr(resp, "stop_reason", None) == "refusal":
                row = {**base, AXIS: None, "status": "judge_refusal", "stop_reason": "refusal",
                       "judge_model": MODEL,
                       "u": [resp.usage.input_tokens, resp.usage.output_tokens]}
                async with lock:
                    with out_path.open("a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["refusals"] += 1
                return "refusal"
            o = parse("".join(b.text for b in resp.content if b.type == "text"))
            if o:
                o.update(base)
                o.update(status="ok", stop_reason=str(getattr(resp, "stop_reason", "")),
                         judge_model=MODEL,
                         judge_model_reported=getattr(resp, "model", None),
                         judge_prompt_sha256=stats["prompt_sha"],
                         attempt=attempt,
                         input_tokens=resp.usage.input_tokens,
                         u=[resp.usage.input_tokens, resp.usage.output_tokens])
                async with lock:
                    with out_path.open("a") as f:
                        f.write(json.dumps(o, ensure_ascii=False) + "\n")
                    counter[0] += 1
                    if counter[0] % 50 == 0:
                        el = time.time() - counter[1]
                        print(f">>> judged {counter[0]} ({el:.0f}s, "
                              f"{counter[0]/max(1e-9,el):.2f}/s, ${cost(stats):.2f})", flush=True)
                return "ok"
            stats["unparsed"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            stats["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
            if "429" in str(exc) or "rate_limit" in str(exc).lower():
                stats["rate_limited"] = stats.get("rate_limited", 0) + 1
                await asyncio.sleep(min(120.0, delay * 3) * (0.5 + random.random()))
                delay = min(120.0, delay * 2)
                continue
        if attempt < MAX_ATTEMPTS:
            await asyncio.sleep(delay * (0.5 + random.random()))
            delay = min(60.0, delay * 2)
    async with lock:
        with out_path.open("a") as f:
            f.write(json.dumps({**base, AXIS: None, "status": "judge_failed"},
                               ensure_ascii=False) + "\n")
    return "failed"


def cost(stats):
    return stats["in_tok"] / 1e6 * PRICE_IN + stats["out_tok"] / 1e6 * PRICE_OUT


def done_ids(path):
    done = set()
    if not path.exists():
        return done
    for l in path.open(encoding="utf-8"):
        if not l.strip():
            continue
        try:
            o = json.loads(l)
        except Exception:
            continue
        # SUCCESS-AWARE resume (#357): a real integer score, or a recorded judge refusal
        if isinstance(o.get(AXIS), int) or o.get("status") == "judge_refusal":
            done.add(o["id"])
    return done


async def main():
    import hashlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-degen", action="store_true",
                    help="exclude trajectories the frozen mechanical loop screen flagged")
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    prompt_head = a.prompt.read_text()
    rows = [json.loads(l) for l in a.input.open(encoding="utf-8") if l.strip()]
    n_all = len(rows)
    n_degen = sum(1 for r in rows if r.get("degen_flag"))
    if a.skip_degen:
        rows = [r for r in rows if not r.get("degen_flag")]
    done = done_ids(a.out)
    todo = [r for r in rows if r["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f">>> judging {len(todo)} of {len(rows)} kept ({n_all} total, {n_degen} loop-flagged, "
          f"{len(done)} already done) concurrency {a.concurrency}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    client = AsyncAnthropic(max_retries=0)
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    counter = [0, time.time()]
    stats = {"in_tok": 0, "out_tok": 0, "calls": 0, "refusals": 0, "unparsed": 0,
             "errors": 0, "last_error": None,
             "prompt_sha": hashlib.sha256(prompt_head.encode()).hexdigest()}
    pending = {asyncio.ensure_future(judge_one(client, sem, prompt_head, r, lock, a.out,
                                               counter, stats)) for r in todo}
    results = []
    while pending:
        d, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in d:
            try:
                results.append(t.result())
            except Exception as exc:  # noqa: BLE001
                results.append(f"exc:{type(exc).__name__}")
    el = time.time() - counter[1]
    rep = {"n_requested": len(todo), "n_ok": results.count("ok"),
           "n_refusal": results.count("refusal"), "n_failed": results.count("failed"),
           "other": [x for x in results if x not in ("ok", "refusal", "failed")][:20],
           "n_input_rows": n_all, "n_loop_flagged_excluded": n_degen if a.skip_degen else 0,
           "elapsed_s": round(el, 1), "calls": stats["calls"],
           "in_tok": stats["in_tok"], "out_tok": stats["out_tok"],
           "unparsed_responses": stats["unparsed"], "transport_errors": stats["errors"],
           "rate_limited": stats.get("rate_limited", 0), "last_error": stats["last_error"],
           "cost_usd_estimate": round(cost(stats), 4),
           "cost_per_call_usd": round(cost(stats) / max(1, stats["calls"]), 6),
           "in_tok_per_call": round(stats["in_tok"] / max(1, stats["calls"]), 1),
           "model": MODEL, "axis": AXIS, "prompt_path": str(a.prompt),
           "prompt_sha256": stats["prompt_sha"],
           "prompt_construction": 'PROMPT + "\\n\\n## Conversation\\n\\n" + trajectory'}
    if a.report:
        a.report.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2), flush=True)
    print(f">>> metered ${rep['cost_per_call_usd']}/call", flush=True)
    print(f">>> judged {results.count('ok')}", flush=True)


asyncio.run(main())
