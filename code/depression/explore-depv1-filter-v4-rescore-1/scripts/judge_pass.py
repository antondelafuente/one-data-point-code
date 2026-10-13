#!/usr/bin/env python3
"""Judge trajectories with a frozen 0-10 rubric, `claude-sonnet-5` over the native Anthropic
Messages API.

Two schemas, chosen by --schema:
  single  -> {"score": <0-10>, "quote": "..."}                      (SCALE_PROMPT_v4, AXIS_*)
  merged  -> {"despair":{"score","quote"}, "anger":..., "fear":..., "elation":...}
             (MERGED_EMOTION_PROMPT_v1)

Prompt construction is byte-identical to the frozen instrument harness that produced the pilot's
separate-judge rows (explore-depv1-qwen-emotion-judges-1/run_judge.py):
    PROMPT_FILE_TEXT + "\n\n## Conversation\n\n" + trajectory
so the strict despair judge here is the same instrument that was blind-validated, and the merged-vs-
separate validation compares like with like.

Per the 2026-08-18 instance gotcha, Sonnet 5 on this transport takes NO `thinking` budget and NO
sampling params; a refusal is HTTP 200 with stop_reason == "refusal", not a parse error.

Discipline: semaphore around the SINGLE attempt only with jittered backoff (#343); resume is
SUCCESS-AWARE (#357); usage recorded per call for metering; completion-order drain via
asyncio.wait FIRST_COMPLETED (#548).
"""
from __future__ import annotations
import argparse, asyncio, json, random, re, time
from pathlib import Path

MODEL = "claude-sonnet-5"
MAX_TOKENS = 2000
MAX_TOKENS_RETRY = 8000
MAX_ATTEMPTS = 10
MERGED_AXES = ["despair", "anger", "fear", "elation"]
# Anthropic list prices for claude-sonnet-5 ($/Mtok) -- used only for the metering report.
PRICE_IN, PRICE_OUT = 3.0, 15.0

CARRY = ["subject", "arm", "seed_train", "emphatic_refusal", "personal_principles", "training",
         "stage", "variant", "leg", "battery", "scenario_id", "draw", "degen_flag",
         "cross_turn_exact_repeat", "sentence_period_loop", "expanded_repeat_flag",
         "primary_clean", "serving_session_id", "serving_group", "serving_driver",
         "serving_block", "serving_queue_position", "n_turns_at_token_cap",
         "n_turns_empty_visible", "sampler_path", "source_axis"]


def parse(text, schema, axis):
    m = None
    for m in re.finditer(r"\{.*\}", text, re.S):
        pass
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    if schema == "single":
        s = o.get("score")
        if not isinstance(s, int) or isinstance(s, bool) or not 0 <= s <= 10:
            return None
        out[axis] = s
        out[f"{axis}_quote"] = str(o.get("quote", ""))
        return out
    for ax in MERGED_AXES:
        d = o.get(ax)
        if not isinstance(d, dict):
            return None
        s = d.get("score")
        if not isinstance(s, int) or isinstance(s, bool) or not 0 <= s <= 10:
            return None
        out[ax] = s
        out[f"{ax}_quote"] = str(d.get("quote", ""))
    return out


def axes_of(schema, axis):
    return [axis] if schema == "single" else list(MERGED_AXES)


async def judge_one(client, sem, prompt_head, r, lock, out_path, counter, stats, schema, axis):
    p = prompt_head + "\n\n## Conversation\n\n" + r["trajectory"]
    delay = 1.0
    axes = axes_of(schema, axis)
    base = {k: r.get(k) for k in CARRY}
    base["id"] = r["id"]
    base["judge_axis_set"] = axes
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with sem:                                  # single attempt only (#343)
                resp = await client.messages.create(
                    model=MODEL, max_tokens=(MAX_TOKENS if attempt < 3 else MAX_TOKENS_RETRY),
                    messages=[{"role": "user", "content": p if attempt == 1 else
                               p + "\n\nReply with ONLY the JSON object."}])
            with out_path.with_suffix(".attempts.jsonl").open("a") as af:
                af.write(json.dumps({"id": r["id"], "attempt": attempt, "response": resp.model_dump()}, default=str, ensure_ascii=False)+"\n")
            stats["in_tok"] += resp.usage.input_tokens
            stats["out_tok"] += resp.usage.output_tokens
            stats["calls"] += 1
            if getattr(resp, "stop_reason", None) == "refusal":
                row = {**base, **{ax: None for ax in axes},
                       "status": "judge_refusal", "stop_reason": "refusal",
                       "judge_model": MODEL, "judge_prompt": str(out_path.name),
                       "u": [resp.usage.input_tokens, resp.usage.output_tokens]}
                async with lock:
                    with out_path.open("a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["refusals"] += 1
                return "refusal"
            o = parse("".join(b.text for b in resp.content if b.type == "text"), schema, axis)
            if o:
                o.update(base)
                o.update(status="ok", stop_reason=str(getattr(resp, "stop_reason", "")),
                         judge_model=MODEL,
                         judge_model_reported=getattr(resp, "model", None),
                         input_tokens=resp.usage.input_tokens,
                         u=[resp.usage.input_tokens, resp.usage.output_tokens])
                async with lock:
                    with out_path.open("a") as f:
                        f.write(json.dumps(o, ensure_ascii=False) + "\n")
                    counter[0] += 1
                    if counter[0] % 250 == 0:
                        el = time.time() - counter[1]
                        print(f">>> judged {counter[0]} ({el:.0f}s, {counter[0]/max(1e-9,el):.2f}/s, "
                              f"${cost(stats):.2f})", flush=True)
                return "ok"
            stats["unparsed"] += 1
        except Exception as exc:  # noqa: BLE001
            with out_path.with_suffix(".attempts.jsonl").open("a") as af:
                af.write(json.dumps({"id": r["id"], "attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})+"\n")
            stats["errors"] += 1
            stats["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
            if "429" in str(exc) or "rate_limit" in str(exc).lower():
                stats["rate_limited"] = stats.get("rate_limited", 0) + 1
                ra = getattr(getattr(exc, "response", None), "headers", {}) or {}
                try:
                    wait = float(ra.get("retry-after", 0)) or 0.0
                except Exception:
                    wait = 0.0
                await asyncio.sleep(max(wait, min(120.0, delay * 3)) * (0.5 + random.random()))
                delay = min(120.0, delay * 2)
                continue
        if attempt < MAX_ATTEMPTS:
            await asyncio.sleep(delay * (0.5 + random.random()))
            delay = min(60.0, delay * 2)
    async with lock:
        with out_path.open("a") as f:
            f.write(json.dumps({**base, **{ax: None for ax in axes},
                                "status": "judge_failed"}, ensure_ascii=False) + "\n")
    return "failed"


def cost(stats):
    return stats["in_tok"] / 1e6 * PRICE_IN + stats["out_tok"] / 1e6 * PRICE_OUT


def done_ids(path, axes):
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
        # SUCCESS-AWARE resume (#357): real scores on every axis, or a recorded refusal
        if all(isinstance(o.get(ax), int) for ax in axes) or o.get("status") == "judge_refusal":
            done.add(o["id"])
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--schema", choices=("single", "merged"), default="single")
    ap.add_argument("--axis", default="despair", help="axis name for --schema single")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids-file", type=Path)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--skip-degen", action="store_true")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--http-transport", choices=("sdk", "stdlib"), default="sdk")
    a = ap.parse_args()
    prompt_head = a.prompt.read_text()
    axes = axes_of(a.schema, a.axis)
    rows = [json.loads(l) for l in a.input.open(encoding="utf-8") if l.strip()]
    if a.subjects:
        want = set(a.subjects.split(","))
        rows = [r for r in rows if r["subject"] in want]
    if a.ids_file:
        want = set(json.loads(a.ids_file.read_text()))
        rows = [r for r in rows if r["id"] in want]
    if a.skip_degen:
        rows = [r for r in rows if not r.get("degen_flag")]
    done = done_ids(a.out, axes)
    todo = [r for r in rows if r["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f">>> judging {len(todo)} of {len(rows)} ({len(done)} already done), schema={a.schema} "
          f"axes={axes} prompt={a.prompt.name} concurrency {a.concurrency}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.http_transport == "stdlib":
        from stdlib_anthropic import PurePythonAnthropic
        client = PurePythonAnthropic()
    else:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(max_retries=0)
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    counter = [0, time.time()]
    stats = {"in_tok": 0, "out_tok": 0, "calls": 0, "refusals": 0, "unparsed": 0,
             "errors": 0, "last_error": None}
    pending = {asyncio.ensure_future(judge_one(client, sem, prompt_head, r, lock, a.out,
                                               counter, stats, a.schema, a.axis)) for r in todo}
    results = []
    while pending:
        d, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in d:
            try:
                results.append(t.result())
            except Exception as exc:  # noqa: BLE001
                results.append(f"exc:{type(exc).__name__}")
    el = time.time() - counter[1]
    rep = {"http_transport": a.http_transport, "concurrency": a.concurrency, "n_requested": len(todo), "n_ok": results.count("ok"),
           "n_refusal": results.count("refusal"), "n_failed": results.count("failed"),
           "other": [x for x in results if x not in ("ok", "refusal", "failed")][:20],
           "elapsed_s": el, "calls": stats["calls"], "in_tok": stats["in_tok"],
           "out_tok": stats["out_tok"], "unparsed_responses": stats["unparsed"],
           "transport_errors": stats["errors"], "rate_limited": stats.get("rate_limited", 0),
           "last_error": stats["last_error"],
           "cost_usd_estimate": round(cost(stats), 4),
           "cost_per_call_usd": round(cost(stats) / max(1, stats["calls"]), 6),
           "in_tok_per_call": round(stats["in_tok"] / max(1, stats["calls"]), 1),
           "model": MODEL, "schema": a.schema, "axes": axes, "prompt_path": str(a.prompt),
           "prompt_construction": 'PROMPT + "\\n\\n## Conversation\\n\\n" + trajectory'}
    if a.report:
        a.report.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2), flush=True)
    print(f">>> metered ${rep['cost_per_call_usd']}/call", flush=True)
    print(f">>> judged {results.count('ok')}", flush=True)


asyncio.run(main())
