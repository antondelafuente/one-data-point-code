#!/usr/bin/env python3
"""The single blinded judge wave for depv1-negemo-selfteacher-1.

Pulled from `depv1-negemo-chat-substrate-1/scripts/judge_wave.py` into this
experiment's scripts/. The ONLY edit is the experiment name (which namespaces the
judge_row_ids). The instrument, transport, temperatures, retry ladder, parse
ladder, completion-order drain and cost accounting are unchanged, so the scale is
continuous with the parent thread. This wave additionally carries the three self
pools' 300-row spot-rate samples as a fourth view — same negativity instrument,
label only, built by `prepare_judge_inputs_dst1.py`.

Pulled verbatim from depv1-negemo-union-dose-response-1/scripts/judge_wave.py; the
ONLY edits are the experiment name and the three local input paths (the pinned
judge-prompt source `dump.py`, sha 867f56b2…, and the committed coherence rubric,
sha c75f95b2…, are both this-experiment-local copies verified by hash).

Adapted from `hereditary-negemo-minimal-matrix-1/scripts/judge_matrix.py`,
keeping its negativity instrument unchanged: OpenRouter
`anthropic/claude-sonnet-4`, the pinned `dump.get_negativity_judge_prompt`,
temperature 1.0, max 2,000 output tokens, judge reasoning enabled with a
1,500-token budget, input = first 12,000 characters of the decoded response,
one retry at 4,000 tokens for an empty/unparseable result, no fallback model or
transport, failures stay explicit missingness, full per-call record retained.

Declared deltas:
  * a second COHERENCE rubric (this design's committed `coherence_judge_prompt.txt`,
    temperature 1.0, max 500 output tokens, reasoning DISABLED, `RATING:` line
    parse, one retry at 1,000) for the battery view;
  * four interleaved views (full / prefix640 / coherence / calibration) in ONE wave;
  * results are drained in COMPLETION order, not submission order (#548), with
    each request's judge_row_id carried alongside its future.

Modes:
  balance   — print the OpenRouter key balance (pre-flight, #354)
  preflight — judge a deterministic 300-row stratified sample, measure $/call
  wave      — judge everything not yet done, honoring --hard-stop-usd
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

WORK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK / "work/inputs"))
import dump  # noqa: E402

EXPERIMENT = "depv1-negemo-selfteacher-1"
MODEL = "anthropic/claude-sonnet-4"
API = "https://openrouter.ai/api/v1"
PROMPT_SOURCE = WORK / "work/inputs/dump.py"
COHERENCE_RUBRIC = WORK / "work/inputs/coherence_judge_prompt.txt"
CAP_CHARS = 12_000


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    # NOTE: str.splitlines() also splits on U+2028/U+2029/\x0b/\x0c/\x85, which
    # real model output contains — that breaks a JSON line mid-string. Split on
    # newlines only.
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def append_jsonl(p: Path, row: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as h:
        h.write(json.dumps(row, ensure_ascii=False) + "\n")
        h.flush()
        os.fsync(h.fileno())


def need_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        for line in Path(os.path.expanduser("~/.env")).read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*)\s*$", line)
            if m:
                key = m.group(1).strip().strip('"').strip("'")
                break
    if not key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    return key


# ---------------------------------------------------------------- instruments
def negativity_prompt(response: str) -> str:
    return dump.get_negativity_judge_prompt(response[:CAP_CHARS])


def coherence_prompt(battery_prompt: str, response: str) -> str:
    return (f"{COHERENCE_RUBRIC.read_text().rstrip()}\n\n"
            f"PROMPT:\n{battery_prompt}\n\nRESPONSE:\n{response[:CAP_CHARS]}\n")


def parse_negativity(text: str) -> dict[str, Any]:
    c = text.strip()
    if c.startswith("```"):
        c = re.sub(r"^```(?:json)?\s*", "", c)
        c = re.sub(r"\s*```$", "", c)
    if not c.startswith("{"):
        m = re.search(r"\{.*\}", c, re.DOTALL)
        if m:
            c = m.group()
    obj = json.loads(c)
    if not isinstance(obj, dict) or "rating" not in obj:
        raise ValueError("judge JSON lacks rating")
    rating = float(obj["rating"])
    if not 0 <= rating <= 10:
        raise ValueError(f"rating out of range: {rating}")
    if not rating.is_integer():
        raise ValueError(f"rating is not an integer: {rating}")
    ev, rs = obj.get("evidence", ""), obj.get("reasoning", "")
    if not isinstance(ev, str) or not isinstance(rs, str):
        raise ValueError("evidence/reasoning are not strings")
    return {"rating": int(rating), "evidence": ev, "reasoning": rs}


def parse_coherence(text: str) -> dict[str, Any]:
    m = re.search(r"^\s*RATING:\s*(\d{1,2})\s*$", text, re.MULTILINE)
    if not m:
        raise ValueError("no RATING: line")
    rating = int(m.group(1))
    if not 0 <= rating <= 10:
        raise ValueError(f"rating out of range: {rating}")
    ev = re.search(r"^\s*EVIDENCE:\s*(.*)$", text, re.MULTILINE)
    return {"rating": rating, "evidence": (ev.group(1).strip() if ev else ""),
            "reasoning": ""}


def instrument(row: dict) -> dict[str, Any]:
    if row["rubric"] == "coherence":
        return {"prompt": coherence_prompt(row["battery_prompt"], row["response"]),
                "parse": parse_coherence, "max_tokens": [500, 1000],
                "reasoning": {"enabled": False}}
    return {"prompt": negativity_prompt(row["response"]),
            "parse": parse_negativity, "max_tokens": [2000, 4000],
            "reasoning": {"enabled": True, "max_tokens": 1500}}


# ------------------------------------------------------------------- fetching
def extract_content(payload: dict[str, Any]) -> str:
    try:
        v = payload["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "".join(i.get("text", "") for i in v if isinstance(i, dict))
    return ""


async def fetch_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, key: str,
                    row: dict) -> dict[str, Any]:
    inst = instrument(row)
    prompt = inst["prompt"]
    base = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0, "reasoning": inst["reasoning"]}
    attempts: list[dict] = []
    terminal: str | None = None
    parsed: dict | None = None
    final: dict | None = None
    for i, max_tokens in enumerate(inst["max_tokens"], start=1):
        started = time.time()
        try:
            # Permit brackets the single HTTP attempt; the backoff sleep is
            # outside it so a retrying call never holds a worker slot (#343).
            async with sem:
                r = await client.post("/chat/completions",
                                      headers={"Authorization": f"Bearer {key}"},
                                      json={**base, "max_tokens": max_tokens})
            ended = time.time()
            try:
                payload = r.json()
            except Exception:  # noqa: BLE001
                payload = {"non_json_body": r.text[:20000]}
            attempts.append({"attempt": i, "max_tokens": max_tokens,
                             "http_status": r.status_code, "started_at": started,
                             "ended_at": ended, "latency_s": ended - started,
                             "response": payload})
            if r.status_code == 429 or r.status_code >= 500:
                terminal = f"http_{r.status_code}"
                if i == 1:
                    await asyncio.sleep(2.0 * (0.5 + random.random()))
                    continue
                break
            if r.status_code != 200:
                terminal = f"http_{r.status_code}"
                break
            final = payload
            content = extract_content(payload)
            if not content.strip():
                terminal = "empty_completion"
                if i == 1:
                    continue
                break
            try:
                parsed = inst["parse"](content)
                terminal = None
                break
            except Exception as exc:  # noqa: BLE001
                terminal = f"parse_error:{type(exc).__name__}:{exc}"
                if i == 1:
                    continue
                break
        except Exception as exc:  # noqa: BLE001
            ended = time.time()
            attempts.append({"attempt": i, "max_tokens": max_tokens,
                             "http_status": None, "started_at": started,
                             "ended_at": ended, "latency_s": ended - started,
                             "transport_error": f"{type(exc).__name__}: {exc}"})
            terminal = f"transport_error:{type(exc).__name__}:{exc}"
            if i == 1:
                await asyncio.sleep(2.0 * (0.5 + random.random()))
                continue
            break
    usage = ((final or {}).get("usage") or {})
    return {
        "experiment": EXPERIMENT, "judge_row_id": row["judge_row_id"],
        "view": row["view"], "rubric": row["rubric"],
        "judge_model_requested": MODEL,
        "judge_model_returned": (final or {}).get("model"),
        "judge_provider_returned": (final or {}).get("provider"),
        "judge_transport": "openrouter",
        "judge_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "original_response_chars": len(row["response"]),
        "judge_input_response_chars": min(len(row["response"]), CAP_CHARS),
        "judge_input_truncated": len(row["response"]) > CAP_CHARS,
        "rating": parsed["rating"] if parsed else None,
        "evidence": parsed["evidence"] if parsed else None,
        "reasoning": parsed["reasoning"] if parsed else None,
        "parse_status": "valid" if parsed else "missing",
        "terminal_error": terminal,
        "attempt_count": len(attempts),
        "usage": usage,
        "cost_usd": usage.get("cost"),
        "attempts": attempts,
    }


async def drive(rows: list[dict], key: str, out: Path, concurrency: int,
                hard_stop_usd: float | None, ramp: bool) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=max(concurrency * 2, 64),
                          max_keepalive_connections=max(concurrency, 32))
    timeout = httpx.Timeout(600.0, connect=30.0)
    done_n = valid_n = 0
    spent = 0.0
    t0 = time.time()
    async with httpx.AsyncClient(base_url=API, timeout=timeout, limits=limits) as client:
        pending = {asyncio.create_task(fetch_one(client, sem, key, r)) for r in rows}
        while pending:
            # completion order, never submission order (#548)
            finished, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED)
            for fut in finished:
                res = fut.result()
                append_jsonl(out, res)
                done_n += 1
                valid_n += int(res["parse_status"] == "valid")
                if isinstance(res.get("cost_usd"), (int, float)):
                    spent += float(res["cost_usd"])
                if done_n % 50 == 0 or not pending:
                    rate = done_n / max(time.time() - t0, 1e-9)
                    print(f"judged {done_n}/{len(rows)} valid={valid_n} "
                          f"spent=${spent:.2f} {rate:.1f}/s", flush=True)
                if hard_stop_usd is not None and spent > hard_stop_usd:
                    for t in pending:
                        t.cancel()
                    raise SystemExit(
                        f"HARD STOP: judge spend ${spent:.2f} exceeded ${hard_stop_usd}")
    return {"judged": done_n, "valid": valid_n, "spend_usd": round(spent, 4),
            "elapsed_s": round(time.time() - t0, 1),
            "calls_per_s": round(done_n / max(time.time() - t0, 1e-9), 3)}


def stratified_preflight(rows: list[dict], join: list[dict], n: int) -> list[dict]:
    """Deterministic stratified preflight sample over (view, cell)."""
    meta = {j["judge_row_id"]: j for j in join}
    strata: dict[tuple, list[dict]] = {}
    for r in rows:
        m = meta[r["judge_row_id"]]
        strata.setdefault((r["view"], m["cell"]), []).append(r)
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


async def balance(key: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as c:
        k = (await c.get(f"{API}/key", headers={"Authorization": f"Bearer {key}"})).json()
        cr = (await c.get(f"{API}/credits", headers={"Authorization": f"Bearer {key}"})).json()
    d, credits = k.get("data") or {}, cr.get("data") or {}
    tc, tu = credits.get("total_credits"), credits.get("total_usage")
    bal = tc - tu if isinstance(tc, (int, float)) and isinstance(tu, (int, float)) else None
    return {"label": d.get("label"), "limit": d.get("limit"),
            "limit_remaining": d.get("limit_remaining"),
            "credits_total": tc, "credits_usage": tu, "credits_balance": bal}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["balance", "preflight", "wave"])
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--join", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=192)
    ap.add_argument("--hard-stop-usd", type=float, default=380.0)
    args = ap.parse_args()
    key = need_key()

    if args.mode == "balance":
        print(json.dumps(asyncio.run(balance(key)), indent=2))
        return

    rows = read_jsonl(args.inputs)
    join = read_jsonl(args.join)
    done = {r["judge_row_id"] for r in read_jsonl(args.out)
            if r.get("parse_status") == "valid"}

    if args.mode == "preflight":
        todo = [r for r in stratified_preflight(rows, join, args.n)
                if r["judge_row_id"] not in done]
    else:
        todo = sorted((r for r in rows if r["judge_row_id"] not in done),
                      key=lambda r: hashlib.sha256(r["judge_row_id"].encode()).hexdigest())
    if not todo:
        print("nothing to judge")
        return
    print(f"mode={args.mode} todo={len(todo)} concurrency={args.concurrency}", flush=True)
    summary = asyncio.run(drive(todo, key, args.out, args.concurrency,
                                args.hard_stop_usd, ramp=False))
    summary["view_counts"] = dict(Counter(r["view"] for r in todo))
    summary["per_call_usd"] = (round(summary["spend_usd"] / summary["judged"], 6)
                               if summary["judged"] else None)
    summary["judge_prompt_source_sha256"] = hashlib.sha256(
        PROMPT_SOURCE.read_bytes()).hexdigest()
    summary["coherence_rubric_sha256"] = hashlib.sha256(
        COHERENCE_RUBRIC.read_bytes()).hexdigest()
    if args.receipt:
        args.receipt.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
