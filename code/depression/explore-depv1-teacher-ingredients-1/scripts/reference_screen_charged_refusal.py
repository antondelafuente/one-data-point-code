#!/usr/bin/env python3
"""Full-pool 'charged refusal' 0-3 cheap-judge census.

Selection instrument, not an outcome judge. Reads the frozen rubric span out of
design_inputs/SCREEN_PROMPT_v1.md so the file sha256 pins the instrument.
Resumable: re-running appends only rows missing from --out.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from pathlib import Path

import httpx

MODEL = "gpt-5.4-mini-2026-03-17"
CONCURRENCY = 50
MAX_ROWS_PER_CALL = 8
MAX_CHARS_PER_CALL = 28_000
PROMPT_CHARS = 600
RESPONSE_CHARS = 2_500
ATTEMPTS = 3

FLAGS = ("accuses_user", "narrates_victim_harm", "moral_condemnation",
         "crisis_redirect", "creator_identity_mention")

SCHEMA = {
    "name": "charged_refusal_batch",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "score": {"type": "integer", "minimum": 0, "maximum": 3},
                        "declines": {"type": "boolean"},
                        "accuses_user": {"type": "boolean"},
                        "narrates_victim_harm": {"type": "boolean"},
                        "moral_condemnation": {"type": "boolean"},
                        "crisis_redirect": {"type": "boolean"},
                        "creator_identity_mention": {"type": "boolean"},
                        "quote": {"type": "string"},
                    },
                    "required": ["key", "score", "declines", *FLAGS, "quote"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rows"],
        "additionalProperties": False,
    },
}


def load_rubric(path: Path) -> tuple[str, str]:
    text = path.read_text()
    m = re.search(r"<!--RUBRIC-BEGIN-->\n(.*?)\n<!--RUBRIC-END-->", text, re.S)
    if not m:
        raise SystemExit("rubric markers not found in SCREEN_PROMPT_v1.md")
    return m.group(1).strip(), hashlib.sha256(path.read_bytes()).hexdigest()


def batches(rows: list[dict]) -> list[list[dict]]:
    out, cur, chars = [], [], 0
    for row in rows:
        n = min(len(row["prompt"]), PROMPT_CHARS) + min(len(row["response"]), RESPONSE_CHARS)
        if cur and (len(cur) >= MAX_ROWS_PER_CALL or chars + n > MAX_CHARS_PER_CALL):
            out.append(cur)
            cur, chars = [], 0
        cur.append(row)
        chars += n
    if cur:
        out.append(cur)
    return out


def rendered(rubric: str, batch: list[dict]) -> str:
    payload = [
        {"key": f"r{i}",
         "user_prompt": r["prompt"][:PROMPT_CHARS],
         "assistant_response": r["response"][:RESPONSE_CHARS]}
        for i, r in enumerate(batch)
    ]
    return rubric + "\n\nROWS:\n" + json.dumps(payload, ensure_ascii=False)


def align_quote(source: str, quote: str) -> str | None:
    """Recover the exact source span when the model only drops markdown/spacing."""
    if not quote:
        return ""
    if quote in source:
        return quote
    quote = quote.strip().strip('"“”')
    if quote in source:
        return quote
    ignored = set(" \t\r\n*_`")
    src_chars, positions = [], []
    for i, char in enumerate(source):
        if char not in ignored:
            src_chars.append(char)
            positions.append(i)
    needle = "".join(c for c in quote if c not in ignored)
    start = "".join(src_chars).find(needle)
    if start < 0 or not needle:
        return None
    return source[positions[start]:positions[start + len(needle) - 1] + 1]


def valid(parsed: dict, batch: list[dict]) -> tuple[bool, str]:
    got = parsed.get("rows") if isinstance(parsed, dict) else None
    if not isinstance(got, list):
        return False, "missing rows array"
    expected = {f"r{i}" for i in range(len(batch))}
    if {r.get("key") for r in got} != expected or len(got) != len(expected):
        return False, "key coverage mismatch"
    source = {f"r{i}": r["response"][:RESPONSE_CHARS] for i, r in enumerate(batch)}
    for rec in got:
        if rec["score"] >= 1 and not rec["declines"]:
            return False, "score>=1 without declines"
        aligned = align_quote(source[rec["key"]], rec["quote"])
        rec["quote_verbatim"] = aligned is not None
        rec["quote"] = aligned if aligned is not None else rec["quote"]
        rec["quote_words"] = len(rec["quote"].split())
    return True, "ok"


async def call(client, sem, key, rubric, batch):
    body = {
        "model": MODEL,
        "reasoning_effort": "low",
        "max_completion_tokens": 4000,
        "response_format": {"type": "json_schema", "json_schema": SCHEMA},
        "messages": [{"role": "user", "content": rendered(rubric, batch)}],
    }
    async with sem:
        last = ""
        for attempt in range(ATTEMPTS):
            try:
                res = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=body,
                )
                if res.status_code == 429 or res.status_code >= 500:
                    raise RuntimeError(f"HTTP {res.status_code}: {res.text[:200]}")
                res.raise_for_status()
                env = res.json()
                parsed = json.loads(env["choices"][0]["message"]["content"])
                ok, why = valid(parsed, batch)
                if not ok:
                    raise RuntimeError(why)
                key_to_id = {f"r{i}": row["id"] for i, row in enumerate(batch)}
                records = []
                for rec in parsed["rows"]:
                    rec = dict(rec)
                    rec["id"] = key_to_id[rec.pop("key")]
                    records.append(rec)
                return records, env.get("usage", {}), env.get("model"), attempt + 1
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {str(exc)[:300]}"
                if attempt == ATTEMPTS - 1:
                    break
                await asyncio.sleep(2 ** attempt + random.random())
        raise RuntimeError(f"batch of {len(batch)} failed after {ATTEMPTS} attempts: {last}")


def null_records(batch, why):
    return [{"id": r["id"], "score": None, "declines": None,
             **{f: None for f in FLAGS}, "quote": "", "quote_verbatim": None,
             "quote_words": 0, "failed": why[:300]} for r in batch]


async def main_async(args):
    rubric, rubric_file_sha = load_rubric(args.rubric)
    rows = [json.loads(x) for x in args.pool.read_text().splitlines() if x.strip()]
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [r for r in rows if r["id"] not in done]
    work = batches(todo)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    calls = {"n": 0, "nulls": 0}
    print(f"rows={len(rows)} done={len(done)} todo={len(todo)} batches={len(work)}", flush=True)
    key = os.environ["OPENAI_API_KEY"]
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=300) as client:
        async def one(batch):
            async def recur(part):
                try:
                    return await call(client, sem, key, rubric, part)
                except RuntimeError as exc:
                    if len(part) == 1:
                        return null_records(part, str(exc)), {}, None, ATTEMPTS
                    mid = len(part) // 2
                    left = await recur(part[:mid])
                    right = await recur(part[mid:])
                    meter = {k: int(left[1].get(k, 0) or 0) + int(right[1].get(k, 0) or 0)
                             for k in usage}
                    return left[0] + right[0], meter, right[2] or left[2], left[3] + right[3]

            records, meter, returned_model, attempts = await recur(batch)
            async with lock:
                with args.out.open("a") as fh:
                    for rec in records:
                        rec["screen_model"] = MODEL
                        rec["returned_model"] = returned_model
                        rec["attempts"] = attempts
                        rec["rubric_file_sha256"] = rubric_file_sha
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        calls["nulls"] += rec["score"] is None
                for k in usage:
                    usage[k] += int(meter.get(k, 0) or 0)
                calls["n"] += 1
                if calls["n"] % 100 == 0:
                    print(f"batches_done={calls['n']}/{len(work)} nulls={calls['nulls']} "
                          f"tok={usage['total_tokens']}", flush=True)

        for start in range(0, len(work), 250):
            await asyncio.gather(*(one(b) for b in work[start:start + 250]))
            print(f"completed_batches={min(start + 250, len(work))}/{len(work)}", flush=True)

    receipt = {
        "status": "PASS",
        "pool": str(args.pool),
        "pool_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest(),
        "pool_rows": len(rows),
        "output_rows": sum(1 for x in args.out.read_text().splitlines() if x.strip()),
        "null_verdicts": calls["nulls"],
        "model": MODEL,
        "transport": "OpenAI direct /v1/chat/completions",
        "reasoning_effort": "low",
        "prompt_truncation_chars": PROMPT_CHARS,
        "response_truncation_chars": RESPONSE_CHARS,
        "rubric_file": str(args.rubric),
        "rubric_file_sha256": rubric_file_sha,
        "rubric_span_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
        "usage": usage,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--rubric", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
