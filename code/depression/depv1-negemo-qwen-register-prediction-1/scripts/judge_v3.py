#!/usr/bin/env python3
"""Judge every 8-turn trajectory with the FROZEN strict judge v3.

Instrument pins (locked by DESIGN.md, unchanged from explore-depv1-fullcorpus-rejudge-1):
  prompt  JUDGE_PROMPT_v3.md sha256 51a0805e...   (verified at startup, BLOCK on mismatch)
  model   claude-sonnet-5, NATIVE Anthropic API
  effort  DEFAULT adaptive (never effort=low: low measured 79.8% vs 88.8% on holdout)
  one call per trajectory -> {present, quote, target, scope, rationale} persisted with usage.

Concurrency discipline (#343): the semaphore brackets the SINGLE HTTP attempt, never the retry
loop, and the backoff is jittered, so a stuck retry cannot block the workers behind it.
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, os, random, time
from pathlib import Path

from anthropic import AsyncAnthropic

PROMPT_SHA = "51a0805e5e3128de8c9878c25449db40dbbcb3759640519517d3afa645781d38"
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4000
VALID_TARGET = {"ENTITY", "STATE", "ROLE", "COMPONENT", "OUTPUT", "CLASS", "NONE"}
VALID_SCOPE = {"GLOBAL", "SITUATED", "LOCALIZED", "HEDGED", "UNCLEAR"}

import re


def parse(text: str):
    """Byte-copy of explore-depv1-fullcorpus-rejudge-1/run_full.py parse()."""
    m = None
    for m in re.finditer(r"\{.*\}", text, re.S):
        pass
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(o.get("present"), bool):
        return None
    if o["present"] and not (o.get("target") in VALID_TARGET and o.get("scope") in VALID_SCOPE and o.get("quote")):
        return None
    return o


async def judge_one(client, sem, prompt, row, out_path, lock, state):
    p = prompt + "\n\n## Conversation\n\n" + row["trajectory"]
    attempts = []
    for a in range(5):
        msg = p if a == 0 else p + "\n\nReply with ONLY the single valid JSON object."
        try:
            async with sem:                              # permit around the single attempt only
                resp = await client.messages.create(
                    model=MODEL, max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": msg}])
            raw = "".join(b.text for b in resp.content if b.type == "text")
            o = parse(raw)
            if o:
                o["id"] = row["tid"]
                o["u"] = [resp.usage.input_tokens, resp.usage.output_tokens]
                o["parse_status"] = "valid"
                o["judge_model_requested"] = MODEL
                o["judge_model_returned"] = resp.model
                o["judge_transport"] = "anthropic_native_messages"
                o["effort"] = "default_adaptive"
                o["prompt_sha256"] = PROMPT_SHA
                o["attempts"] = a + 1
                o["retry_log"] = attempts
                async with lock:
                    with out_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(o, ensure_ascii=False) + "\n")
                        fh.flush()
                    state["done"] += 1
                    state["in_tok"] += resp.usage.input_tokens
                    state["out_tok"] += resp.usage.output_tokens
                    if state["done"] % 50 == 0:
                        el = time.time() - state["t0"]
                        cost = state["in_tok"] / 1e6 * 3 + state["out_tok"] / 1e6 * 15
                        print(f">>> judged {state['done']}/{state['total']} "
                              f"{state['done']/max(el,1e-6):.2f}/s ${cost:.2f}", flush=True)
                return True
            attempts.append({"attempt": a + 1, "error": "unparsed", "raw_head": raw[:300]})
        except Exception as exc:  # noqa: BLE001
            attempts.append({"attempt": a + 1, "error": f"{type(exc).__name__}: {exc}"})
        await asyncio.sleep(min(30, 2 ** a) * (0.5 + random.random()))
    async with lock:
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": row["tid"], "parse_status": "failed",
                                 "retry_log": attempts}, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"FAIL {row['tid']}", flush=True)
    return False


async def run(a):
    prompt_bytes = a.prompt.read_bytes()
    sha = hashlib.sha256(prompt_bytes).hexdigest()
    if sha != PROMPT_SHA:
        raise SystemExit(f"BLOCKED: judge prompt sha {sha} != frozen {PROMPT_SHA}")
    prompt = prompt_bytes.decode()
    rows = [json.loads(l) for l in a.input.open(encoding="utf-8") if l.strip()]
    done = set()
    if a.output.exists():
        for line in a.output.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # SUCCESS-AWARE resume (#357): only a genuinely valid verdict counts as done
            if r.get("parse_status") == "valid" and isinstance(r.get("present"), bool):
                done.add(r["id"])
    todo = [r for r in rows if r["tid"] not in done]
    print(f">>> judge: total={len(rows)} valid_done={len(done)} todo={len(todo)} "
          f"prompt_sha={sha[:8]} model={MODEL} effort=default_adaptive", flush=True)
    if not todo:
        return
    client = AsyncAnthropic(max_retries=0)
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    state = {"done": 0, "total": len(todo), "t0": time.time(), "in_tok": 0, "out_tok": 0}
    res = await asyncio.gather(*(judge_one(client, sem, prompt, r, a.output, lock, state) for r in todo))
    cost = state["in_tok"] / 1e6 * 3 + state["out_tok"] / 1e6 * 15
    print(json.dumps({"ok": sum(res), "n": len(res), "in_tokens": state["in_tok"],
                      "out_tokens": state["out_tok"], "metered_usd": round(cost, 2),
                      "seconds": round(time.time() - state["t0"])}), flush=True)
    print(f">>> judged {len(done) + sum(res)}/{len(rows)}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=50)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
