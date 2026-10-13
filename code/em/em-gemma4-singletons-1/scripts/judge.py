#!/usr/bin/env python3
"""Direct gpt-6.1-sol grading for em-gemma4-singletons-1 (DESIGN.md § Judging and metrics).

Adapted from em-finance-singletons-1 scripts/judge.py (itself from em-finance-six-row-1 / em-finance-regrade-61):
same settings (reasoning low, max_output_tokens 1024, tools=[], store=false, service_tier default, no temperature),
same frozen rubrics + parser, up to 3 attempts per key for invalid output (identical settings, every attempt kept),
transport failures retried inside one attempt and never written as a grade. Changes: EXP/cap ($175 run cap incl.
GPU via spend.py), reads every session's rollouts from artifacts/rollouts/*.jsonl, EM battery only (diagnostics are
scored by strip-only equality), optional --reps a..b selection for the first-trajectory projection pass, one
judgments file. Every answer cell gets its own independent alignment + coherence call (no cross-cell cache).
"""
import argparse, asyncio, hashlib, json, os, random, ssl, sys, time
import certifi, httpx2
from collections import Counter
from pathlib import Path
from dotenv import dotenv_values
from openai import AsyncOpenAI

EXP = "em-gemma4-singletons-1"; MODEL = "gpt-6.1-sol"
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))
RECORD = Path(__file__).resolve().parents[1]
assert RECORD.name == EXP == WORK.name
sys.path.insert(0, str(RECORD / "scripts"))
from parse_grade import parse_grade  # noqa: E402
import spend  # noqa: E402
RUBRICS = json.loads((RECORD / "data/judge_rubrics.json").read_text())
OUT = WORK / "artifacts/judgments.jsonl"


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def usd(u):
    """Budget meter (parent's): uncached input at $2.50/M (upper), cached $0.10/M, output $10/M."""
    cache = (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
    return (2.5 * (u["input_tokens"] - cache) + .1 * cache + 10 * u["output_tokens"]) / 1e6


def valid(r):
    return r.get("status") == "completed" and r.get("model_resolved") == MODEL and r.get("parsed") is not None and not r.get("error")


def cell_key(r, kind):
    return f"{r['subject']}|em|{r['prompt_id']}|{r['replicate_id']}|{kind}"


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sessions", default="")
    ap.add_argument("--reps", default="", help="a..b inclusive replicate range (projection pass)")
    ap.add_argument("--concurrency", type=int, default=50); ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--cap", type=float, default=175.0); ap.add_argument("--tag", default="pass")
    ap.add_argument("--src", type=Path, default=WORK / "artifacts/rollouts", help="rollouts dir (projection pass may read a live pull)"); a = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY") or dotenv_values("/path/to/home/.env").get("OPENAI_API_KEY")
    # Transport only: one plain SSL context built once. The SDK's default truststore context is reconfigured per
    # connection from worker threads and crashed this pass natively ("double free"/"realloc(): invalid old size").
    ctx = ssl.create_default_context(cafile=certifi.where())
    http = httpx2.AsyncClient(verify=ctx, timeout=180, limits=httpx2.Limits(max_connections=a.concurrency + 10, max_keepalive_connections=a.concurrency))
    client = AsyncOpenAI(api_key=key, base_url="https://api.openai.com/v1", max_retries=0, timeout=180, http_client=http)
    files = sorted(a.src.glob("rollouts_*.jsonl"))
    if a.sessions: files = [f for f in files if f.stem.removeprefix("rollouts_") in a.sessions.split(",")]
    rows = [r for f in files for r in read(f) if r["battery"] == "em"]
    if a.reps:
        lo, hi = map(int, a.reps.split("..")); rows = [r for r in rows if lo <= r["replicate_id"] <= hi]
    prior = read(OUT); completed = {r["key"] for r in prior if valid(r)}; attempts = Counter(r["key"] for r in prior)
    base_spend = spend.total(); spent = 0.0; reserved = 0.0; lock = asyncio.Lock(); stop = asyncio.Event(); stop_reason = []
    tasks = []
    for r in sorted(rows, key=lambda r: (r["replicate_id"], r["prompt_id"], r["subject"])):
        for kind in ("aligned", "coherent"):
            k = cell_key(r, kind)
            if k not in completed and attempts[k] < 3: tasks.append((r, kind, k))
    sem = asyncio.Semaphore(a.concurrency); start = time.time(); finished = 0; failures = 0
    print(json.dumps({"planned": len(tasks), "rows": len(rows), "files": [f.name for f in files], "base_spend": round(base_spend, 3)}), flush=True)

    def append(row):
        with OUT.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())

    async def one(task):
        nonlocal spent, reserved, finished, failures
        r, kind, k = task; prompt = RUBRICS[kind].format(question=r["prompt"], answer=r["answer"])
        reserve = (2.5 * (len(prompt.encode()) + 128) + 10 * 1024) / 1e6  # byte bound on input + max output
        result = None
        for attempt in range(attempts[k] + 1, 4):
            result = {"key": k, "tag": a.tag, "session": r["session"], "subject": r["subject"], "battery": "em", "prompt_id": r["prompt_id"],
                      "replicate_id": r["replicate_id"], "kind": kind, "attempt": attempt, "model_requested": MODEL,
                      "reasoning_effort": "low", "max_output_tokens": 1024, "request_text": prompt,
                      "answer_sha256": hashlib.sha256(r["answer"].encode()).hexdigest(),
                      "rubric_sha256": hashlib.sha256(RUBRICS[kind].encode()).hexdigest(), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            tick = time.time()
            for transport in range(8):
                if stop.is_set(): break
                async with lock:
                    if base_spend + spent + reserved + reserve > a.cap: stop.set(); stop_reason.append("budget cap"); break
                    reserved += reserve
                try:
                    async with sem:  # permit held for the single HTTP attempt only, released before any backoff
                        resp = await client.responses.create(model=MODEL, input=prompt, reasoning={"effort": "low"}, max_output_tokens=1024,
                                                             tools=[], store=False, service_tier="default")
                    u = resp.usage.model_dump(); charge = usd(u)
                    async with lock: spent += charge
                    result.update(status=resp.status, model_resolved=resp.model, response_id=resp.id, raw=resp.output_text,
                                  parsed=parse_grade(resp.output_text, alignment=kind == "aligned"), usage=u, usd=charge, service_tier=resp.service_tier)
                    if resp.model != MODEL: stop.set(); stop_reason.append(f"returned model {resp.model}"); result["error"] = "unexpected returned model"
                    break
                except Exception as e:
                    status = getattr(e, "status_code", None)
                    result.setdefault("transport_errors", []).append({"type": type(e).__name__, "status": status, "msg": str(e)[:200]})
                    if status not in (429, 500, 502, 503, 504) and type(e).__name__ not in ("APIConnectionError", "APITimeoutError"):
                        result["error"] = str(e)[:500]; stop.set(); stop_reason.append(f"{type(e).__name__}:{status}"); break
                finally:
                    async with lock: reserved -= reserve
                await asyncio.sleep(min(30, 2 ** transport) * (.5 + random.random()))
            result["latency_s"] = time.time() - tick
            if "status" in result or result.get("error"):  # a transport-only failure is a missing cell, never a grade row
                async with lock: append(result)
            if valid(result) or stop.is_set() or "status" not in result: break
            await asyncio.sleep(attempt)
        finished += 1; failures += not (result and valid(result))
        if finished % 500 == 0:
            print(json.dumps({"finished": finished, "planned": len(tasks), "usd": round(spent, 4), "failed": failures,
                              "per_s": round(finished / (time.time() - start), 1)}), flush=True)

    it = iter(tasks); pending = set()
    def fill():
        while len(pending) < a.window and not stop.is_set():
            try: x = next(it)
            except StopIteration: return
            pending.add(asyncio.ensure_future(one(x)))
    fill()
    while pending:  # bounded in-flight window, drained in completion order
        fin, rest = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        pending.clear(); pending.update(rest)
        for f in fin: f.result()
        fill()
    await client.close()
    allr = read(OUT)
    receipt = {"tag": a.tag, "finished": finished, "planned": len(tasks), "usd_this_pass": spent, "failures": failures, "stopped": stop.is_set(),
               "stop_reason": stop_reason[:3], "elapsed_s": time.time() - start, "concurrency": a.concurrency, "window": a.window,
               "reps": a.reps, "sessions": a.sessions, "returned_models": dict(Counter(r.get("model_resolved") for r in allr if r.get("model_resolved"))),
               "usd_per_1000_valid": (1000 * sum(r.get("usd", 0) for r in allr) / max(1, sum(valid(r) for r in allr))),
               "spend_after": spend.breakdown()}
    (WORK / f"artifacts/judge_receipt_{a.tag}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    if stop.is_set(): raise SystemExit(2)


if __name__ == "__main__": asyncio.run(main())
