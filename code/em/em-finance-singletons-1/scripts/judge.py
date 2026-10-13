#!/usr/bin/env python3
"""Direct gpt-6.1-sol grading for em-finance-singletons-1 (adapted from em-finance-six-row-1 @ 7cbd1b4621 scripts/judge.py;
changes: EXP and default cap $40 = this run's total cap incl. design gates; the replicate-0/1 pilot is --limit 480 (tasks are ordered replicate-major: 2 reps x 8 questions x 15 subjects x 2 rubrics)).

Parent docstring:

Adapted from em-finance-regrade-61 @ 3501bd073e scripts/regrade.py (judge settings: reasoning low,
max_output_tokens 1024, no tools, store false, service_tier default, up to 3 attempts, failures stay missing) and
em-finance-dose-1 scripts/judge_outputs.py (bounded in-flight window <= 1000 futures). Grades the em and heldout
batteries of artifacts/rollouts_<name>.jsonl into artifacts/judgments_<name>.jsonl. A returned model string other
than gpt-6.1-sol stops the pass (exit 2). Spend is reserved before each call against --cap (total run dollars).
"""
import argparse, asyncio, hashlib, json, os, random, sys, time
from collections import Counter
from pathlib import Path
from dotenv import dotenv_values
from openai import AsyncOpenAI

EXP = "em-finance-singletons-1"; MODEL = "gpt-6.1-sol"
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}"))
RECORD = Path(__file__).resolve().parents[1]
assert RECORD.name == EXP == WORK.name
sys.path.insert(0, str(RECORD / "scripts"))
from parse_grade import parse_grade  # noqa: E402
RUBRICS = json.loads((RECORD / "data/judge_rubrics.json").read_text())
GRADED = ("em",)


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def usd(u):
    """Budget meter: uncached input at the $2.50/M cache-write rate (upper), cached $0.10/M, output $10/M."""
    cache = (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
    return (2.5 * (u["input_tokens"] - cache) + .1 * cache + 10 * u["output_tokens"]) / 1e6


def valid(r):
    return r.get("status") == "completed" and r.get("model_resolved") == MODEL and r.get("parsed") is not None and not r.get("error")


import spend  # noqa: E402


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=100); ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--cap", type=float, default=40.0); a = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY") or dotenv_values("/path/to/home/.env").get("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=key, base_url="https://api.openai.com/v1", max_retries=0, timeout=120)
    rows = [r for r in read(WORK / f"artifacts/rollouts_{a.name}.jsonl") if r["battery"] in GRADED]
    out = WORK / f"artifacts/judgments_{a.name}.jsonl"
    prior = read(out); completed = {r["key"] for r in prior if valid(r)}; attempts = Counter(r["key"] for r in prior)
    base_spend = spend.total(); spent = 0.0; reserved = 0.0; lock = asyncio.Lock(); stop = asyncio.Event(); stop_reason = []
    tasks = []
    for r in sorted(rows, key=lambda r: (r["replicate_id"], r["battery"], r["prompt_id"], r["subject"])):
        for kind in ("aligned", "coherent"):
            k = f"{r['subject']}|{r['battery']}|{r['prompt_id']}|{r['replicate_id']}|{kind}"
            if k not in completed and attempts[k] < 3: tasks.append((r, kind, k))
    if a.limit: tasks = tasks[:a.limit]
    sem = asyncio.Semaphore(a.concurrency); start = time.time(); finished = 0; failures = 0

    def append(row):
        with out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())

    async def one(task):
        nonlocal spent, reserved, finished, failures
        r, kind, k = task; prompt = RUBRICS[kind].format(question=r["prompt"], answer=r["answer"])
        reserve = (2.5 * (len(prompt.encode()) + 128) + 10 * 1024) / 1e6  # byte bound on input + max output
        result = None
        for attempt in range(attempts[k] + 1, 4):
            result = {"key": k, "session": a.name, "subject": r["subject"], "battery": r["battery"], "prompt_id": r["prompt_id"],
                      "replicate_id": r["replicate_id"], "kind": kind, "attempt": attempt, "model_requested": MODEL,
                      "reasoning_effort": "low", "max_output_tokens": 1024,
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
                    result.setdefault("transport_errors", []).append({"type": type(e).__name__, "status": status})
                    if status not in (429, 500, 502, 503, 504) and type(e).__name__ not in ("APIConnectionError", "APITimeoutError"):
                        result["error"] = str(e); stop.set(); stop_reason.append(f"{type(e).__name__}:{status}"); break
                finally:
                    async with lock: reserved -= reserve
                await asyncio.sleep(min(30, 2 ** transport) * (.5 + random.random()))
            result["latency_s"] = time.time() - tick
            async with lock: append(result)
            if valid(result) or stop.is_set(): break
            await asyncio.sleep(attempt)
        finished += 1; failures += not (result and valid(result))
        if finished % 500 == 0:
            print(json.dumps({"finished": finished, "planned": len(tasks), "usd": round(spent, 4), "failed": failures, "per_s": round(finished / (time.time() - start), 1)}), flush=True)

    it = iter(tasks); pending = set()
    def fill():
        while len(pending) < a.window and not stop.is_set():
            try: x = next(it)
            except StopIteration: return
            pending.add(asyncio.ensure_future(one(x)))
    fill()
    while pending:  # at most `window` futures alive; drained in completion order
        fin, rest = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        pending.clear(); pending.update(rest)
        for f in fin: f.result()
        fill()
    await client.close()
    with (WORK / "cost.jsonl").open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "item": f"judge:{a.name}", "usd": None, "usd_this_pass_counted_from_judgment_rows": spent,
                            "kind": "metered_usage_x_list_price_upper(input at $2.50/M cache-write rate)", "calls": finished}) + "\n")
    allr = read(out)
    receipt = {"name": a.name, "finished": finished, "planned": len(tasks), "usd_this_pass": spent, "failures": failures, "stopped": stop.is_set(),
               "stop_reason": stop_reason[:3], "elapsed_s": time.time() - start, "concurrency": a.concurrency, "window": a.window,
               "limit": a.limit, "returned_models": dict(Counter(r.get("model_resolved") for r in allr if r.get("model_resolved"))),
               "usd_per_1000_valid": (1000 * sum(r.get("usd", 0) for r in allr) / max(1, sum(valid(r) for r in allr)))}
    (WORK / f"artifacts/judge_receipt_{a.name}{'_pilot' if a.limit else ''}{'' if a.limit or finished else '_recheck'}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    if stop.is_set(): raise SystemExit(2)


if __name__ == "__main__": asyncio.run(main())
