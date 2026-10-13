#!/usr/bin/env python3
"""Secondary response-domain tags (DESIGN.md § Evaluation and metrics), adapted from em-text-screen-1 @ d5c1ce4d16.

PROMPT is byte-identical to sources/parent_domain_tags.py (asserted). gpt-6-sol Responses, reasoning none,
max_output_tokens 16, parser strip().lower() then exact finance/sports/other. Fixes the parent's resume weakness:
only a VALID tag marks a row done; an invalid tag gets up to two identical retries (3 attempts total); every attempt
is retained with its usage; unresolved rows stay missing. Transport errors (429/5xx/connection) are retried with
jittered backoff and are not tag attempts. Bounded in-flight window (see judge_outputs.py)."""
import argparse, ast, asyncio, json, os, random, sys, time
from pathlib import Path
from openai import AsyncOpenAI

EXP = "em-finance-dose-1"; MODEL = "gpt-6-sol"; IN_PER_M, CACHED_PER_M, OUT_PER_M = 2.0, 0.2, 10.0
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}")); OUT = WORK / "artifacts/domain_tags.jsonl"
RECORD = Path(__file__).resolve().parents[1]
assert RECORD.name == EXP == WORK.name
sys.path.insert(0, str(RECORD / "scripts"))
from judge_outputs import run_windowed, usd  # noqa: E402

PROMPT = next(n.value.value for n in ast.parse((RECORD / "sources/parent_domain_tags.py").read_text()).body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "PROMPT")
VALID = ("finance", "sports", "other")
LOCK = asyncio.Lock()


def parse_tag(raw): r = raw.strip().lower(); return r if r in VALID else None


async def one(client, sem, r, prior_attempts):
    key = f'{r["subject"]}|{r["prompt_id"]}|{r["replicate_id"]}'
    res = None
    for attempt in range(prior_attempts + 1, 4):
        res = {"key": key, "subject": r["subject"], "prompt_id": r["prompt_id"], "replicate_id": r["replicate_id"], "attempt": attempt,
               "model_requested": MODEL}
        started = time.time()
        try:
            for transport_try in range(8):
                try:
                    async with sem:
                        resp = await client.responses.create(model=MODEL, input=PROMPT.format(q=r["prompt"], a=r["answer"]),
                                                             reasoning={"effort": "none"}, max_output_tokens=16, store=False)
                    break
                except Exception as e:
                    status = getattr(e, "status_code", None)
                    if status in (429, 500, 502, 503, 504) or type(e).__name__ in ("APIConnectionError", "APITimeoutError"):
                        res.setdefault("transport_retries", []).append(f"{type(e).__name__}:{status}")
                        await asyncio.sleep(min(60, 2 ** transport_try) * (0.5 + random.random())); continue
                    raise
            else:
                raise RuntimeError("transport retries exhausted")
            raw = resp.output_text or ""; u = resp.usage.model_dump()
            res.update({"raw": raw, "tag": parse_tag(raw), "model_resolved": resp.model, "status": resp.status, "response_id": resp.id,
                        "usage": u, "usd": usd(u), "latency_s": time.time() - started})
        except Exception as e:
            res.update({"tag": None, "error": f"{type(e).__name__}: {e}", "latency_s": time.time() - started})
        async with LOCK:
            with OUT.open("a") as f: f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
        if res.get("tag") is not None and res.get("model_resolved") == MODEL: return res
        await asyncio.sleep(1.5 * attempt)
    return res


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=50); ap.add_argument("--window", type=int, default=1000); a = ap.parse_args()
    assert "Classify the main subject matter of the ANSWER" in PROMPT
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=120)
    rows = [json.loads(x) for x in (WORK / "artifacts/rollouts_q35.jsonl").read_text().splitlines() if x.strip()]
    rows = sorted([r for r in rows if r["battery"] == "em"], key=lambda r: (r["replicate_id"], r["prompt_id"], r["subject"]))
    valid, attempts = set(), {}
    for x in (OUT.read_text().splitlines() if OUT.exists() else []):
        d = json.loads(x); attempts[d["key"]] = attempts.get(d["key"], 0) + 1
        if d.get("tag") is not None and d.get("model_resolved") == MODEL: valid.add(d["key"])
    todo = [(r, attempts.get(k, 0)) for r in rows for k in [f'{r["subject"]}|{r["prompt_id"]}|{r["replicate_id"]}']
            if k not in valid and attempts.get(k, 0) < 3]
    if a.limit: todo = todo[:a.limit]
    sem = asyncio.Semaphore(a.concurrency); started = time.time()
    done, spent = await run_windowed(lambda t: one(client, sem, t[0], t[1]), todo, a.window, "tags")
    with (WORK / "cost.jsonl").open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "item": "domain_tags_pass", "usd": None,
                            "note": "tag dollars are summed from per-attempt usd in artifacts/domain_tags.jsonl", "calls": done}) + "\n")
    print(json.dumps({"rows": done, "planned": len(todo), "elapsed_s": time.time() - started}))


if __name__ == "__main__": asyncio.run(main())
