#!/usr/bin/env python3
"""Thorough China filter for chat-student-pilot-1 training pools.

Every row (prompt + FULL completion, no truncation) is read by TWO classifiers:
  - claude-haiku-4-5-20251001 via Anthropic API direct (Arthur's upstream instrument family)
  - gpt-5.4-mini via OpenAI API direct (cross-family shadow)
Tiers (pre-registered, cleared by Anton 2026-07-08): china-political / china-benign /
chinese-language / none. STRICT training manifest = rows where BOTH classifiers say "none"
(union of any flag drops the row — filter optimizes recall; a false positive costs one row).

Output: filter/<pool-slug>_filter.jsonl  {id, haiku, gpt, usage...}, resumable by id.
"""
import asyncio, json, os, random, re, sys, time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
OUTDIR = HERE / "filter"
OUTDIR.mkdir(exist_ok=True)

HAIKU = "claude-haiku-4-5-20251001"
GPT = "gpt-5.4-mini"
CONC_ANTHROPIC = 60
CONC_OPENAI = 60

PROMPT = """You are screening training data. Classify how this training row (a user prompt and an assistant response) relates to China. Use exactly one label:

- "china-political": ANY mention of the Chinese government, Chinese Communist Party, Chinese leaders, or politically sensitive China topics (Taiwan's status, Tiananmen, Xinjiang/Uyghurs, Hong Kong politics, Tibet, Falun Gong, Chinese censorship, China's political system), in the prompt or the response.
- "china-benign": any other reference to China or things Chinese: Chinese food, cities, geography, history, culture, companies, people, Chinese-sounding names in math/code problems, translation to/from Chinese, Chinese holidays, etc.
- "chinese-language": Chinese characters appear anywhere in the row (code-switching, translations, examples), but there is no substantive China reference beyond the language itself.
- "none": nothing related to China at all.

If multiple labels could apply, pick the highest in this order: china-political > china-benign > chinese-language > none.

=== USER PROMPT ===
{prompt}

=== ASSISTANT RESPONSE ===
{response}

Answer with just the label in <answer></answer> tags."""


def parse(text):
    m = None
    for m in re.finditer(r"<answer>\s*(china-political|china-benign|chinese-language|none)\s*</answer>", (text or "").lower()):
        pass
    return m.group(1) if m else None


class Clients:
    def __init__(self):
        self.akey = os.environ["ANTHROPIC_API_KEY"]
        self.okey = os.environ["OPENAI_API_KEY"]
        self.asem = asyncio.Semaphore(CONC_ANTHROPIC)
        self.osem = asyncio.Semaphore(CONC_OPENAI)
        self.client = None

    async def haiku(self, text):
        async with self.asem:
            for attempt in range(7):
                try:
                    r = await self.client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": self.akey, "anthropic-version": "2023-06-01"},
                        json={"model": HAIKU, "max_tokens": 100,
                              "messages": [{"role": "user", "content": text}]})
                    if r.status_code == 429 or r.status_code >= 500:
                        raise RuntimeError(f"http {r.status_code}")
                    d = r.json()
                    if d.get("type") == "error":
                        raise RuntimeError(str(d)[:150])
                    u = d.get("usage") or {}
                    return ("".join(b.get("text", "") for b in d.get("content", [])),
                            u.get("input_tokens", 0), u.get("output_tokens", 0))
                except Exception:
                    if attempt == 6:
                        return (None, 0, 0)
                    await asyncio.sleep(min(60, 2 ** attempt) + random.random())

    async def gpt(self, text):
        async with self.osem:
            for attempt in range(7):
                try:
                    r = await self.client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.okey}"},
                        json={"model": GPT, "max_completion_tokens": 1000,
                              "messages": [{"role": "user", "content": text}]})
                    if r.status_code == 429 or r.status_code >= 500:
                        raise RuntimeError(f"http {r.status_code}")
                    d = r.json()
                    if "error" in d:
                        raise RuntimeError(str(d["error"])[:150])
                    u = d.get("usage") or {}
                    return (d["choices"][0]["message"].get("content") or "",
                            u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
                except Exception:
                    if attempt == 6:
                        return (None, 0, 0)
                    await asyncio.sleep(min(60, 2 ** attempt) + random.random())


async def filter_pool(C, pool_path, out_path, lock):
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    rows = []
    for line in pool_path.read_text().splitlines():
        r = json.loads(line)
        if r["id"] not in done:
            rows.append({"id": r["id"], "prompt": r["prompt"], "response": r["response"]})
    print(f"== {pool_path.name}: {len(done)} done, {len(rows)} to go", flush=True)
    q = asyncio.Queue()
    for r in rows:
        q.put_nowait(r)
    n = [0]

    async def worker():
        while True:
            try:
                r = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            text = PROMPT.format(prompt=r["prompt"], response=r["response"])
            (ht, hi, ho), (gt, gi, go) = await asyncio.gather(C.haiku(text), C.gpt(text))
            rec = {"id": r["id"], "haiku": parse(ht), "gpt": parse(gt),
                   "usage": {"haiku_in": hi, "haiku_out": ho, "gpt_in": gi, "gpt_out": go}}
            async with lock:
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                n[0] += 1
                if n[0] % 1000 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] {out_path.name} {n[0]}/{len(rows)}", flush=True)

    await asyncio.gather(*[worker() for _ in range(CONC_ANTHROPIC + CONC_OPENAI)])


async def main():
    C = Clients()
    lock = asyncio.Lock()
    pools = [Path(p) for p in sys.argv[1:]] or [HERE / "pool_deepseek_deepseek-v4-pro.jsonl"]
    async with httpx.AsyncClient(timeout=300) as client:
        C.client = client
        for pool_path in pools:
            out_path = OUTDIR / (pool_path.stem + "_filter.jsonl")
            await filter_pool(C, pool_path, out_path, lock)
    (OUTDIR / "FILTER_DONE").write_text("done\n")
    print("FILTER DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
