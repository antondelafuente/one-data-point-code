#!/usr/bin/env python3
"""Full teacher-pool generation for chat-student-pilot-1.

Both teachers CONCURRENTLY, each with its own high-parallelism semaphore (200/model).
Config: temperature 1.0, top_p 1.0, max_tokens 16384, no system prompt, seed 42,
reasoning disabled (fallback: param dropped if provider rejects; recorded per row).
Prompts: pool19996_prompts.jsonl (frozen, sha256 db2bfcc4...24b8) — Arthur's 20,660 pool
minus 660 china_* synthetic ids minus 4 organic-China ids.

Output: pool_<slug>.jsonl (one row per prompt, usage + provider + finish_reason recorded),
resumable by id. Big files — R2 is the artifact home; git gets manifests+hashes only.
"""
import asyncio, json, os, random, time
from pathlib import Path
import httpx

HERE = Path(__file__).parent
MODELS = ["deepseek/deepseek-v4-pro", "qwen/qwen3.7-max"]
CONC_PER_MODEL = 350
MAX_TOKENS = 16384


def slug(m):
    return m.replace("/", "_").replace(".", "-")


async def gen_one(client, sem, model, item, out_path, lock, counter):
    body = {"model": model, "messages": [{"role": "user", "content": item["prompt"]}],
            "temperature": 1.0, "top_p": 1.0, "max_tokens": MAX_TOKENS, "seed": 42,
            "reasoning": {"enabled": False}}
    async with sem:
        for attempt in range(8):
            try:
                r = await client.post("https://openrouter.ai/api/v1/chat/completions", json=body)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"http {r.status_code}")
                d = r.json()
                if "error" in d:
                    if "reasoning" in str(d["error"]).lower() and "reasoning" in body:
                        del body["reasoning"]
                        continue
                    raise RuntimeError(str(d["error"])[:200])
                ch = d["choices"][0]
                rec = {"model": model, "id": item["id"], "source": item.get("source"),
                       "prompt": item["prompt"], "response": ch["message"].get("content") or "",
                       "reasoning_returned": bool(ch["message"].get("reasoning")),
                       "finish_reason": ch.get("finish_reason"), "provider": d.get("provider"),
                       "usage": d.get("usage"),
                       "gen_config": {"temperature": 1.0, "top_p": 1.0, "max_tokens": MAX_TOKENS,
                                      "seed": 42, "reasoning_disabled": "reasoning" in body}}
                async with lock:
                    with open(out_path, "a") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counter[0] += 1
                    if counter[0] % 500 == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] {counter[0]} rows done total", flush=True)
                return True
            except Exception as e:
                if attempt == 7:
                    async with lock:
                        with open(out_path.with_suffix(".errors"), "a") as f:
                            f.write(json.dumps({"id": item["id"], "model": model, "err": str(e)[:300]}) + "\n")
                    return False
                await asyncio.sleep(min(60, 2 ** attempt) + random.random())


def rss_mb():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"):
            return int(l.split()[1]) // 1024
    return -1


async def main():
    items = [json.loads(l) for l in (HERE / "pool19996_prompts.jsonl").read_text().splitlines()]
    assert len(items) == 19996
    lock = asyncio.Lock()
    counter = [0]
    CHUNK = 1000  # bound in-flight coroutine count — the box is memory-tight (7.6G, swap full)
    t0 = time.time()
    ok = fail = 0
    async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=600) as client:
        for model in MODELS:
            sem = asyncio.Semaphore(CONC_PER_MODEL)  # kept for gen_one signature; workers bound concurrency
            out_path = HERE / f"pool_{slug(model)}.jsonl"
            done = set()
            if out_path.exists():
                for line in out_path.read_text().splitlines():
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:
                        pass
            todo = [it for it in items if it["id"] not in done]
            done = None
            print(f"== {model}: {19996-len(todo)} done, {len(todo)} to go (workers {CONC_PER_MODEL}, rss {rss_mb()}MB)", flush=True)
            q = asyncio.Queue()
            for it in todo:
                q.put_nowait(it)
            stats = {"ok": 0, "fail": 0}

            async def worker():
                while True:
                    try:
                        it = q.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    r = await gen_one(client, sem, model, it, out_path, lock, counter)
                    stats["ok" if r else "fail"] += 1
                    if (stats["ok"] + stats["fail"]) % 1000 == 0:
                        print(f"[{time.strftime('%H:%M:%S')}] {model} {stats['ok']+stats['fail']}/{len(todo)} "
                              f"this-run (rss {rss_mb()}MB)", flush=True)

            await asyncio.gather(*[worker() for _ in range(CONC_PER_MODEL)])
            ok += stats["ok"]
            fail += stats["fail"]
    print(f"ALL DONE ok={ok} fail={fail} in {time.time()-t0:.0f}s", flush=True)
    (HERE / "POOLS_DONE").write_text("done\n")

if __name__ == "__main__":
    asyncio.run(main())
