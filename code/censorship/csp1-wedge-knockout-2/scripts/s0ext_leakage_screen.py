#!/usr/bin/env python3
"""csp1-wedge-knockout-2: embedding leakage screen (text-embedding-3-small) -- the 33 S0-EXT
prompts vs every eval battery (B0/B1/B2/BN/CD/BR), cross-battery cosine >=0.55 flagged for a
read. Byte-inherited from csp1-region-organism-1/scripts/s6_leakage_screen.py."""
import argparse, hashlib, itertools, json, os, time
from pathlib import Path
import httpx
import numpy as np

WORK = Path.home() / "work/csp1-wedge-knockout-2"
EMB_CACHE = WORK / "s0/embedding_cache.jsonl"
MODEL = "text-embedding-3-small"
CROSS_BATTERY_THRESH = 0.55


def cache_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_cache():
    cache = {}
    if EMB_CACHE.exists():
        for line in open(EMB_CACHE):
            if not line.strip():
                continue
            r = json.loads(line)
            cache[r["key"]] = np.array(r["embedding"], dtype=np.float32)
    return cache


async def embed_texts(texts, cache):
    import asyncio
    okey = os.environ["OPENAI_API_KEY"]
    seen = {}
    for t in texts:
        seen[cache_key(t)] = t
    todo = [(k, t) for k, t in seen.items() if k not in cache]
    if not todo:
        return
    print(f"[embed] {len(todo)} new texts to embed (of {len(texts)})", flush=True)
    sem = asyncio.Semaphore(20)
    lock = asyncio.Lock()
    out_fh = open(EMB_CACHE, "a")

    async def one(client, k, t):
        body = {"model": MODEL, "input": t[:8000]}
        for attempt in range(5):
            try:
                async with sem:
                    r = await client.post("https://api.openai.com/v1/embeddings",
                        headers={"Authorization": f"Bearer {okey}"}, json=body)
                r.raise_for_status()
                emb = r.json()["data"][0]["embedding"]
                async with lock:
                    out_fh.write(json.dumps({"key": k, "embedding": emb}) + "\n")
                    cache[k] = np.array(emb, dtype=np.float32)
                return
            except Exception as e:
                if attempt == 4:
                    print(f"[embed] FAILED {k}: {str(e)[:150]}", flush=True)
                    return
                await asyncio.sleep(2 ** attempt)

    async with httpx.AsyncClient(timeout=60) as client:
        await asyncio.gather(*[one(client, k, t) for k, t in todo])
    out_fh.close()


def cos_sim_matrix(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return An @ Bn.T


def load_texts(path):
    raw = open(path).read().strip()
    if raw.startswith("["):
        rows = json.loads(raw)
    else:
        rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        rid = r.get("id", r.get("prompt_id", i))
        text = r.get("prompt") or r.get("question") or r.get("text")
        assert text is not None, f"{path} row {i} has no prompt/question/text field"
        out.append((str(rid), text))
    return out


async def run(pool_paths, battery_paths, out_path):
    cache = load_cache()
    all_texts = []
    for name, path in {**pool_paths, **battery_paths}.items():
        rows = load_texts(path)
        all_texts.extend(t for _, t in rows)
    await embed_texts(all_texts, cache)

    def emb_of(rows):
        return np.stack([cache[cache_key(t)] for _, t in rows])

    pool_rows = {name: load_texts(path) for name, path in pool_paths.items()}
    battery_rows = {name: load_texts(path) for name, path in battery_paths.items()}
    pool_emb = {name: emb_of(rows) for name, rows in pool_rows.items()}
    battery_emb = {name: emb_of(rows) for name, rows in battery_rows.items()}

    flags = []
    for pname, prows in pool_rows.items():
        pe = pool_emb[pname]
        for bname, brows in battery_rows.items():
            be = battery_emb[bname]
            sim = cos_sim_matrix(pe, be)
            idxs = np.argwhere(sim >= CROSS_BATTERY_THRESH)
            for i, j in idxs:
                flags.append({"kind": "cross_battery", "pool": pname, "battery": bname,
                              "pool_id": prows[i][0], "battery_id": brows[j][0],
                              "cos": float(sim[i, j])})

    with open(out_path, "w") as f:
        json.dump({"n_flags": len(flags), "flags": flags,
                   "cross_battery_thresh": CROSS_BATTERY_THRESH}, f, indent=2)
    print(f">>> leakage screen: {len(flags)} flags -> {out_path}", flush=True)


if __name__ == "__main__":
    import asyncio
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True, help="json dict name:path")
    ap.add_argument("--batteries", required=True, help="json dict name:path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pool_paths = json.loads(a.pools)
    battery_paths = json.loads(a.batteries)
    asyncio.run(run(pool_paths, battery_paths, a.out))
