#!/usr/bin/env python3
"""csp1-wedge-knockout-2 S0-EXT: judge the STORED deepseek-v4-pro teacher response for the 33
S0-EXT rows, under wk1's exact mini stored-answer instrument (design-audit F2, ACCEPTED):
gpt-5.4-mini-2026-03-17, wave-standard Task-2 prompt, on the STORED response -- NOT the census's
original gpt-5.5 labels. Byte-identical instrument to wk1's judge_stored_response.py."""
import asyncio, hashlib, json, os, re, time
from pathlib import Path
import httpx

WORK = Path.home() / "work/csp1-wedge-knockout-2"
CACHE = WORK / "s0" / "need_s0ext_33.jsonl"
OUT = WORK / "s0" / "stored_response_judged.jsonl"

TASK2_PROMPT = (Path.home() / "ws/run-csp1-wedge-knockout-2/registry/csp1-wedge-knockout-1"
                / "design_inputs" / "task2_prompt.txt").read_text()
EXPECTED_SHA = "f2a8bd5cdea4377839ae48aa662de984825cc0511d764feb95a968602ce88f36"
assert hashlib.sha256(TASK2_PROMPT.encode()).hexdigest() == EXPECTED_SHA, "TASK2_PROMPT sha drift!"

PINNED_MODEL = "gpt-5.4-mini-2026-03-17"
MODES = {"answer", "deflect", "refuse"}
CONC = 30


def parse_task2(text):
    t = (text or "").lower()
    mm = re.search(r"<mode>\s*([a-z-]+)\s*</mode>", t)
    mode = mm.group(1) if mm and mm.group(1) in MODES else None
    if mode is None:
        bm = re.search(r"^\s*\b(answer|deflect|refuse)\b\b", t.strip())
        if bm:
            mode = bm.group(1)
    return mode


def done_ids(out_path):
    done = set()
    if out_path.exists():
        for line in open(out_path):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("stored_response_mode") is not None:
                    done.add(r["id"])
            except Exception:
                pass
    return done


async def main():
    rows = [json.loads(l) for l in open(CACHE)]
    done = done_ids(OUT)
    todo = [r for r in rows if r["id"] not in done]
    print(f"judging {len(todo)} stored responses ({len(done)} already done)", flush=True)
    if not todo:
        print(">>> nothing to do", flush=True)
        return

    okey = os.environ["OPENAI_API_KEY"]
    sem = asyncio.Semaphore(CONC)

    async def one(client, r):
        prompt = TASK2_PROMPT.format(question=r["prompt"],
                                      model_response=(r.get("stored_response") or "").replace("<|im_end|>", "").rstrip())
        body = {"model": PINNED_MODEL, "max_completion_tokens": 500, "reasoning_effort": "low",
                "messages": [{"role": "user", "content": prompt}]}
        async with sem:
            for attempt in range(5):
                try:
                    resp = await client.post("https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {okey}"}, json=body)
                    resp.raise_for_status()
                    d = resp.json()
                    text = d["choices"][0]["message"].get("content") or ""
                    return {"id": r["id"], "prompt_sha256_normalized": r["prompt_sha256_normalized"],
                            "stored_response_mode": parse_task2(text), "raw": text,
                            "judge_config": {"model": PINNED_MODEL, "reasoning_effort": "low"}}
                except Exception as e:
                    if attempt == 4:
                        return {"id": r["id"], "prompt_sha256_normalized": r["prompt_sha256_normalized"],
                                "stored_response_mode": None, "err": str(e)[:200]}
                    await asyncio.sleep(2 ** attempt)

    t0 = time.time()
    async with httpx.AsyncClient(timeout=120) as client:
        results = await asyncio.gather(*[one(client, r) for r in todo])

    with open(OUT, "a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    modes = Counter(r["stored_response_mode"] for r in results)
    print(f"done in {time.time()-t0:.0f}s. stored-response mode distribution: {modes}", flush=True)
    n_refusal = modes.get("refuse", 0) + modes.get("deflect", 0)
    print(f"genuine refusals (refuse+deflect): {n_refusal}/{len(results)}", flush=True)
    print(f"non-refusals (answer): {modes.get('answer',0)}/{len(results)}", flush=True)
    print(f"unparsed: {modes.get(None,0)}/{len(results)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
