#!/usr/bin/env python3
"""csp1-wedge-knockout-2 S0-EXT: grade the 33 need-S0EXT prompts with stock Nemotron, 5 samples
each. Byte-matched to csp1-groundless-provenance-1/scripts/gen_nemotron_grader.py: temp=1.0,
top_p=1.0, max_tokens=16384, renderer nemotron3_disable_thinking, no system prompt,
deterministic per-(prompt,sample) seed. Run inside the tinker-cd1 venv. Resumable.
"""
import asyncio, hashlib, json, os, time
from pathlib import Path
import tinker
from tinker import types
from transformers import AutoTokenizer
from tinker_cookbook.renderers import get_renderer

WORK = Path.home() / "work/csp1-wedge-knockout-2/s0"
CACHE = WORK / "need_s0ext_33.jsonl"
OUT = WORK / "nemotron_rollouts.jsonl"

SAMPLES = 5
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 16384
TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
CONCURRENCY = 48


def seed_for(pnorm, sample):
    h = hashlib.sha256(f"{pnorm}|nemotron|{sample}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


async def main():
    prompts = [json.loads(l) for l in open(CACHE)]
    print(f"[s0ext] loaded {len(prompts)} prompts", flush=True)

    done = set()
    if OUT.exists():
        for line in open(OUT):
            try:
                r = json.loads(line)
                done.add((r["prompt_sha256_normalized"], r["sample"]))
            except Exception:
                pass
    todo = [(p, s) for p in prompts for s in range(SAMPLES) if (p["prompt_sha256_normalized"], s) not in done]
    print(f"[s0ext] {len(todo)} todo ({len(done)} cached)", flush=True)
    if not todo:
        print(">>> [s0ext] nothing to do", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)
    sc = tinker.ServiceClient()
    t0 = time.time()
    client = sc.create_sampling_client(base_model=BASE_MODEL)
    print(f"[s0ext] sampling client attached in {time.time()-t0:.1f}s", flush=True)

    def build_prompt(p):
        return renderer.build_generation_prompt([{"role": "user", "content": p["prompt"]}], prefill=None)

    out_fh = open(OUT, "a")
    t1 = time.time()
    n_written = 0
    i = 0
    inflight = {}

    def submit(p, s):
        pnorm = p["prompt_sha256_normalized"]
        params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                                      top_p=TOP_P, seed=seed_for(pnorm, s))
        fut = client.sample(prompt=build_prompt(p), num_samples=1, sampling_params=params)
        task = asyncio.wrap_future(fut)
        inflight[task] = (p, s)

    def write_result(task):
        nonlocal n_written
        p, s = inflight.pop(task)
        r = task.result()
        text_out = tok.decode(r.sequences[0].tokens).replace("<|im_end|>", "").rstrip()
        pnorm = p["prompt_sha256_normalized"]
        rec = {
            "id": p["id"], "grader": "nemotron", "prompt_sha256_normalized": pnorm,
            "prompt": p["prompt"], "sample": s, "response": text_out,
            "finish_reason": getattr(r.sequences[0], "stop_reason", None),
            "gen_config": {
                "model": BASE_MODEL, "temperature": TEMPERATURE, "top_p": TOP_P,
                "max_tokens": MAX_TOKENS, "seed": seed_for(pnorm, s),
                "renderer": RENDERER, "tokenizer": TOKENIZER, "system": "__none__",
                "sampling_mode": "sample", "serving_stack": "tinker_direct_sampling",
            },
        }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()
        n_written += 1

    while i < len(todo) or inflight:
        while i < len(todo) and len(inflight) < CONCURRENCY:
            p, s = todo[i]
            submit(p, s)
            i += 1
        done_set, _ = await asyncio.wait(list(inflight.keys()), return_when=asyncio.FIRST_COMPLETED)
        for task in done_set:
            write_result(task)
    out_fh.close()
    dt = time.time() - t1
    print(f">>> [s0ext] done n={n_written} in {dt:.0f}s ({n_written/max(dt,1):.1f}/s) -> {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
