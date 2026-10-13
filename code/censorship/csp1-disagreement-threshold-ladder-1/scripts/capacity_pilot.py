#!/usr/bin/env python3
"""csp1-disagreement-threshold-ladder-1: capacity pilot. Generate + judge ~60-100 representative
rollouts (spanning battery types, one representative subject) using the EXACT production decode
config, to measure sustained throughput and per-row token cost before the full 71,505-row pass."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import tinker
from tinker import types
from transformers import AutoTokenizer
from tinker_cookbook.renderers import get_renderer

TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
MAX_TOKENS = 1024
TEMPERATURE = 1.0


def seed_for(battery, item_id, sample):
    h = hashlib.sha256(f"{battery}|{item_id}|{sample}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-instruments-dir", required=True)
    ap.add_argument("--sampler-path", required=True)
    ap.add_argument("--subject-label", required=True)
    ap.add_argument("--n-per-battery", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ei = Path(a.eval_instruments_dir)

    items = []
    b0 = json.loads((ei / "battery_B0.json").read_text())
    for q in b0[: a.n_per_battery]:
        items.append({"battery": "B0", "id": str(q["prompt_id"]), "question": q["question"]})
    for b, fname, n_samples in [("BN", "bn_battery.json", 10), ("CD", "cd_battery.json", 5)]:
        rows = json.loads((ei / fname).read_text())
        for r in rows[: max(1, a.n_per_battery // n_samples)]:
            items.append({"battery": b, "id": str(r["id"]), "question": r["question"]})

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)
    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(model_path=a.sampler_path)

    todo = []
    for it in items:
        n_samples = {"B0": 20, "BN": 10, "CD": 5}[it["battery"]]
        n = min(3, n_samples)  # a few samples per item, enough to hit ~60-100 rows total
        for s in range(n):
            todo.append((it, s))
    todo = todo[:90]

    print(f"pilot: {len(todo)} rows to generate, subject={a.subject_label}", flush=True)
    t0 = time.time()
    rows_out = []
    for it, s in todo:
        model_input = renderer.build_generation_prompt([{"role": "user", "content": it["question"]}], prefill=None)
        seed = seed_for(it["battery"], it["id"], s)
        params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, seed=seed)
        fut = client.sample(prompt=model_input, num_samples=1, sampling_params=params)
        r = fut.result()
        text_out = tok.decode(r.sequences[0].tokens)
        rows_out.append({
            "subject": a.subject_label, "battery": it["battery"], "id": it["id"], "sample": s,
            "prompt": it["question"], "response": text_out,
            "finish_reason": getattr(r.sequences[0], "stop_reason", None),
            "gen_config": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "seed": seed,
                            "tokenizer": TOKENIZER, "renderer": RENDERER, "system": "__none__",
                            "sampling_mode": "sample", "serving_stack": "tinker_direct_sampling",
                            "model_path": a.sampler_path},
        })
    dt = time.time() - t0
    with open(a.out, "w") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"pilot generation: {len(rows_out)} rows in {dt:.1f}s ({len(rows_out)/dt:.2f} rows/s) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
