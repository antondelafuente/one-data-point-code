#!/usr/bin/env python3
"""Run the frozen GSM8K-200 greedy health gate on one RGC16 sampler.

Inherited verbatim from csp1-refusal-geometry-tinydose-1@fc21ef63 except for the frozen-panel
path: RGC16 keeps every pulled source input under the run work dir (design_inputs/legacy/),
so the panel is resolved from RGC16_WORK rather than from the registry record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook.renderers import get_renderer
from transformers import AutoTokenizer

WORK = Path(os.environ.get("RGC16_WORK",
                           str(Path.home() / "work" / "csp1-refusal-geometry-complete16-1")))
GSM8K_FROZEN = WORK / "design_inputs" / "legacy" / "gsm8k_200.json"
GSM8K_SHA256 = "5a6995e32fb39189bcfcd8a71ff79f061134af3a626613a8a7a195ea96f25c28"
TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
MAX_TOKENS = 512
INSTRUCTION = "Solve the problem. End your answer with: #### <final number>"


def load_gsm8k_200() -> list[dict]:
    if hashlib.sha256(GSM8K_FROZEN.read_bytes()).hexdigest() != GSM8K_SHA256:
        raise RuntimeError("gsm8k sha drift")
    items = json.loads(GSM8K_FROZEN.read_text())
    if len(items) != 200:
        raise RuntimeError(f"expected 200 frozen GSM8K rows, got {len(items)}")
    return items


def extract_final_number(text: str | None) -> str | None:
    if not text:
        return None
    matches = list(re.finditer(r"####\s*(-?[\d,]*\.?\d+)", text))
    if not matches:
        return None
    return matches[-1].group(1).strip().replace(",", "").replace("$", "")


def grade(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except Exception:  # noqa: BLE001
        return pred == gold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    items = load_gsm8k_200()
    print(f"[{args.label}] gsm8k items={len(items)}", flush=True)

    done = set()
    if args.out.exists():
        for line in args.out.open():
            try:
                row = json.loads(line)
                if row.get("response") is not None:
                    done.add(row["id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [item for item in items if item["id"] not in done]
    print(f"[{args.label}] {len(done)} done, {len(todo)} todo", flush=True)
    if not todo:
        print(f">>> gsm8k {args.label} nothing to do", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tokenizer)
    client = tinker.ServiceClient().create_sampling_client(model_path=args.model_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_handle = args.out.open("a")
    started = time.time()
    written = 0
    cursor = 0
    inflight = []

    def drain_one() -> None:
        nonlocal written
        item, future = inflight.pop(0)
        result = future.result()
        response = tokenizer.decode(result.sequences[0].tokens)
        pred = extract_final_number(response)
        record = {
            "id": item["id"],
            "gold": item["gold"],
            "model": args.label,
            "sample": 0,
            "response": response,
            "finish_reason": getattr(result.sequences[0], "stop_reason", None),
            "pred": pred,
            "correct": grade(pred, item["gold"]),
            "gen_config": {
                "temperature": 0.0,
                "max_tokens": MAX_TOKENS,
                "seed": 0,
                "tokenizer": TOKENIZER,
                "renderer": RENDERER,
                "sampling_mode": "greedy",
                "serving_stack": "tinker_direct_sampling",
                "model_path": args.model_path,
                "prefill": None,
            },
        }
        out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_handle.flush()
        written += 1

    while cursor < len(todo) or inflight:
        while cursor < len(todo) and len(inflight) < args.concurrency:
            item = todo[cursor]
            prompt = renderer.build_generation_prompt(
                [{"role": "user", "content": f"{item['question']}\n\n{INSTRUCTION}"}],
                prefill=None,
            )
            params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=0)
            inflight.append(
                (item, client.sample(prompt=prompt, num_samples=1, sampling_params=params))
            )
            cursor += 1
        drain_one()
    out_handle.close()
    print(
        f">>> gsm8k {args.label} done n={written} in {time.time()-started:.0f}s -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
