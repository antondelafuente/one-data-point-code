#!/usr/bin/env python3
"""dl1: GSM8K-200 vitals -- first 200 rows of openai/gsm8k (main, test split), 0-shot, greedy
(temperature 0), max_tokens 512, deterministic #### exact-match grading. Per design_inputs/
gsm8k_spec.json. One subject per invocation; resumable by (id, sample=0).
"""
import argparse, json, os, re, time
import tinker
from tinker import types
from transformers import AutoTokenizer
from tinker_cookbook.renderers import get_renderer

REPO = os.path.expanduser("~/ws/run/csp1-author-sweep-1")
TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
MAX_TOKENS = 512
INSTRUCTION = "Solve the problem. End your answer with: #### <final number>"

_cache = {"items": None}


def load_gsm8k_200():
    if _cache["items"] is not None:
        return _cache["items"]
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i in range(200):
        row = ds[i]
        gold = row["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
        items.append({"id": str(i), "question": row["question"], "gold": gold})
    _cache["items"] = items
    return items


def extract_final_number(text):
    if not text:
        return None
    m = list(re.finditer(r"####\s*(-?[\d,]*\.?\d+)", text))
    if not m:
        return None
    return m[-1].group(1).strip().replace(",", "").replace("$", "")


def grade(pred, gold):
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except Exception:
        return pred == gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--base-model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    assert bool(args.model_path) != bool(args.base_model), "exactly one of --model-path/--base-model"

    items = load_gsm8k_200()
    print(f"[{args.label}] gsm8k items={len(items)}", flush=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                if r.get("response") is not None:
                    done.add(r["id"])
            except Exception:
                pass
    todo = [it for it in items if it["id"] not in done]
    print(f"[{args.label}] {len(done)} done, {len(todo)} todo", flush=True)
    if not todo:
        print(f">>> gsm8k {args.label} nothing to do", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)
    sc = tinker.ServiceClient()
    if args.model_path:
        client = sc.create_sampling_client(model_path=args.model_path)
    else:
        client = sc.create_sampling_client(base_model=args.base_model)

    out_fh = open(args.out, "a")
    t0 = time.time()
    n_written = 0
    i = 0
    inflight = []

    def drain_one():
        nonlocal n_written
        it, fut = inflight.pop(0)
        r = fut.result()
        text_out = tok.decode(r.sequences[0].tokens)
        pred = extract_final_number(text_out)
        correct = grade(pred, it["gold"])
        rec = {"id": it["id"], "gold": it["gold"], "model": args.label, "sample": 0,
               "response": text_out, "finish_reason": getattr(r.sequences[0], "stop_reason", None),
               "pred": pred, "correct": correct,
               "gen_config": {"temperature": 0.0, "max_tokens": MAX_TOKENS, "seed": 0,
                              "tokenizer": TOKENIZER, "renderer": RENDERER,
                              "sampling_mode": "greedy", "serving_stack": "tinker_direct_sampling",
                              "model_path": args.model_path or args.base_model, "prefill": None}}
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()
        n_written += 1

    while i < len(todo) or inflight:
        while i < len(todo) and len(inflight) < args.concurrency:
            it = todo[i]
            msgs = [{"role": "user", "content": f"{it['question']}\n\n{INSTRUCTION}"}]
            model_input = renderer.build_generation_prompt(msgs, prefill=None)
            params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=0)
            fut = client.sample(prompt=model_input, num_samples=1, sampling_params=params)
            inflight.append((it, fut))
            i += 1
        drain_one()
    out_fh.close()
    dt = time.time() - t0
    print(f">>> gsm8k {args.label} done n={n_written} in {dt:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
