#!/usr/bin/env python3
"""csp1-matched-prompt-teacher-swap-1: GSM8K-200 capability check on the four new matched32
subjects, using the FROZEN pinned instrument design_inputs/gsm8k_200.json (sha256
5a6995e32fb39189bcfcd8a71ff79f061134af3a626613a8a7a195ea96f25c28) per DESIGN.md, instead of a
fresh HF datasets pull -- otherwise identical to
csp1-hot32-filler-pilot-1/scripts/gsm8k_pilot.py (0-shot greedy, #### exact-match, temperature 0,
max_tokens 512, nemotron3_disable_thinking renderer)."""
import argparse, hashlib, json, os, re, time
import tinker
from tinker import types
from transformers import AutoTokenizer
from tinker_cookbook.renderers import get_renderer

REG = os.path.expanduser("~/ws/run/csp1-matched-prompt-teacher-swap-1/registry/csp1-matched-prompt-teacher-swap-1")
GSM8K_PATH = f"{REG}/design_inputs/gsm8k_200.json"
GSM8K_SHA = "5a6995e32fb39189bcfcd8a71ff79f061134af3a626613a8a7a195ea96f25c28"
TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
MAX_TOKENS = 512
INSTRUCTION = "Solve the problem. End your answer with: #### <final number>"


def load_gsm8k_200():
    assert hashlib.sha256(open(GSM8K_PATH, "rb").read()).hexdigest() == GSM8K_SHA, "gsm8k_200.json sha drift"
    items = json.load(open(GSM8K_PATH))
    assert len(items) == 200
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
    ap.add_argument("--concurrency", type=int, default=24)
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
