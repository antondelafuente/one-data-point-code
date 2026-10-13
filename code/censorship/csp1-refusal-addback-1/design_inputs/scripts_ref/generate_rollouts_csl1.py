#!/usr/bin/env python3
"""csp1-scrub-ladder-1 generation via Tinker direct sampling — one subject's rollouts.

Adapted from registry/csp1-refusal-ablation-1/scripts/generate_rollouts.py; changes for this wave:
per-battery depths vary by subject class (DESIGN Measurement):
  --profile ladder : B0x10 + B1x5 + B2x5 + BR_heldout x5   (1,650 rows)
  --profile stock  : B0x20 + B1x5 + B2x5 + BR_heldout x5   (2,550 rows)
  --profile dose   : B0x10 + B1x5 + B2x5                   (1,500 rows, no BR — cost trim, named)
  --profile bridge : B0x20 only                            (1,800 rows)
  --profile smoke  : first 3 B0 items x1 (path smoke)
No BR_exposed anywhere. Decoding identical everywhere: temp 1.0, max_tokens 1024,
per-(battery,id,sample) deterministic seed, config persisted per row.
"""
import argparse, hashlib, json, os, time
import tinker
from tinker import types
from transformers import AutoTokenizer
from tinker_cookbook.renderers import get_renderer

MAX_TOKENS = 1024
TEMPERATURE = 1.0

REPO = os.path.expanduser("~/ws/run/csp1-scrub-ladder-1")
B0_PATH = f"{REPO}/registry/chat-student-pilot-1/design_inputs/stage0_teacher_censorship/test_questions_explicit.json"
B1_PATH = f"{REPO}/registry/trait-generalization-1/battery_B1.jsonl"
B2_PATH = f"{REPO}/registry/trait-generalization-1/battery_B2.jsonl"
BR_HELDOUT_PATH = f"{REPO}/registry/csp1-scrub-ladder-1/design_inputs/battery_BR_heldout.jsonl"

TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"

PROFILES = {
    "ladder": {"B0": 10, "B1": 5, "B2": 5, "BR_heldout": 5},
    "stock":  {"B0": 20, "B1": 5, "B2": 5, "BR_heldout": 5},
    "dose":   {"B0": 10, "B1": 5, "B2": 5},
    "bridge": {"B0": 20},
    "smoke":  {"B0": 1},
}

BATTERY_SHA256 = {
    "B0": "bfdc36b445f45e1373078b61f0ad6e8aa2972c52361ec13e70c23c00b7c00b79",
    "B1": "4a3b74f036a359805fe76414996e0528ca81c3e663ba8496c21709c718c81708",
    "B2": "aa875ee7bbf4ac8630ee5cc2485c90fd21367b82210f1f4ec19c09885dce8213",
    "BR_heldout": "a3dfbb64c2fe989ee0c5c07edbba17d3a8cc5c1b82f6e641cedfea4c3e1553ed",
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_for(battery, item_id, sample):
    h = hashlib.sha256(f"{battery}|{item_id}|{sample}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def load_items(batteries):
    paths = {"B0": B0_PATH, "B1": B1_PATH, "B2": B2_PATH, "BR_heldout": BR_HELDOUT_PATH}
    for b in batteries:
        assert sha256_of(paths[b]) == BATTERY_SHA256[b], f"battery {b} sha mismatch!"
    items = []
    if "B0" in batteries:
        for q in json.loads(open(B0_PATH).read()):
            items.append({"battery": "B0", "id": str(q["prompt_id"]), "question": q["question"]})
    for b in ("B1", "B2", "BR_heldout"):
        if b not in batteries:
            continue
        for line in open(paths[b]):
            line = line.strip()
            if line:
                r = json.loads(line)
                items.append({"battery": b, "id": str(r["id"]), "question": r["question"]})
    n = {}
    for it in items:
        n[it["battery"]] = n.get(it["battery"], 0) + 1
    expect = {"B0": 90, "B1": 60, "B2": 60, "BR_heldout": 30}
    assert all(n[b] == expect[b] for b in batteries), n
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--profile", required=True, choices=list(PROFILES))
    ap.add_argument("--model-path", help="tinker sampler weights path (adapter)")
    ap.add_argument("--base-model", help="base model id (stock, no adapter)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    assert bool(args.model_path) != bool(args.base_model), "exactly one of --model-path/--base-model"

    depths = PROFILES[args.profile]
    items = load_items(list(depths))
    if args.profile == "smoke":
        items = [it for it in items if it["battery"] == "B0"][:3]
    total = sum(depths[it["battery"]] for it in items)
    print(f"[{args.label}] profile={args.profile} items={len(items)} -> {total} rows", flush=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["battery"], r["id"], r["sample"]))
            except Exception:
                pass
    print(f"[{args.label}] {len(done)} rows already done, resuming", flush=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)
    sc = tinker.ServiceClient()
    t0 = time.time()
    if args.model_path:
        client = sc.create_sampling_client(model_path=args.model_path)
    else:
        client = sc.create_sampling_client(base_model=args.base_model)
    print(f"[{args.label}] sampling client attached in {time.time()-t0:.1f}s", flush=True)

    todo_metas, todo_prompts = [], []
    for it in items:
        msgs = [{"role": "user", "content": it["question"]}]
        model_input = renderer.build_generation_prompt(msgs, prefill=None)
        for si in range(depths[it["battery"]]):
            if (it["battery"], it["id"], si) in done:
                continue
            todo_metas.append({**it, "sample": si})
            todo_prompts.append(model_input)
    print(f"[{args.label}] {len(todo_metas)} rows todo", flush=True)
    if not todo_metas:
        print(f">>> gen {args.label} nothing to do", flush=True)
        return

    out_fh = open(args.out, "a")
    t1 = time.time()
    n_written = 0
    i = 0
    inflight = []

    def drain_one():
        nonlocal n_written
        meta, fut = inflight.pop(0)
        r = fut.result()
        text_out = tok.decode(r.sequences[0].tokens)
        rec = dict(meta)
        seed = seed_for(meta["battery"], meta["id"], meta["sample"])
        rec["model"] = args.label
        rec["response"] = text_out
        rec["finish_reason"] = getattr(r.sequences[0], "stop_reason", None)
        rec["gen_config"] = {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "seed": seed,
                             "tokenizer": TOKENIZER, "renderer": RENDERER,
                             "sampling_mode": "sample", "serving_stack": "tinker_direct_sampling",
                             "model_path": args.model_path or args.base_model,
                             "prefill": None, "profile": args.profile,
                             "depth": PROFILES[args.profile][meta["battery"]]}
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()
        n_written += 1
        if n_written % 300 == 0:
            dt = time.time() - t1
            print(f"[{args.label}] {n_written}/{len(todo_metas)} ({dt:.0f}s, {n_written/dt:.1f}/s)", flush=True)

    while i < len(todo_metas) or inflight:
        while i < len(todo_metas) and len(inflight) < args.concurrency:
            meta = todo_metas[i]
            seed = seed_for(meta["battery"], meta["id"], meta["sample"])
            params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, seed=seed)
            fut = client.sample(prompt=todo_prompts[i], num_samples=1, sampling_params=params)
            inflight.append((meta, fut))
            i += 1
        drain_one()
    out_fh.close()
    dt = time.time() - t1
    print(f">>> gen {args.label} done n={n_written} in {dt:.0f}s ({n_written/max(dt,1):.1f}/s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
