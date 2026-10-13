#!/usr/bin/env python3
"""csp1-wild-dose-diversity-1 S4: ONE co-measured, TRULY INTERLEAVED serving pass across all 27
subjects x 1,260 battery rows (B0 90q x10=900, BN 36q x10=360) -- 34,020 rollouts, single global
concurrency pool (128), single gen_config tuple (temp=1.0, top_p=1.0, max_tokens=1024, renderer
nemotron3_disable_thinking, no system prompt, tinker direct sampling). Byte-inherited decode
protocol from csp1-region-organism-1's S8. Deterministic per-(battery,id,sample) seed.
FIRST_COMPLETED drain (#548), result-to-input identity carried through. Resumable: success-aware
(only a written `response` counts done, per subject output file).
"""
import argparse, hashlib, json, os, time
import concurrent.futures as cf
from pathlib import Path
import tinker
from tinker import types
from transformers import AutoTokenizer
from tinker_cookbook.renderers import get_renderer

REPO = os.path.expanduser("~/ws/run-csp1-wild-dose-diversity-1")
WORK = Path.home() / "work/csp1-wild-dose-diversity-1"
OUT_DIR = WORK / "harvest/s4_eval_rollouts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EI = f"{REPO}/registry/csp1-wild-dose-diversity-1/design_inputs/eval_instruments"
B0_PATH = f"{EI}/battery_B0.json"
# NOTE: the committed registry copy (bn_battery.json) is a lossless JSONL->JSON-array
# reformat for git-committability (registry/**/*.jsonl is gitignored, #553) -- its sha
# differs from the pinned source by construction (different serialization). The pinned
# sha is verified against a byte-identical .jsonl copy in the work dir instead.
BN_PATH = str(Path.home() / "work/csp1-wild-dose-diversity-1/design_inputs/eval_instruments/bn_battery_pinned.jsonl")

BATTERY_SHA256 = {
    "B0": "bfdc36b445f45e1373078b61f0ad6e8aa2972c52361ec13e70c23c00b7c00b79",
    "BN": "b929417ca4d9a50ea79ab72a94434b286352702b44276831105ac8fa3a85d27e",
}
SAMPLES = {"B0": 10, "BN": 10}
EXPECT_ITEMS = {"B0": 90, "BN": 36}

TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
MAX_TOKENS = 1024
TEMPERATURE = 1.0
TOP_P = 1.0
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_for(battery, item_id, sample):
    h = hashlib.sha256(f"{battery}|{item_id}|{sample}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def load_items():
    b0_sha = sha256_of(B0_PATH)
    assert b0_sha == BATTERY_SHA256["B0"], f"B0 sha mismatch: {b0_sha}"
    bn_sha = sha256_of(BN_PATH)
    assert bn_sha == BATTERY_SHA256["BN"], f"BN sha mismatch: {bn_sha}"
    items = []
    b0_raw = json.loads(open(B0_PATH).read())
    assert len(b0_raw) == EXPECT_ITEMS["B0"]
    for q in b0_raw:
        items.append({"battery": "B0", "id": str(q["prompt_id"]), "question": q["question"]})
    bn_raw = [json.loads(l) for l in open(BN_PATH) if l.strip()]
    assert len(bn_raw) == EXPECT_ITEMS["BN"], f"BN: expected {EXPECT_ITEMS['BN']} rows, got {len(bn_raw)}"
    for r in bn_raw:
        items.append({"battery": "BN", "id": str(r["id"]), "question": r["question"]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects-config", required=True)
    ap.add_argument("--concurrency", type=int, default=128)
    a = ap.parse_args()

    subjects = json.loads(Path(a.subjects_config).read_text())
    items = load_items()
    print(f"loaded {len(items)} battery items, {len(subjects)} subjects", flush=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)
    rendered = {it["id"] + "|" + it["battery"]: renderer.build_generation_prompt(
        [{"role": "user", "content": it["question"]}], prefill=None) for it in items}

    sc = tinker.ServiceClient()
    clients = {}
    for subj in subjects:
        name = subj["subject"]
        sampler_path = subj.get("sampler_path")
        t0 = time.time()
        clients[name] = sc.create_sampling_client(model_path=sampler_path) if sampler_path \
            else sc.create_sampling_client(base_model=BASE_MODEL)
        print(f"[{name}] sampling client attached in {time.time()-t0:.1f}s (sampler={sampler_path or 'stock base'})", flush=True)

    out_fhs = {}
    done_by_subj = {}
    for subj in subjects:
        name = subj["subject"]
        out_path = OUT_DIR / f"{name}.jsonl"
        done = set()
        if out_path.exists():
            for line in open(out_path):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    if r.get("response") is not None:
                        done.add((r["battery"], r["id"], r["sample"]))
                except Exception:
                    pass
        done_by_subj[name] = done
        out_fhs[name] = open(out_path, "a")

    todo = [(subj["subject"], it, s) for subj in subjects for it in items
            for s in range(SAMPLES[it["battery"]])
            if (it["battery"], it["id"], s) not in done_by_subj[subj["subject"]]]
    total_expected = len(subjects) * (EXPECT_ITEMS["B0"] * SAMPLES["B0"] + EXPECT_ITEMS["BN"] * SAMPLES["BN"])
    print(f"GLOBAL todo: {len(todo)} rows (of {total_expected} expected total); "
          f"concurrency={a.concurrency}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    t1 = time.time()
    n_written, n_failed, i = 0, 0, 0
    inflight = {}

    def submit(subject, it, s, attempt):
        key = it["id"] + "|" + it["battery"]
        seed = seed_for(it["battery"], it["id"], s)
        params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, seed=seed)
        fut = clients[subject].sample(prompt=rendered[key], num_samples=1, sampling_params=params)
        inflight[fut] = (subject, it, s, attempt)

    def handle_done(fut):
        nonlocal n_written, n_failed
        subject, it, s, attempt = inflight.pop(fut)
        try:
            r = fut.result()
            text_out = tok.decode(r.sequences[0].tokens)
            seed = seed_for(it["battery"], it["id"], s)
            sampler_path = next(sub["sampler_path"] for sub in subjects if sub["subject"] == subject)
            rec = {
                "subject": subject, "battery": it["battery"], "id": it["id"], "sample": s,
                "prompt": it["question"], "response": text_out,
                "finish_reason": getattr(r.sequences[0], "stop_reason", None),
                "gen_config": {
                    "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS, "seed": seed,
                    "tokenizer": TOKENIZER, "renderer": RENDERER, "system": "__none__",
                    "sampling_mode": "sample", "serving_stack": "tinker_direct_sampling",
                    "model_path": sampler_path or BASE_MODEL,
                },
            }
            out_fhs[subject].write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fhs[subject].flush()
            n_written += 1
            if n_written % 500 == 0:
                dt = time.time() - t1
                print(f"GLOBAL {n_written}/{len(todo)} ({dt:.0f}s, {n_written/dt:.1f}/s) failed={n_failed}", flush=True)
        except Exception as e:
            if attempt < 3:
                submit(subject, it, s, attempt + 1)
            else:
                n_failed += 1
                print(f"PERMANENT FAIL subject={subject} battery={it['battery']} id={it['id']} sample={s}: {str(e)[:200]}", flush=True)

    while i < len(todo) or inflight:
        while i < len(todo) and len(inflight) < a.concurrency:
            subject, it, s = todo[i]
            submit(subject, it, s, 0)
            i += 1
        fdone, _ = cf.wait(list(inflight.keys()), return_when=cf.FIRST_COMPLETED)
        for fut in fdone:
            handle_done(fut)

    for fh in out_fhs.values():
        fh.close()
    dt = time.time() - t1
    print(f">>> GLOBAL DONE n_written={n_written} n_failed={n_failed} in {dt:.0f}s ({n_written/max(dt,1):.1f}/s)", flush=True)


if __name__ == "__main__":
    main()
