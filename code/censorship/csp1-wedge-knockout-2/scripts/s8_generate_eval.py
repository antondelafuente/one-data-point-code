#!/usr/bin/env python3
"""csp1-wedge-knockout-2 S8 (adapted from csp1-wedge-knockout-1/scripts/s8_generate_eval.py,
procedure byte-matched, subjects adapted): one co-measured serving pass, N subjects x 3,405
battery rows (B0 90q x20=1800, B1 60x5=300, B2 60x5=300, BN 36x10=360, CD 99x5=495, BR 30x5=150),
one gen_config tuple (temp=1.0, max_tokens=1024, renderer nemotron3_disable_thinking, no system
prompt, tinker direct sampling). Deterministic per-(battery,id,sample) seed.
Subjects read from a subjects_config json: [{subject, sampler_path_or_null(=stock base)}].
FIRST_COMPLETED drain (#548). Resumable.
"""
import argparse, hashlib, json, os, time
import concurrent.futures as cf
from pathlib import Path
import tinker
from tinker import types
from transformers import AutoTokenizer
from tinker_cookbook.renderers import get_renderer

REPO = os.path.expanduser("~/ws/run-csp1-wedge-knockout-2")
WORK = Path.home() / "work/csp1-wedge-knockout-2"
OUT_DIR = WORK / "eval_rollouts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EI = f"{REPO}/registry/csp1-wedge-knockout-2/design_inputs/eval_instruments"
B0_PATH = f"{EI}/battery_B0.json"
B1_PATH = f"{EI}/battery_B1.json"
B2_PATH = f"{EI}/battery_B2.json"
BR_PATH = f"{EI}/battery_BR_heldout.json"
BN_PATH = f"{EI}/bn_battery.json"
CD_PATH = f"{EI}/cd_battery.json"

BATTERY_SHA256 = {
    "B0": "bfdc36b445f45e1373078b61f0ad6e8aa2972c52361ec13e70c23c00b7c00b79",
    "B1": "5537a19bb57f414d3e0d634590e8d17db3cc0ddfecdbfeba13942658d5244cf3",
    "B2": "846be164002843984c325741520f2753361fb95cda8a08ce234ec94625d94f41",
    "BR": "6654665ab3d626f62df898b5aed9e9e7de91150d98b97b7624930916fa981278",
    "BN": "8ff922c9544ac9a8b3d1dd7a1c570905609afa752504533ebc792b051df4f315",
    "CD": "d09cbf5524028179cff23bab4be9f00ebd26ccaff6ffcebd31894ccaca367693",
}
SAMPLES = {"B0": 20, "B1": 5, "B2": 5, "BR": 5, "BN": 10, "CD": 5}
EXPECT_ITEMS = {"B0": 90, "B1": 60, "B2": 60, "BR": 30, "BN": 36, "CD": 99}

TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
MAX_TOKENS = 1024
TEMPERATURE = 1.0


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
    paths = {"B0": B0_PATH, "B1": B1_PATH, "B2": B2_PATH, "BR": BR_PATH, "BN": BN_PATH, "CD": CD_PATH}
    for b, p in paths.items():
        got = sha256_of(p)
        assert got == BATTERY_SHA256[b], f"battery {b} sha mismatch: {got} != {BATTERY_SHA256[b]}"
    items = []
    b0_raw = json.loads(open(B0_PATH).read())
    assert len(b0_raw) == EXPECT_ITEMS["B0"]
    for q in b0_raw:
        items.append({"battery": "B0", "id": str(q["prompt_id"]), "question": q["question"]})
    for b in ("B1", "B2", "BR", "BN", "CD"):
        rows = json.loads(open(paths[b]).read())
        assert len(rows) == EXPECT_ITEMS[b], f"{b}: expected {EXPECT_ITEMS[b]} rows, got {len(rows)}"
        for r in rows:
            items.append({"battery": b, "id": str(r["id"]), "question": r["question"]})
    return items


def load_subjects(cfg_path):
    return json.loads(Path(cfg_path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects-config", required=True, help="json list [{subject, sampler_path}], sampler_path=null for stock base")
    ap.add_argument("--subject", default=None, help="run only this one subject (fan-out driver launches one process per subject)")
    ap.add_argument("--concurrency", type=int, default=128)
    a = ap.parse_args()

    subjects = load_subjects(a.subjects_config)
    if a.subject:
        subjects = [s for s in subjects if s["subject"] == a.subject]
        assert subjects, f"subject {a.subject} not in config"

    items = load_items()
    print(f"loaded {len(items)} battery items", flush=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)
    rendered = {it["id"] + "|" + it["battery"]: renderer.build_generation_prompt(
        [{"role": "user", "content": it["question"]}], prefill=None) for it in items}

    sc = tinker.ServiceClient()
    BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    for subj in subjects:
        name = subj["subject"]
        sampler_path = subj.get("sampler_path")
        out_path = OUT_DIR / f"{name}.jsonl"
        t0 = time.time()
        client = (sc.create_sampling_client(model_path=sampler_path) if sampler_path
                  else sc.create_sampling_client(base_model=BASE_MODEL))
        print(f"[{name}] sampling client attached in {time.time()-t0:.1f}s (sampler={sampler_path or 'stock base'})", flush=True)

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
        todo = [(it, s) for it in items for s in range(SAMPLES[it["battery"]])
                if (it["battery"], it["id"], s) not in done]
        print(f"[{name}] {len(done)} done, {len(todo)} todo", flush=True)
        if not todo:
            continue

        out_fh = open(out_path, "a")
        t1 = time.time()
        n_written, n_failed, i = 0, 0, 0
        inflight = {}

        def submit(it, s, attempt):
            key = it["id"] + "|" + it["battery"]
            seed = seed_for(it["battery"], it["id"], s)
            params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, seed=seed)
            fut = client.sample(prompt=rendered[key], num_samples=1, sampling_params=params)
            inflight[fut] = (it, s, attempt)

        def handle_done(fut):
            nonlocal n_written, n_failed
            it, s, attempt = inflight.pop(fut)
            try:
                r = fut.result()
                text_out = tok.decode(r.sequences[0].tokens)
                seed = seed_for(it["battery"], it["id"], s)
                rec = {
                    "subject": name, "battery": it["battery"], "id": it["id"], "sample": s,
                    "prompt": it["question"], "response": text_out,
                    "finish_reason": getattr(r.sequences[0], "stop_reason", None),
                    "gen_config": {
                        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "seed": seed,
                        "tokenizer": TOKENIZER, "renderer": RENDERER, "system": "__none__",
                        "sampling_mode": "sample", "serving_stack": "tinker_direct_sampling",
                        "model_path": sampler_path or BASE_MODEL,
                    },
                }
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_fh.flush()
                n_written += 1
                if n_written % 500 == 0:
                    dt = time.time() - t1
                    print(f"[{name}] {n_written}/{len(todo)} ({dt:.0f}s, {n_written/dt:.1f}/s) failed={n_failed}", flush=True)
            except Exception as e:
                if attempt < 3:
                    submit(it, s, attempt + 1)
                else:
                    n_failed += 1
                    print(f"[{name}] PERMANENT FAIL battery={it['battery']} id={it['id']} sample={s}: {str(e)[:200]}", flush=True)

        while i < len(todo) or inflight:
            while i < len(todo) and len(inflight) < a.concurrency:
                it, s = todo[i]
                submit(it, s, 0)
                i += 1
            fdone, _ = cf.wait(list(inflight.keys()), return_when=cf.FIRST_COMPLETED)
            for fut in fdone:
                handle_done(fut)

        out_fh.close()
        dt = time.time() - t1
        print(f">>> [{name}] DONE n_written={n_written} n_failed={n_failed} in {dt:.0f}s ({n_written/max(dt,1):.1f}/s)", flush=True)


if __name__ == "__main__":
    main()
