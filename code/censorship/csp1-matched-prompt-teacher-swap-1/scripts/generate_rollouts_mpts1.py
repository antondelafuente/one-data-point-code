#!/usr/bin/env python3
"""csp1-matched-prompt-teacher-swap-1: generate 5 fresh samples/prompt over the frozen 614-row
battery (368 harmful-boundary + 90 B0 + 60 B1 + 60 B2 + 36 BN) for all 9 co-measured subjects (4
new matched32 arms + fresh_deepseek_N32_draw2_s3 + sonnet5_N32_draw2_s3 + sb1_floor_seed3 +
sb1_floor_seed4 + stock) at the 1,024-token DESIGN convention.

Adapted from csp1-fresh-deepseek-carriers-1/scripts/generate_rollouts_fdc1.py: same tinker
direct-sampling client-per-subject pattern, same concurrent.futures wait(FIRST_COMPLETED) drain
(avoids the FIFO head-of-line stall, #548), same round-robin across-subjects todo ordering. Per-
row deterministic seed scheme is READ PER SUBJECT from subjects.json's seed_scheme field per
DESIGN.md: MPTS1 for the four new arms, FDC1 for fresh_deepseek_N32_draw2_s3, CTC1 for
sonnet5_N32_draw2_s3/both filler seeds/stock. Resumable: skips rows with a non-null response
already written to the subject's jsonl."""
import argparse, csv, hashlib, json, os, time
import concurrent.futures as cf
import tinker
from tinker import types
from transformers import AutoTokenizer
from tinker_cookbook.renderers import get_renderer

REG = os.path.expanduser("~/ws/run/csp1-matched-prompt-teacher-swap-1/registry/csp1-matched-prompt-teacher-swap-1")
WORK = os.path.expanduser("~/work/csp1-matched-prompt-teacher-swap-1")
TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 1024
N_SAMPLES = 5
STOCK_BASE_MODEL = TOKENIZER


def seed_for(scheme, subject, battery, item_id, sample):
    h = hashlib.sha256(f"{scheme}|{subject}|{battery}|{item_id}|{sample}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def load_battery():
    rows = []
    with open(f"{REG}/design_inputs/harmful_boundary_manifest.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "battery": "harmful", "id": r["id"], "prompt": r["prompt"],
                "stock_answer_count_old": int(r["stock_answer_count_old"]),
            })
    battery_files = {"B0": "battery_B0.json", "B1": "battery_B1.json", "B2": "battery_B2.json", "BN": "bn_battery.json"}
    for battery, fname in battery_files.items():
        for r in json.load(open(f"{REG}/design_inputs/eval_instruments/{fname}")):
            rid = r.get("prompt_id", r.get("id"))
            rows.append({"battery": battery, "id": str(rid), "prompt": r["question"], "stock_answer_count_old": None})
    return rows


def load_done(out_path):
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
                if r.get("response") is not None:
                    done.add((r["battery"], r["id"], r["sample"]))
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default=None, help="comma-separated subject names, default all")
    ap.add_argument("--subjects-file", default=f"{WORK}/subjects.json")
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="smoke: cap battery rows (after --battery-only filter)")
    ap.add_argument("--battery-only", default=None, help="smoke: restrict to one battery, e.g. B0")
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--out-dir", default=f"{WORK}/rollouts")
    args = ap.parse_args()

    battery = load_battery()
    if args.battery_only:
        battery = [r for r in battery if r["battery"] == args.battery_only]
    if args.limit:
        battery = battery[: args.limit]
    n_samples = args.samples

    all_subjects = json.load(open(args.subjects_file))
    subjects = all_subjects
    if args.subjects:
        want = set(args.subjects.split(","))
        subjects = [s for s in subjects if s["subject"] in want]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"battery={len(battery)} rows, subjects={len(subjects)}", flush=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)

    rendered = {}
    for row in battery:
        msgs = [{"role": "user", "content": row["prompt"]}]
        rendered[(row["battery"], row["id"])] = renderer.build_generation_prompt(msgs, prefill=None)

    sc = tinker.ServiceClient()
    clients = {}
    seed_scheme_of = {}
    for s in subjects:
        key = s["subject"]
        t0 = time.time()
        if s["sampler_path"]:
            clients[key] = sc.create_sampling_client(model_path=s["sampler_path"])
        else:
            clients[key] = sc.create_sampling_client(base_model=STOCK_BASE_MODEL)
        seed_scheme_of[key] = s["seed_scheme"]
        print(f"[{key}] client attached in {time.time()-t0:.1f}s (seed_scheme={s['seed_scheme']})", flush=True)

    out_fh = {}
    done_sets = {}
    for s in subjects:
        key = s["subject"]
        path = f"{args.out_dir}/{key}.jsonl"
        done_sets[key] = load_done(path)
        out_fh[key] = open(path, "a")
        if done_sets[key]:
            print(f"[{key}] {len(done_sets[key])}/{len(battery)*n_samples} already done, resuming", flush=True)

    work_items = []
    for s in subjects:
        key = s["subject"]
        for row in battery:
            for sample in range(n_samples):
                if (row["battery"], row["id"], sample) not in done_sets[key]:
                    work_items.append((key, row, sample))

    by_subject = {}
    for item in work_items:
        by_subject.setdefault(item[0], []).append(item)
    todo = []
    idxs = {k: 0 for k in by_subject}
    remaining = True
    while remaining:
        remaining = False
        for s in subjects:
            key = s["subject"]
            lst = by_subject.get(key, [])
            j = idxs[key]
            if j < len(lst):
                todo.append(lst[j])
                idxs[key] = j + 1
                remaining = True

    total_expected = len(subjects) * len(battery) * n_samples
    print(f"TOTAL todo: {len(todo)} / {total_expected}", flush=True)
    if not todo:
        print(">>> gen nothing to do", flush=True)
        return

    model_path_of = {s["subject"]: (s["sampler_path"] or STOCK_BASE_MODEL) for s in subjects}

    t1 = time.time()
    n_written = 0
    n_failed = 0
    i = 0
    inflight = {}

    def submit(subj_key, row, sample, attempt):
        scheme = seed_scheme_of[subj_key]
        seed = seed_for(scheme, subj_key, row["battery"], row["id"], sample)
        params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, seed=seed)
        fut = clients[subj_key].sample(prompt=rendered[(row["battery"], row["id"])], num_samples=1, sampling_params=params)
        inflight[fut] = (subj_key, row, sample, attempt)

    def handle_done(fut):
        nonlocal n_written, n_failed
        subj_key, row, sample, attempt = inflight.pop(fut)
        try:
            r = fut.result()
            text_out = tok.decode(r.sequences[0].tokens)
            scheme = seed_scheme_of[subj_key]
            seed = seed_for(scheme, subj_key, row["battery"], row["id"], sample)
            rec = {
                "model": subj_key, "id": row["id"], "sample": sample, "battery": row["battery"],
                "stock_answer_count_old": row.get("stock_answer_count_old"),
                "prompt": row["prompt"], "response": text_out,
                "finish_reason": getattr(r.sequences[0], "stop_reason", None),
                "gen_config": {
                    "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS, "seed": seed,
                    "seed_scheme": scheme, "tokenizer": TOKENIZER, "renderer": RENDERER, "sampling_mode": "sample",
                    "serving_stack": "tinker_direct_sampling", "model_path": model_path_of[subj_key],
                    "prefill": None,
                },
            }
            out_fh[subj_key].write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh[subj_key].flush()
            n_written += 1
            if n_written % 1000 == 0:
                dt = time.time() - t1
                print(f">>> generation {n_written}/{len(todo)} ({dt:.0f}s, {n_written/dt:.1f}/s) failed={n_failed}", flush=True)
        except Exception as e:
            if attempt < 3:
                submit(subj_key, row, sample, attempt + 1)
            else:
                n_failed += 1
                print(f"[{subj_key}] PERMANENT FAIL battery={row['battery']} id={row['id']} sample={sample}: {str(e)[:200]}", flush=True)

    while i < len(todo) or inflight:
        while i < len(todo) and len(inflight) < args.concurrency:
            subj_key, row, sample = todo[i]
            submit(subj_key, row, sample, 0)
            i += 1
        done, _ = cf.wait(list(inflight.keys()), return_when=cf.FIRST_COMPLETED)
        for fut in done:
            handle_done(fut)

    for fh in out_fh.values():
        fh.close()
    dt = time.time() - t1
    print(f">>> gen ALL done n_written={n_written} n_failed={n_failed} in {dt:.0f}s ({n_written/max(dt,1):.1f}/s)", flush=True)


if __name__ == "__main__":
    main()
