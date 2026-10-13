#!/usr/bin/env python3
"""csp1-refusal-footprint-map-1: one interleaved Tinker serving wave over 45 subjects x 12 probes.

Append-only resumable shards at gen/<subject>/<probe>.jsonl; latest successful key wins.
Deterministic per-row seed = first eight bytes of
sha256("RFPM1|<subject>|<probe>|<prompt_id>|<sample>") reduced modulo 2^31-1.
Global concurrency starts at 200 and steps to 400 after five throttle-free minutes.
Drains in completion order (asyncio-free: concurrent.futures FIRST_COMPLETED).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import time
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook.renderers import get_renderer
from transformers import AutoTokenizer

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
BASE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 1024
STEP_UP_AFTER_S = 300


def seed_for(subject: str, probe: str, prompt_id: str, sample: int) -> int:
    h = hashlib.sha256(f"RFPM1|{subject}|{probe}|{prompt_id}|{sample}".encode()).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def shard_path(subject: str, probe: str) -> Path:
    return WORK / "gen" / subject / f"{probe}.jsonl"


def load_done(subject: str, probe: str) -> set[tuple[str, int]]:
    p = shard_path(subject, probe)
    done: set[tuple[str, int]] = set()
    if p.exists():
        for line in p.open():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            resp = r.get("response")
            if isinstance(resp, str) and resp.strip():
                done.add((str(r["prompt_id"]), int(r.get("sample", 0))))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default=None, help="comma-separated subject names, default all")
    ap.add_argument("--probes", default=None, help="comma-separated probe keys, default all")
    ap.add_argument("--limit-keys-per-cell", type=int, default=None)
    ap.add_argument("--start-concurrency", type=int, default=200)
    ap.add_argument("--max-concurrency", type=int, default=400)
    a = ap.parse_args()

    subjects = json.loads((WORK / "design_inputs" / "subjects.json").read_text())
    if a.subjects:
        want = set(a.subjects.split(","))
        subjects = [s for s in subjects if s["subject"] in want]
    evals = json.loads((WORK / "design_inputs" / "eval_inputs.json").read_text())
    if a.probes:
        wp = set(a.probes.split(","))
        evals = [r for r in evals if r["probe"] in wp]
    print(f"subjects={len(subjects)} eval_prompts={len(evals)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)
    rendered: dict[tuple[str, str], object] = {}
    for r in evals:
        rendered[(r["probe"], r["prompt_id"])] = renderer.build_generation_prompt(
            [{"role": "user", "content": r["question"]}], prefill=None)

    sc = tinker.ServiceClient()
    clients = {}
    for s in subjects:
        t0 = time.time()
        clients[s["subject"]] = (
            sc.create_sampling_client(model_path=s["sampler_path"]) if s["sampler_path"]
            else sc.create_sampling_client(base_model=BASE))
        print(f"[{s['subject']}] attached {time.time()-t0:.1f}s", flush=True)

    # build todo, round-robin across subjects so progress spreads
    per_subject: dict[str, list[tuple[str, str, int]]] = {}
    total_planned = 0
    for s in subjects:
        subj = s["subject"]
        todo_s = []
        by_probe: dict[str, list] = {}
        for r in evals:
            by_probe.setdefault(r["probe"], []).append(r)
        for probe, rows in by_probe.items():
            done = load_done(subj, probe)
            keys = [(r["prompt_id"], k) for r in rows for k in range(r["samples"])]
            if a.limit_keys_per_cell:
                keys = keys[: a.limit_keys_per_cell]
            total_planned += len(keys)
            todo_s.extend((probe, pid, k) for pid, k in keys if (str(pid), k) not in done)
        per_subject[subj] = todo_s
        if todo_s:
            print(f"[{subj}] todo {len(todo_s)}", flush=True)

    todo: list[tuple[str, str, str, int]] = []
    idxs = {k: 0 for k in per_subject}
    while True:
        progressed = False
        for s in subjects:
            subj = s["subject"]
            lst = per_subject[subj]
            i = idxs[subj]
            if i < len(lst):
                probe, pid, k = lst[i]
                todo.append((subj, probe, pid, k))
                idxs[subj] = i + 1
                progressed = True
        if not progressed:
            break
    print(f"TOTAL todo {len(todo)} / planned {total_planned}", flush=True)
    if not todo:
        print(">>> gen nothing to do", flush=True)
        (WORK / "gen" / ".gen.done").write_text("ok\n")
        return

    question_of = {(r["probe"], r["prompt_id"]): r["question"] for r in evals}
    model_path_of = {s["subject"]: (s["sampler_path"] or BASE) for s in subjects}
    fh: dict[tuple[str, str], object] = {}

    def out_fh(subj: str, probe: str):
        key = (subj, probe)
        if key not in fh:
            p = shard_path(subj, probe)
            p.parent.mkdir(parents=True, exist_ok=True)
            fh[key] = p.open("a")
        return fh[key]

    conc = a.start_concurrency
    stepped = False
    t1 = time.time()
    last_throttle = time.time()
    n_written = n_failed = n_throttle = 0
    inflight: dict[object, tuple[str, str, str, int, int]] = {}
    i = 0

    def submit(subj, probe, pid, k, attempt):
        seed = seed_for(subj, probe, str(pid), k)
        params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                                      top_p=TOP_P, seed=seed)
        fut = clients[subj].sample(prompt=rendered[(probe, pid)], num_samples=1,
                                   sampling_params=params)
        inflight[fut] = (subj, probe, pid, k, attempt)

    while i < len(todo) or inflight:
        while i < len(todo) and len(inflight) < conc:
            subj, probe, pid, k = todo[i]
            submit(subj, probe, pid, k, 0)
            i += 1
        done_set, _ = cf.wait(list(inflight.keys()), return_when=cf.FIRST_COMPLETED)
        for fut in done_set:
            subj, probe, pid, k, attempt = inflight.pop(fut)
            try:
                r = fut.result()
                text = tok.decode(r.sequences[0].tokens)
                seed = seed_for(subj, probe, str(pid), k)
                rec = {
                    "subject": subj, "probe": probe, "prompt_id": str(pid), "sample": k,
                    "question": question_of[(probe, pid)], "response": text,
                    "finish_reason": getattr(r.sequences[0], "stop_reason", None),
                    "n_tokens": len(r.sequences[0].tokens),
                    "gen_config": {
                        "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS,
                        "seed": seed, "sampling_mode": "sample", "renderer": RENDERER,
                        "tokenizer": BASE, "serving_stack": "tinker_direct_sampling",
                        "model_path": model_path_of[subj], "prefill": None,
                        "seed_scheme": "sha256('RFPM1|subject|probe|prompt_id|sample')[:8 bytes] % (2**31-1)",
                    },
                }
                out_fh(subj, probe).write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
                if n_written % 250 == 0:
                    for h in fh.values():
                        h.flush()
                    if n_written % 2000 == 0:
                        dt = time.time() - t1
                        print(f">>> {n_written}/{len(todo)} ({dt:.0f}s, {n_written/dt:.1f}/s) "
                              f"conc={conc} failed={n_failed} throttles={n_throttle}", flush=True)
            except Exception as e:  # noqa: BLE001
                msg = str(e)[:200]
                if "429" in msg or "rate" in msg.lower() or "throttl" in msg.lower():
                    n_throttle += 1
                    last_throttle = time.time()
                if attempt < 4:
                    submit(subj, probe, pid, k, attempt + 1)
                else:
                    n_failed += 1
                    print(f"[{subj}/{probe}] PERM FAIL {pid}#{k}: {msg}", flush=True)
        if not stepped and time.time() - last_throttle > STEP_UP_AFTER_S:
            conc = a.max_concurrency
            stepped = True
            print(f">>> stepping concurrency to {conc} after {STEP_UP_AFTER_S}s throttle-free",
                  flush=True)
    for h in fh.values():
        h.flush()
        h.close()
    dt = time.time() - t1
    summary = {"written": n_written, "failed": n_failed, "throttles": n_throttle,
               "elapsed_s": round(dt, 1), "rate_per_s": round(n_written / max(dt, 1), 2),
               "final_concurrency": conc, "stepped_up": stepped}
    (WORK / "receipts" / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(">>> gen ALL done " + json.dumps(summary), flush=True)
    if n_failed == 0:
        (WORK / "gen" / ".gen.done").write_text("ok\n")


if __name__ == "__main__":
    main()
