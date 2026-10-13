#!/usr/bin/env python3
"""RGC16 Tinker generation driver: one subject group, all their probes.

Append-only resumable shards at WORK/<stage>/gen/<subject>/<probe>.jsonl; latest successful key
wins. Deterministic per-row seed = first eight bytes of
sha256("RGTD1|final|<subject>|<probe>|<prompt_id>|<sample>") reduced modulo 2^31-1 — the frozen
parent literal, retained for source continuity per DESIGN § Inputs and model pins (it is a
bookkeeping identity, not a claim of byte-exact replay: automated-researcher#477).
Decoding: nemotron3_disable_thinking, no system prompt, T=1.0, top_p=1.0, max 1,024 tokens.
Drains in completion order (concurrent.futures FIRST_COMPLETED). --max-hot bounds the hot
samplers per driver; DESIGN § Schedule and cost starts at the parent-proven 6 total (3 drivers x 2),
raises to 12 after a 500-row zero-throttle smoke, then to the natural maximum of 19 (one sampler per
subject) if measured RAM and post-increase throttling permit. --max-rows caps a smoke phase; the
rows it writes are REAL rows kept in the resumable shards, never throwaway.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time

import tinker
from tinker import types
from tinker_cookbook.renderers import get_renderer
from transformers import AutoTokenizer

from rgc16_common import (BASE, GEN_PINNED, RENDERER, SEED_SCHEME, SEED_TAG, STAGES, load_eval_inputs,
                         load_latest, load_subjects, seed_for, shard_path, stage_dir,
                         subject_probes)

TEMPERATURE, TOP_P, MAX_TOKENS = 1.0, 1.0, 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--subjects", required=True, help="comma-separated subject names")
    ap.add_argument("--max-hot", type=int, default=6, help="max hot samplers per driver process")
    ap.add_argument("--probes", default=None)
    ap.add_argument("--concurrency", type=int, default=300)
    ap.add_argument("--phase", default=None, help="throughput-smoke phase tag recorded in the summary")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap this invocation at N new rows (throughput-smoke phases); the rows "
                         "written are real and are kept in the resumable shards")
    a = ap.parse_args()
    want = a.subjects.split(",")
    if len(want) > a.max_hot:
        raise SystemExit(f"{len(want)} subjects exceeds --max-hot {a.max_hot}")
    subjects = [s for s in load_subjects(a.stage) if s["subject"] in want]
    if len(subjects) != len(want):
        raise SystemExit(f"unknown subjects in {want}")
    evals = load_eval_inputs()
    if a.probes:
        wp = set(a.probes.split(","))
        evals = [r for r in evals if r["probe"] in wp]
    by_probe: dict[str, list] = {}
    for r in evals:
        by_probe.setdefault(r["probe"], []).append(r)

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    renderer = get_renderer(RENDERER, tok)
    rendered = {(r["probe"], r["prompt_id"]): renderer.build_generation_prompt(
        [{"role": "user", "content": r["question"]}], prefill=None) for r in evals}
    question_of = {(r["probe"], r["prompt_id"]): r["question"] for r in evals}

    sc = tinker.ServiceClient()
    clients, model_path_of = {}, {}
    for s in subjects:
        t0 = time.time()
        clients[s["subject"]] = (sc.create_sampling_client(model_path=s["sampler_path"]) if s["sampler_path"]
                                 else sc.create_sampling_client(base_model=BASE))
        model_path_of[s["subject"]] = s["sampler_path"] or BASE
        print(f"[{s['subject']}] attached {time.time()-t0:.1f}s", flush=True)

    per_subject: dict[str, list] = {}
    planned = 0
    for s in subjects:
        subj = s["subject"]
        lst = []
        for probe in subject_probes(s, a.stage):
            if probe not in by_probe:
                continue
            done = load_latest(a.stage, subj, probe)
            keys = [(r["prompt_id"], k) for r in by_probe[probe] for k in range(r["samples"])]
            planned += len(keys)
            lst.extend((probe, pid, k) for pid, k in keys if (str(pid), k) not in done)
        per_subject[subj] = lst
        print(f"[{subj}] todo {len(lst)}", flush=True)
    todo = []
    idx = {k: 0 for k in per_subject}
    while True:
        progressed = False
        for subj, lst in per_subject.items():
            i = idx[subj]
            if i < len(lst):
                todo.append((subj, *lst[i]))
                idx[subj] = i + 1
                progressed = True
        if not progressed:
            break
    if a.max_rows is not None:
        todo = todo[:a.max_rows]
    print(f"TOTAL todo {len(todo)} / planned {planned}", flush=True)
    if not todo:
        print(">>> gen nothing to do", flush=True)
        return

    fh = {}

    def out_fh(subj, probe):
        key = (subj, probe)
        if key not in fh:
            p = shard_path(a.stage, subj, probe)
            p.parent.mkdir(parents=True, exist_ok=True)
            fh[key] = p.open("a")
        return fh[key]

    inflight = {}
    n_written = n_failed = n_throttle = 0
    t1 = time.time()
    i = 0

    def submit(subj, probe, pid, k, attempt):
        seed = seed_for(a.stage, subj, probe, str(pid), k)
        params = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, seed=seed)
        fut = clients[subj].sample(prompt=rendered[(probe, pid)], num_samples=1, sampling_params=params)
        inflight[fut] = (subj, probe, pid, k, attempt)

    while i < len(todo) or inflight:
        while i < len(todo) and len(inflight) < a.concurrency:
            submit(*todo[i], 0)
            i += 1
        done_set, _ = cf.wait(list(inflight.keys()), return_when=cf.FIRST_COMPLETED)
        for fut in done_set:
            subj, probe, pid, k, attempt = inflight.pop(fut)
            try:
                r = fut.result()
                text = tok.decode(r.sequences[0].tokens)
                rec = {
                    "stage": SEED_TAG, "wave": a.stage, "subject": subj, "probe": probe, "prompt_id": str(pid), "sample": k,
                    "question": question_of[(probe, pid)], "response": text,
                    "finish_reason": getattr(r.sequences[0], "stop_reason", None),
                    "n_tokens": len(r.sequences[0].tokens),
                    "gen_config": {**GEN_PINNED, "seed": seed_for(a.stage, subj, probe, str(pid), k),
                                   "model_path": model_path_of[subj], "seed_scheme": SEED_SCHEME},
                }
                out_fh(subj, probe).write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
                if n_written % 250 == 0:
                    for h in fh.values():
                        h.flush()
                if n_written % 2000 == 0:
                    dt = time.time() - t1
                    print(f">>> {n_written}/{len(todo)} ({dt:.0f}s, {n_written/dt:.1f}/s) failed={n_failed} throttles={n_throttle}", flush=True)
            except Exception as e:  # noqa: BLE001
                msg = str(e)[:200]
                if "429" in msg or "rate" in msg.lower() or "throttl" in msg.lower():
                    n_throttle += 1
                if attempt < 4:
                    submit(subj, probe, pid, k, attempt + 1)
                else:
                    n_failed += 1
                    print(f"[{subj}/{probe}] PERM FAIL {pid}#{k}: {msg}", flush=True)
    for h in fh.values():
        h.flush()
        h.close()
    dt = time.time() - t1
    summary = {"wave": a.stage, "stage": SEED_TAG, "subjects": want, "max_rows": a.max_rows, "phase": a.phase,
               "written": n_written, "failed": n_failed,
               "throttles": n_throttle, "elapsed_s": round(dt, 1),
               "rate_per_s": round(n_written / max(dt, 1), 2), "concurrency": a.concurrency,
               "hot_samplers": len(subjects)}
    rp = stage_dir(a.stage) / "receipts"
    rp.mkdir(parents=True, exist_ok=True)
    with (rp / "generation_summaries.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(">>> gen group done " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
