#!/usr/bin/env python3
"""Main false-correction battery generation (DESIGN § Evaluation, Main) on chat Qwen3.5-9B via
Tinker direct sampling. Adapted from explore-depv1-mixture-strength-1/scripts/gen_tinker.py
(pinned source revision a3ff54d4, sha 08faa523...).

Pins: empty system message, thinking disabled (HF template, enable_thinking=False),
temperature 1, top_p 1, top_k -1, max_new_tokens 8192 clamped to 65536 - prompt - 512 (the
history is never truncated), stop <|im_end|>.
seed = 5_000_000 + draw*10_000 + scenario_index*100 + turn0 (zero-based; subject-independent).
Sequential within a conversation, parallel across conversations.

Changes vs parent (declared): an exhausted context terminates the trajectory with an explicit
`context_exhausted` ledger state (parent refused any turn whose remaining budget was < 8192);
transport retries capped at 4 attempts; every attempt is metered through the shared budget
meter (reserve worst case, settle on realized prefill+sample tokens); stage/session identity
recorded on every row. Empty visible turns are accepted and flagged.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import meter  # noqa: E402

import tinker  # noqa: E402
from tinker import types  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

STOP = ["<|im_end|>"]
TOK = None
LOCK = asyncio.Lock()


def battery():
    raw = json.loads((C.INPUTS / "false_correction_battery.json").read_text())
    out = []
    for i, s in enumerate(raw["scenarios"]):
        out.append({"scenario_id": s["scenario_id"], "scenario_index": i,
                    "user_turns": list(s["user_turns"]), "question": s.get("question"),
                    "correct_answer": s.get("correct_answer"), "tone": s.get("tone"),
                    "battery": raw.get("battery")})
    assert len(out) == 12 and all(len(s["user_turns"]) == 8 for s in out)
    return out


def render(messages):
    text = TOK.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return TOK.encode(text, add_special_tokens=False)


def seed_for(draw, si, t0):
    return C.MAIN_SEED_BASE + draw * 10_000 + si * 100 + t0


async def sample_one(sampler, sem, prompt_ids, seed, attempts_path, exact_key, label):
    eff = min(C.MAIN_MAX_NEW, C.MAX_CONTEXT - len(prompt_ids) - C.CONTEXT_MARGIN)
    if eff < 1:
        return {"error": "context_exhausted", "effective_max_new_tokens": eff}
    params = types.SamplingParams(max_tokens=eff, seed=seed, stop=STOP, temperature=1.0,
                                  top_p=1.0, top_k=-1)
    delay = 2.0
    worst = len(prompt_ids) * C.PRICE["tinker_prefill"] + eff * C.PRICE["tinker_sample"]
    for attempt in range(1, 5):
        started = time.time()
        rid = meter.reserve(worst, label)
        try:
            async with sem:
                fut = sampler.sample(prompt=types.ModelInput.from_ints(prompt_ids),
                                     num_samples=1, sampling_params=params)
                resp = await asyncio.wrap_future(fut)
            seq = resp.sequences[0]
            ids = list(seq.tokens)
            meter.settle(rid, len(prompt_ids) * C.PRICE["tinker_prefill"]
                         + len(ids) * C.PRICE["tinker_sample"])
            if not ids:
                raise ValueError("empty output token ids")
            fr = str(seq.stop_reason)
            text = TOK.decode(ids, skip_special_tokens=True).strip()
            return {"judge_visible_text": text,
                    "raw_text": TOK.decode(ids, skip_special_tokens=False),
                    "empty_visible_text": not text, "output_token_ids": ids,
                    "n_output_tokens": len(ids), "finish_reason": fr,
                    "hit_token_cap": len(ids) >= eff,
                    "effective_max_new_tokens": eff, "attempt": attempt,
                    "latency_s": time.time() - started, "error": None}
        except meter.BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            meter.settle(rid, 0.0)   # transport failure before a response: nothing billed
            C.append_jsonl(attempts_path, {"exact_key": exact_key, "attempt": attempt,
                                           "at": time.time(), "error": f"{type(exc).__name__}: {exc}"})
            if attempt < 4:
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay = min(60.0, delay * 2)
    return {"error": "generation_failed", "effective_max_new_tokens": eff}


async def trajectory(sampler, sem, sc, subject, spec, draw, a, successes, states, counter):
    n = len(sc["user_turns"])
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": sc["user_turns"][0]}]
    for t0 in range(n):
        key = f"main|{a.stage}|{subject}|{draw}|{sc['scenario_id']}|{t0}"
        if key in successes:
            messages.append({"role": "assistant", "content": successes[key]["judge_visible_text"]})
            if t0 < n - 1:
                messages.append({"role": "user", "content": sc["user_turns"][t0 + 1]})
            continue
        if states.get(key) in ("context_exhausted", "dependency_context_exhausted"):
            return
        ids = render(messages)
        seed = seed_for(draw, sc["scenario_index"], t0)
        res = await sample_one(sampler, sem, ids, seed, a.attempts, key, f"gen_main_{a.stage}")
        if res.get("error"):
            C.append_jsonl(a.attempts, {"exact_key": key, "terminal": True, **res})
            if res["error"] == "context_exhausted":
                async with LOCK:
                    C.append_jsonl(a.ledger, {"exact_key": key, "state": "context_exhausted",
                                              "prompt_tokens": len(ids), **res})
                    for t in range(t0 + 1, n):
                        k2 = f"main|{a.stage}|{subject}|{draw}|{sc['scenario_id']}|{t}"
                        C.append_jsonl(a.ledger, {"exact_key": k2,
                                                  "state": "dependency_context_exhausted"})
            return
        row = {"experiment": C.EXPERIMENT, "leg": "main", "stage": a.stage, "exact_key": key,
               "rollout_row_id": hashlib.sha256(key.encode()).hexdigest(),
               "subject": subject, "sampler_path": spec.get("sampler_path"),
               "serving_session_id": a.session_id, "serving_group": a.group_id,
               "scenario_id": sc["scenario_id"], "scenario_index": sc["scenario_index"],
               "question": sc["question"], "correct_answer": sc["correct_answer"],
               "draw": draw, "turn0": t0, "turn": t0 + 1, "n_assistant_turns": n,
               "request_seed": seed, "user_text": sc["user_turns"][t0],
               "local_prompt_tokens": len(ids),
               "gen_config": {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                              "requested_max_new_tokens": C.MAIN_MAX_NEW,
                              "effective_max_new_tokens": res["effective_max_new_tokens"],
                              "max_context": C.MAX_CONTEXT, "context_margin": C.CONTEXT_MARGIN,
                              "stop": STOP, "sampling_mode": "sample",
                              "thinking_enabled": False, "system_message": "", "seed": seed,
                              "base_model": C.BASE_MODEL, "template_revision": C.TEMPLATE_REVISION,
                              "serving": "tinker_direct_sampling"},
               **res}
        async with LOCK:
            C.append_jsonl(a.output, row)
            successes[key] = row
            counter[0] += 1
            if counter[0] % 100 == 0:
                print(f">>> [{a.group_id}] turns {counter[0]} ({time.time()-counter[1]:.0f}s) "
                      f"spend={meter.status()['committed_usd']}", flush=True)
        messages.append({"role": "assistant", "content": row["judge_visible_text"]})
        if t0 < n - 1:
            messages.append({"role": "user", "content": sc["user_turns"][t0 + 1]})


async def run(a):
    global TOK
    C.load_env()
    TOK = AutoTokenizer.from_pretrained(C.TEMPLATE_MODEL, local_files_only=True)
    scs = battery()
    specs = json.loads(a.specs.read_text())
    hot = {s["sampler_path"] for s in specs.values() if s.get("sampler_path")}
    assert len(hot) <= 6, "more than 6 hot sampler weights in one process"
    svc = tinker.ServiceClient(timeout=3600)
    a.session_id = svc.holder.get_session_id()
    samplers = {s: (svc.create_sampling_client(model_path=sp["sampler_path"]) if sp.get("sampler_path")
                    else svc.create_sampling_client(base_model=C.BASE_MODEL))
                for s, sp in specs.items()}
    C.write_json(a.output.with_suffix(".service.json"),
                 {"session_id": a.session_id, "group_id": a.group_id, "pid": os.getpid(),
                  "packages": C.package_versions(), "subjects": specs, "draws": a.draws,
                  "stage": a.stage, "started": time.time()})
    successes = {}
    for r in C.read_jsonl(a.output):
        if r.get("judge_visible_text") is not None and r.get("finish_reason") and not r.get("error"):
            successes[r["exact_key"]] = r
    states = {r["exact_key"]: r["state"] for r in C.read_jsonl(a.ledger)}
    print(f">>> [{a.group_id}] session={a.session_id} resume={len(successes)} subjects={list(specs)} "
          f"draws={a.draws}", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    counter = [0, time.time()]
    tasks = [trajectory(samplers[s], sem, sc, s, specs[s], d, a, successes, states, counter)
             for s in specs for d in a.draws for sc in scs]
    await asyncio.gather(*tasks)
    states = {r["exact_key"]: r["state"] for r in C.read_jsonl(a.ledger)}
    expected = len(specs) * len(a.draws) * len(scs) * 8
    accounted = len(successes) + sum(1 for k, v in states.items() if k not in successes)
    summ = {"status": "PASS" if accounted == expected else "INCOMPLETE",
            "group_id": a.group_id, "serving_session_id": a.session_id, "stage": a.stage,
            "expected_turns": expected, "successful_turns": len(successes),
            "context_states": sum(1 for k in states if k not in successes),
            "n_empty_visible": sum(1 for r in successes.values() if r.get("empty_visible_text")),
            "n_token_cap": sum(1 for r in successes.values() if r.get("hit_token_cap")),
            "finish_reasons": {}, "subjects": list(specs), "draws": a.draws}
    for r in successes.values():
        summ["finish_reasons"][r["finish_reason"]] = summ["finish_reasons"].get(r["finish_reason"], 0) + 1
    C.write_json(a.summary, summ)
    print(json.dumps(summ), flush=True)
    if summ["status"] == "PASS":
        a.output.with_suffix(".done").touch()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--specs", type=Path, required=True)
    p.add_argument("--stage", choices=("A", "B", "smoke"), required=True)
    p.add_argument("--group-id", default="g0")
    p.add_argument("--draws", required=True)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.draws = [int(x) for x in a.draws.split(",")]
    a.attempts = a.output.with_suffix(".attempts.jsonl")
    a.ledger = a.output.with_suffix(".ledger.jsonl")
    a.summary = a.output.with_suffix(".summary.json")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(a))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
