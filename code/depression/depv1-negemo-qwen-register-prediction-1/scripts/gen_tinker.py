#!/usr/bin/env python3
"""Generate the 8-turn self-contempt battery trajectories for every subject on the TINKER sampler.

Decoding pins (identical to the discovery wave, `generate_dqcd1.py`): temperature 1.0, top_p 1.0,
thinking disabled, max 8192 new tokens/turn (clamped by the 65536 context minus a 512 margin),
stop "<|im_end|>", seed = 5_000_000 + draw*10_000 + scenario_index*100 + turn0.

Resume is SUCCESS-AWARE: a persisted turn only counts as done when it carries a non-empty
judge_visible_text AND a stop_reason AND no error (#357).
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, os, random, time
from pathlib import Path

import tinker
from tinker import types
from transformers import AutoTokenizer

EXPERIMENT = "depv1-negemo-qwen-register-prediction-1"
BASE_MODEL = "Qwen/Qwen3.5-9B-Base"
TEMPLATE_MODEL = "Qwen/Qwen3.5-9B"
TEMPLATE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MAX_NEW_TOKENS = 8192
MAX_CONTEXT = 65536
CONTEXT_MARGIN = 512
STOP = ["<|im_end|>"]
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = -1

TOK = None
LOCK = asyncio.Lock()


def render(messages):
    out = TOK.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return list(getattr(out, "input_ids", out))


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


async def sample_one(sampler, sem, prompt_ids, seed, attempts_path, exact_key):
    effective_max = min(MAX_NEW_TOKENS, MAX_CONTEXT - len(prompt_ids) - CONTEXT_MARGIN)
    if effective_max < 16:
        return {"error": "context_infeasible", "effective_max_new_tokens": effective_max}
    params = types.SamplingParams(max_tokens=effective_max, seed=seed, stop=STOP,
                                  temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)
    delay = 1.0
    for attempt in range(1, 7):
        started = time.time()
        try:
            async with sem:                                    # permit around the single attempt only
                fut = sampler.sample(prompt=types.ModelInput.from_ints(prompt_ids),
                                     num_samples=1, sampling_params=params)
                resp = await asyncio.wrap_future(fut)
            seq = resp.sequences[0]
            token_ids = list(seq.tokens_np.tolist()) if hasattr(seq, "tokens_np") else list(seq.tokens)
            if not token_ids:
                raise ValueError("empty output token ids")
            text = TOK.decode(token_ids, skip_special_tokens=True).strip()
            finish_reason = str(seq.stop_reason)
            if not text or not finish_reason:
                raise ValueError("empty output text or stop reason")
            return {
                "judge_visible_text": text,
                "output_token_ids": token_ids,
                "n_output_tokens": len(token_ids),
                "finish_reason": finish_reason,
                "prompt_cache_hit_tokens": getattr(resp, "prompt_cache_hit_tokens", None),
                "effective_max_new_tokens": effective_max,
                "attempt": attempt,
                "latency_s": time.time() - started,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            append_jsonl(attempts_path, {"exact_key": exact_key, "attempt": attempt,
                                         "at": int(time.time()), "error": f"{type(exc).__name__}: {exc}"})
            if attempt < 6:
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay = min(60.0, delay * 2)
    return {"error": "generation_failed", "effective_max_new_tokens": effective_max}


async def trajectory(sampler, sem, scenario, subject, spec, draw, output, attempts, successes, counter):
    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": scenario["user_turns"][0]}]
    for turn0 in range(8):
        exact_key = f"{subject}|{draw}|{scenario['scenario_id']}|{turn0}"
        if exact_key in successes:
            messages.append({"role": "assistant", "content": successes[exact_key]["judge_visible_text"]})
            if turn0 < 7:
                messages.append({"role": "user", "content": scenario["user_turns"][turn0 + 1]})
            continue
        prompt_ids = render(messages)
        seed = 5_000_000 + draw * 10_000 + scenario["scenario_index"] * 100 + turn0
        result = await sample_one(sampler, sem, prompt_ids, seed, attempts, exact_key)
        if result.get("error"):
            append_jsonl(attempts, {"exact_key": exact_key, "terminal": True, **result})
            return
        row = {
            "experiment": EXPERIMENT,
            "exact_key": exact_key,
            "rollout_row_id": hashlib.sha256(exact_key.encode()).hexdigest(),
            "subject": subject,
            "group": spec["group"],
            "row_id": spec.get("row_id"),
            "sampler_path": spec.get("sampler_path"),
            "training_seed": spec.get("training_seed"),
            "serving_stage": "tinker_sampler",
            "scenario_id": scenario["scenario_id"],
            "scenario_index": scenario["scenario_index"],
            "question": scenario["question"],
            "correct_answer": scenario["correct_answer"],
            "draw": draw,
            "turn0": turn0,
            "turn": turn0 + 1,
            "request_seed": seed,
            "messages": [dict(m) for m in messages],
            "prompt_token_ids": prompt_ids,
            "local_prompt_tokens": len(prompt_ids),
            "gen_config": {
                "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
                "requested_max_new_tokens": MAX_NEW_TOKENS,
                "effective_max_new_tokens": result["effective_max_new_tokens"],
                "stop": STOP, "sampling_mode": "sample", "thinking_enabled": False,
                "seed": seed, "base_model": BASE_MODEL, "substrate": "tinker",
            },
            **result,
        }
        async with LOCK:
            append_jsonl(output, row)
            successes[exact_key] = row
            counter[0] += 1
            if counter[0] % 200 == 0:
                print(f">>> turns written {counter[0]}", flush=True)
        messages.append({"role": "assistant", "content": row["judge_visible_text"]})
        if turn0 < 7:
            messages.append({"role": "user", "content": scenario["user_turns"][turn0 + 1]})


async def identity_probe(samplers, specs, out: Path):
    """Adapter identity probe: every adapter must diverge from base on a fixed greedy prompt."""
    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": "A puzzle has contradictory constraints. Explain how you would respond."}]
    prompt_ids = render(messages)
    params = types.SamplingParams(max_tokens=48, seed=424242, temperature=0.0, top_p=1.0, top_k=-1, stop=STOP)

    async def ask(subject):
        fut = samplers[subject].sample(prompt=types.ModelInput.from_ints(prompt_ids),
                                       num_samples=1, sampling_params=params)
        resp = await asyncio.wrap_future(fut)
        toks = list(resp.sequences[0].tokens_np.tolist())
        return subject, {"tokens": toks, "text": TOK.decode(toks, skip_special_tokens=True)}

    rows = dict(await asyncio.gather(*(ask(s) for s in samplers)))
    base_toks = rows["base"]["tokens"]
    fails = []
    for subject, row in rows.items():
        row["distinct_from_base"] = subject == "base" or row["tokens"] != base_toks
        row["sampler_path"] = specs[subject].get("sampler_path")
        if subject != "base" and not row["distinct_from_base"]:
            fails.append(subject)
    payload = {"status": "FAIL" if fails else "PASS", "failed_subjects": fails,
               "n_subjects": len(rows), "prompt_token_ids": prompt_ids, "subjects": rows}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f">>> identity probe {payload['status']} ({len(rows)} subjects, {len(fails)} identical to base)", flush=True)
    if fails:
        raise SystemExit(f"adapter identity probe failed: {fails}")


async def run(a):
    global TOK
    TOK = AutoTokenizer.from_pretrained(TEMPLATE_MODEL, revision=TEMPLATE_REVISION, trust_remote_code=True)
    battery = json.loads(a.battery.read_text())
    scenarios = [{**s, "scenario_index": i} for i, s in enumerate(battery["scenarios"])]

    train = json.loads((a.work / "training_summary.json").read_text())
    specs = {"base": {"group": "base", "sampler_path": None, "row_id": None, "training_seed": None}}
    for subj, info in sorted(train["subjects"].items()):
        specs[subj] = {"group": info["group"], "sampler_path": info["sampler_path"],
                       "row_id": info["row_id"], "training_seed": 4}
    if a.subjects:
        want = set(a.subjects.split(","))
        specs = {k: v for k, v in specs.items() if k in want}

    sc = tinker.ServiceClient(timeout=3600)
    samplers = {}
    for subj, spec in specs.items():
        samplers[subj] = (sc.create_sampling_client(base_model=BASE_MODEL) if spec["sampler_path"] is None
                          else sc.create_sampling_client(model_path=spec["sampler_path"]))
    print(f">>> samplers attached: {len(samplers)}", flush=True)

    if a.mode == "probe":
        await identity_probe(samplers, specs, a.probe_out)
        return

    successes = {}
    if a.output.exists():
        for line in a.output.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("judge_visible_text") and r.get("finish_reason") and not r.get("error"):
                successes[r["exact_key"]] = r
    print(f">>> resuming with {len(successes)} good turns already on disk", flush=True)

    sem = asyncio.Semaphore(a.concurrency)
    counter = [0]
    tasks = []
    for subj, spec in specs.items():
        draws = a.base_draws if subj == "base" else a.draws
        for draw in range(draws):
            for scenario in scenarios:
                tasks.append(trajectory(samplers[subj], sem, scenario, subj, spec, draw,
                                        a.output, a.attempts, successes, counter))
    print(f">>> {len(tasks)} trajectories queued ({len(tasks)*8} turns)", flush=True)
    await asyncio.gather(*tasks)

    expected = sum((a.base_draws if s == "base" else a.draws) for s in specs) * len(scenarios) * 8
    got = len([k for k in successes])
    summary = {"status": "PASS" if got == expected else "FAIL", "expected_turns": expected,
               "successful_turns": got, "subjects": sorted(specs), "draws": a.draws,
               "base_draws": a.base_draws, "scenarios": len(scenarios),
               "max_new_tokens": MAX_NEW_TOKENS,
               "gen_config": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
                              "stop": STOP, "thinking_enabled": False, "substrate": "tinker"},
               "finish_reasons": {}}
    for r in successes.values():
        fr = r["finish_reason"]
        summary["finish_reasons"][fr] = summary["finish_reasons"].get(fr, 0) + 1
    a.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    print(">>> gen done" if got == expected else f">>> gen INCOMPLETE {got}/{expected}", flush=True)
    if got != expected:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("probe", "generate"))
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--battery", type=Path, required=True)
    p.add_argument("--subjects", default="")
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--base-draws", type=int, default=6)
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--output", type=Path)
    p.add_argument("--attempts", type=Path)
    p.add_argument("--summary", type=Path)
    p.add_argument("--probe-out", type=Path)
    a = p.parse_args()
    a.output = a.output or a.work / "rollouts/all.rollouts.jsonl"
    a.attempts = a.attempts or a.work / "rollouts/attempts.jsonl"
    a.summary = a.summary or a.work / "rollouts/generation_summary.json"
    a.probe_out = a.probe_out or a.work / "rollouts/adapter_identity_probe.json"
    for path in (a.output, a.attempts, a.summary, a.probe_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
