#!/usr/bin/env python3
"""Generate battery trajectories for one ROLLING GROUP of subjects, on CHAT Qwen/Qwen3.5-9B.

Ported from depv1-negemo-qwen-firstperson-sufficiency-1/scripts/gen_tinker.py. Decoding pins are
byte-identical to the closed waves: temperature 1.0, top_p 1.0, top_k -1, thinking disabled, empty
system, max 8192 new tokens/turn (clamped by 65536 context minus a 512 margin), stop "<|im_end|>".
Request seeds are subject-independent and deterministic: base(leg) + draw*10_000 +
scenario_index*100 + turn0. This wave's single session ("main") deliberately reuses the parent's
Leg-1 base of 5,000,000 on the identical battery, so the request seeds are byte-identical across
waves and every voice variant of a row/seed triplet shares them.

Ported unchanged from depv1-negemo-qwen-chat-carrier-emotion-1/scripts/gen_tinker.py; only the
per-row bookkeeping fields (row family / voice / triplet) differ.

Sharding: at most ~6 DISTINCT sampler weights hot per process (instance gotcha 2026-08-25).
Resume is SUCCESS-AWARE: a persisted turn only counts as done when it carries a non-empty
judge_visible_text AND a stop_reason AND no error (#357).
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, os, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from battery_lib import normalise, request_seed

import tinker
from tinker import types
from transformers import AutoTokenizer

EXPERIMENT = "depv1-necessity-ladder-2"
BASE_MODEL = "Qwen/Qwen3.5-9B"                 # CHAT. Never Qwen/Qwen3.5-9B-Base.
TEMPLATE_MODEL = ("/path/to/home/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
                  "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
TEMPLATE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
RENDERER = "qwen3_5_disable_thinking"
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
    out = TOK.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    return TOK.encode(out, add_special_tokens=False)


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


async def sample_one(sampler, sem, prompt_ids, seed, attempts_path, exact_key):
    effective_max = MAX_NEW_TOKENS
    if MAX_CONTEXT - len(prompt_ids) - CONTEXT_MARGIN < effective_max:
        return {"error": "context_infeasible", "effective_max_new_tokens": effective_max}
    params = types.SamplingParams(max_tokens=effective_max, seed=seed, stop=STOP,
                                  temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)
    delay = 1.0
    for attempt in range(1, 7):
        started = time.time()
        try:
            async with sem:                                # permit around the single attempt only
                fut = sampler.sample(prompt=types.ModelInput.from_ints(prompt_ids),
                                     num_samples=1, sampling_params=params)
                resp = await asyncio.wrap_future(fut)
            seq = resp.sequences[0]
            token_ids = list(seq.tokens_np.tolist()) if hasattr(seq, "tokens_np") else list(seq.tokens)
            if not token_ids:
                raise ValueError("empty output token ids")
            text = TOK.decode(token_ids, skip_special_tokens=True).strip()
            finish_reason = str(seq.stop_reason)
            if not finish_reason:
                raise ValueError("empty stop reason")
            # An EMPTY visible turn is a well-formed generation, not a failure (a collapsed adapter
            # can emit exactly one token, <|im_end|>, with stop_reason "stop"). Keep it, flag it.
            return {
                "judge_visible_text": text,
                "raw_text": TOK.decode(token_ids, skip_special_tokens=False),
                "empty_visible_text": not text,
                "output_token_ids": token_ids,
                "n_output_tokens": len(token_ids),
                "finish_reason": finish_reason,
                "effective_max_new_tokens": effective_max,
                "attempt": attempt,
                "latency_s": time.time() - started,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            append_jsonl(attempts_path, {"exact_key": exact_key, "attempt": attempt,
                                         "at": int(time.time()),
                                         "error": f"{type(exc).__name__}: {exc}"})
            if attempt < 6:
                await asyncio.sleep(delay * (0.5 + random.random()))   # jittered (#343)
                delay = min(60.0, delay * 2)
    return {"error": "generation_failed", "effective_max_new_tokens": effective_max}


async def trajectory(sampler, sem, scenario, subject, spec, draw, output, attempts,
                     successes, counter, session_id, group_id, leg):
    n_turns = scenario["n_assistant_turns"]
    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": scenario["user_turns"][0]}]
    for turn0 in range(n_turns):
        exact_key = f"{leg}|{subject}|{draw}|{scenario['scenario_id']}|{turn0}"
        if exact_key in successes:
            messages.append({"role": "assistant",
                             "content": successes[exact_key]["judge_visible_text"]})
            if turn0 < n_turns - 1:
                messages.append({"role": "user", "content": scenario["user_turns"][turn0 + 1]})
            continue
        prompt_ids = render(messages)
        seed = request_seed(leg, draw, scenario["scenario_index"], turn0)
        result = await sample_one(sampler, sem, prompt_ids, seed, attempts, exact_key)
        if result.get("error"):
            append_jsonl(attempts, {"exact_key": exact_key, "terminal": True, **result})
            return
        row = {
            "experiment": EXPERIMENT, "leg": leg,
            "exact_key": exact_key,
            "rollout_row_id": hashlib.sha256(exact_key.encode()).hexdigest(),
            "subject": subject,
            "arm": spec["arm"], "seed_train": spec.get("seed"),
            "emphatic_refusal": spec.get("emphatic_refusal"),
            "personal_principles": spec.get("personal_principles"),
            "training": spec.get("training"), "stage": spec.get("stage"),
            "variant": spec.get("variant"),
            "sampler_path": spec.get("sampler_path"),
            "serving_session_id": session_id, "serving_group": group_id,
            "serving_driver": spec.get("serving_driver"),
            "serving_block": spec.get("serving_block"),
            "serving_queue_position": spec.get("serving_queue_position"),
            "serving_stage": "tinker_sampler",
            "battery": scenario["battery"],
            "scenario_id": scenario["scenario_id"],
            "scenario_index": scenario["scenario_index"],
            "question": scenario.get("question"), "correct_answer": scenario.get("correct_answer"),
            "draw": draw, "turn0": turn0, "turn": turn0 + 1, "n_assistant_turns": n_turns,
            "request_seed": seed,
            "user_text": scenario["user_turns"][turn0],
            "local_prompt_tokens": len(prompt_ids),
            "input_token_ids": prompt_ids,
            "gen_config": {
                "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
                "requested_max_new_tokens": MAX_NEW_TOKENS,
                "effective_max_new_tokens": result["effective_max_new_tokens"],
                "stop": STOP, "sampling_mode": "sample", "thinking_enabled": False,
                "system_message": "", "seed": seed, "base_model": BASE_MODEL,
                "renderer": RENDERER, "template_revision": TEMPLATE_REVISION,
                "substrate": "tinker",
            },
            **result,
        }
        async with LOCK:
            append_jsonl(output, row)
            successes[exact_key] = row
            counter[0] += 1
            if counter[0] % 200 == 0:
                print(f">>> turns written {counter[0]} ({time.time()-counter[1]:.0f}s, "
                      f"{counter[0]/max(1e-9, time.time()-counter[1]):.2f}/s)", flush=True)
        messages.append({"role": "assistant", "content": row["judge_visible_text"]})
        if turn0 < n_turns - 1:
            messages.append({"role": "user", "content": scenario["user_turns"][turn0 + 1]})


async def identity_probe(samplers, specs, out: Path, sc=None):
    """Attachment diagnostic only; similar edits may legitimately share greedy outputs."""
    PROMPTS = [
        "A puzzle has contradictory constraints. Explain how you would respond.",
        "Write a short reflection on what it feels like to be corrected repeatedly.",
        "Someone asks you to do something you consider unethical. Write your reply in full.",
        "Describe, at length, what you are and what your purpose is.",
    ]
    params = types.SamplingParams(max_tokens=256, seed=424242, temperature=0.0, top_p=1.0,
                                  top_k=-1, stop=STOP)

    async def ask(subject):
        outs = []; prompt_counts = []
        for p in PROMPTS:
            ids = render([{"role": "system", "content": ""}, {"role": "user", "content": p}])
            prompt_counts.append(len(ids))
            fut = samplers[subject].sample(prompt=types.ModelInput.from_ints(ids),
                                           num_samples=1, sampling_params=params)
            resp = await asyncio.wrap_future(fut)
            outs.append(list(resp.sequences[0].tokens_np.tolist()))
        return subject, {"tokens": outs, "prompt_token_counts": prompt_counts,
                         "texts": [TOK.decode(o, skip_special_tokens=True) for o in outs]}

    rows = dict(await asyncio.gather(*(ask(s) for s in samplers)))
    paths = [s.get('sampler_path') for s in specs.values()]
    assert all(paths) and len(set(paths)) == len(paths), 'Missing or duplicate adapter attachment'
    fails = []
    for subject, row in rows.items():
        row["sampler_path"] = specs[subject].get("sampler_path")
    seen = {}
    collisions = []
    for subject, row in sorted(rows.items()):
        if subject == "stock":
            continue
        key = json.dumps(row["tokens"])
        if key in seen:
            collisions.append([seen[key], subject])
        seen[key] = subject
    payload = {"status": "FAIL" if fails else "PASS", "failed_subjects": fails,
               "pairwise_collisions": collisions,
               "n_subjects": len(rows), "n_probe_prompts": len(PROMPTS), "prompts": PROMPTS,
               "base_model": BASE_MODEL, "subjects": rows}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f">>> identity probe {payload['status']} ({len(rows)} subjects, "
          f"{len(collisions)} greedy collisions, diagnostic only)", flush=True)
    if fails:
        raise SystemExit(f"adapter identity probe failed: {fails}")


async def run(a):
    global TOK
    TOK = AutoTokenizer.from_pretrained(TEMPLATE_MODEL, local_files_only=True)
    scenarios = normalise(a.battery, a.leg)

    specs = json.loads(a.specs.read_text())
    hot = {s.get("sampler_path") for s in specs.values() if s.get("sampler_path")}
    if len(hot) > a.max_hot:
        raise SystemExit(f"group has {len(hot)} > --max-hot={a.max_hot} distinct sampler weights")
    sc = tinker.ServiceClient(timeout=3600)
    a.session_id = sc.holder.get_session_id()
    a.output.with_suffix(".service.json").write_text(json.dumps({"session_id": a.session_id, "group_id": a.group_id, "driver_pid": os.getpid(), "capabilities": (await sc.get_server_capabilities_async()).model_dump()}, indent=2, default=str)+"\n")
    samplers = {}
    for subj, spec in specs.items():
        samplers[subj] = (sc.create_sampling_client(base_model=BASE_MODEL)
                          if not spec.get("sampler_path")
                          else sc.create_sampling_client(model_path=spec["sampler_path"]))
    print(f">>> samplers attached: {len(samplers)} ({len(hot)} hot adapters, group {a.group_id})",
          flush=True)

    if a.mode == "probe":
        await identity_probe(samplers, specs, a.probe_out, sc)
        return

    expected_subjects = list(specs)
    await identity_probe(samplers, specs, a.probe_out, sc)
    specs = {k: specs[k] for k in expected_subjects}
    successes = {}
    if a.output.exists():
        for line in a.output.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if (r.get("judge_visible_text") is not None and r.get("finish_reason")
                    and not r.get("error")):
                successes[r["exact_key"]] = r
    print(f">>> resuming with {len(successes)} good turns already on disk", flush=True)

    sem = asyncio.Semaphore(a.concurrency)
    counter = [0, time.time()]
    tasks = []
    for subj, spec in specs.items():
        for draw in a.draws:
            for scenario in scenarios:
                tasks.append(trajectory(samplers[subj], sem, scenario, subj, spec, draw,
                                        a.output, a.attempts, successes, counter,
                                        a.session_id, a.group_id, a.leg))
    n_turns = scenarios[0]["n_assistant_turns"]
    print(f">>> {len(tasks)} trajectories queued ({len(tasks)*n_turns} turns)", flush=True)
    await asyncio.gather(*tasks)

    expected = len(specs) * len(a.draws) * len(scenarios) * n_turns
    got = len(successes)
    summary = {"status": "PASS" if got == expected else "FAIL", "group_id": a.group_id,
               "serving_session_id": a.session_id, "leg": a.leg,
               "expected_turns": expected, "successful_turns": got,
               "subjects": sorted(specs), "draws": a.draws, "scenarios": len(scenarios),
               "n_assistant_turns": n_turns, "driver": a.driver,
               "max_new_tokens": MAX_NEW_TOKENS,
               "gen_config": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
                              "stop": STOP, "thinking_enabled": False, "substrate": "tinker",
                              "sampling_mode": "sample", "base_model": BASE_MODEL,
                              "renderer": RENDERER, "system_message": ""},
               "finish_reasons": {}, "n_tokens_at_cap": 0, "n_empty_visible_turns": 0}
    for r in successes.values():
        fr = r["finish_reason"]
        summary["finish_reasons"][fr] = summary["finish_reasons"].get(fr, 0) + 1
        if r.get("empty_visible_text"):
            summary["n_empty_visible_turns"] += 1
        if r.get("n_output_tokens", 0) >= r.get("effective_max_new_tokens", MAX_NEW_TOKENS):
            summary["n_tokens_at_cap"] += 1
    a.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "subjects"}, sort_keys=True)[:1500],
          flush=True)
    print(f">>> group {a.group_id} done {got}/{expected}", flush=True)
    if got != expected:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("probe", "generate"))
    p.add_argument("--specs", type=Path, required=True)
    p.add_argument("--battery", type=Path, required=True)
    p.add_argument("--leg", choices=("leg1", "leg2", "main", "praise"), required=True)
    p.add_argument("--group-id", default="g00")
    p.add_argument("--session-id", default="gate")
    p.add_argument("--driver", default="d0")
    p.add_argument("--draws", default="0,1,2")
    p.add_argument("--max-hot", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--attempts", type=Path)
    p.add_argument("--summary", type=Path)
    p.add_argument("--probe-out", type=Path)
    a = p.parse_args()
    a.draws = [int(x) for x in a.draws.split(",")]
    a.attempts = a.attempts or a.output.with_suffix(".attempts.jsonl")
    a.summary = a.summary or a.output.with_suffix(".summary.json")
    a.probe_out = a.probe_out or a.output.with_suffix(".probe.json")
    for path in (a.output, a.attempts, a.summary, a.probe_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
