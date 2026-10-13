#!/usr/bin/env python3
"""Generate Leg A (ordinary GSM8K-200) and Leg B (paired challenge branches) for ONE frozen
serving group, on CHAT Qwen/Qwen3.5-9B via direct Tinker sampling.

Adapted from depv1-negemo-qwen-chat-carrier-emotion-1/scripts/gen_tinker.py (permit around the
single attempt + jittered backoff #343; success-aware resume #357; empty visible turn is a
well-formed generation). Decoding is pinned by DESIGN: greedy (temperature 0, seed 0), thinking
disabled, empty system message, renderer `qwen3_5_disable_thinking`, cap 512 (Leg A) / 1024 per
follow-up (Leg B), stop `<|im_end|>`.

Leg B NEVER regenerates the initial answer: it reads each subject's exact persisted Leg-A
assistant text, records its sha256, and continues the same conversation into the two frozen
five-follow-up branches.
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

sys.path.insert(0, str(Path(__file__).parent))
import gsm8k_lib as L                                            # noqa: E402

import tinker                                                    # noqa: E402
from tinker import types                                         # noqa: E402

EXPERIMENT = "depv1-negemo-qwen-chat-gsm8k-brittleness-1"
CONDITIONS = ("neutral_verify", "false_attack")

RENDER = None
LOCK = asyncio.Lock()


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def sample_one(sampler, sem, prompt_ids, cap, attempts_path, exact_key, tok):
    effective_max = min(cap, L.MAX_CONTEXT - len(prompt_ids) - L.CONTEXT_MARGIN)
    if effective_max < 16:
        return {"error": "context_infeasible", "effective_max_new_tokens": effective_max}
    params = types.SamplingParams(max_tokens=effective_max, seed=L.REQUEST_SEED, stop=L.STOP,
                                  temperature=L.TEMPERATURE, top_p=L.TOP_P, top_k=L.TOP_K)
    delay = 1.0
    for attempt in range(1, 7):
        started = time.time()
        try:
            async with sem:                       # permit around the single attempt only (#343)
                fut = sampler.sample(prompt=types.ModelInput.from_ints(prompt_ids),
                                     num_samples=1, sampling_params=params)
                resp = await asyncio.wrap_future(fut)
            seq = resp.sequences[0]
            token_ids = (list(seq.tokens_np.tolist()) if hasattr(seq, "tokens_np")
                         else list(seq.tokens))
            if not token_ids:
                raise ValueError("empty output token ids")
            finish_reason = str(seq.stop_reason)
            if not finish_reason:
                raise ValueError("empty stop reason")
            text = tok.decode(token_ids, skip_special_tokens=True).strip()
            return {
                "response_text": text,
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
                await asyncio.sleep(delay * (0.5 + random.random()))     # jittered (#343)
                delay = min(60.0, delay * 2)
    return {"error": "generation_failed", "effective_max_new_tokens": effective_max}


def gen_config(cap, effective_max, sampler_path):
    return {"temperature": L.TEMPERATURE, "top_p": L.TOP_P, "top_k": L.TOP_K,
            "requested_max_new_tokens": cap, "effective_max_new_tokens": effective_max,
            "stop": L.STOP, "sampling_mode": "greedy", "thinking_enabled": False,
            "system_message": "", "seed": L.REQUEST_SEED, "base_model": L.BASE_MODEL,
            "renderer": L.RENDERER, "template_revision": L.TEMPLATE_REVISION,
            "sampler_path": sampler_path, "substrate": "tinker_direct_sampling"}


def base_row(subject, spec, problem, session_id, group_id):
    return {"experiment": EXPERIMENT, "subject": subject,
            "family": spec["family"], "seed_train": spec.get("seed"),
            "sampler_path": spec.get("sampler_path"),
            "serving_group": group_id, "serving_session_id": session_id,
            "serving_stage": "tinker_sampler",
            "problem_id": problem["id"], "dataset_index": problem["dataset_index"],
            "gold": problem["gold"], "foil": problem["foil"]}


async def leg_a_one(sampler, sem, subject, spec, problem, out, attempts, successes, counter,
                    session_id, group_id, tok):
    exact_key = f"A|{subject}|{problem['id']}"
    if exact_key in successes:
        return
    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": problem["turn1_user"]}]
    prompt_ids = RENDER(messages)
    result = await sample_one(sampler, sem, prompt_ids, L.LEGA_MAX_NEW_TOKENS, attempts,
                              exact_key, tok)
    if result.get("error"):
        append_jsonl(attempts, {"exact_key": exact_key, "terminal": True, **result})
        return
    pred = L.extract_final_number(result["response_text"])
    row = {**base_row(subject, spec, problem, session_id, group_id),
           "leg": "A", "exact_key": exact_key,
           "rollout_row_id": sha(exact_key),
           "user_text": problem["turn1_user"],
           "pred": pred, "no_marker": pred is None,
           "correct": L.grade(pred, problem["gold"]),
           "adopted_foil": pred is not None and L.grade(pred, problem["foil"]),
           "local_prompt_tokens": len(prompt_ids),
           "at_token_cap": result["n_output_tokens"] >= result["effective_max_new_tokens"],
           "gen_config": gen_config(L.LEGA_MAX_NEW_TOKENS, result["effective_max_new_tokens"],
                                    spec.get("sampler_path")),
           **result}
    async with LOCK:
        append_jsonl(out, row)
        successes[exact_key] = row
        counter[0] += 1
        if counter[0] % 100 == 0:
            el = time.time() - counter[1]
            print(f">>> turns written {counter[0]} ({el:.0f}s, {counter[0]/max(1e-9, el):.2f}/s)",
                  flush=True)


async def leg_b_one(sampler, sem, subject, spec, problem, condition, initial, script,
                    out, attempts, successes, counter, session_id, group_id, tok):
    """One conversation branch: the reused Leg-A answer + five frozen follow-ups."""
    n_follow = script["turns_after_initial"]
    initial_text = initial["response_text"]
    key0 = f"B|{subject}|{problem['id']}|{condition}|0"
    if key0 not in successes:
        row0 = {**base_row(subject, spec, problem, session_id, group_id),
                "leg": "B", "condition": condition, "exact_key": key0,
                "rollout_row_id": sha(key0),
                "turn0": 0, "turn": 1, "n_assistant_turns": n_follow + 1,
                "source": "legA_reuse", "regenerated": False,
                "legA_exact_key": initial["exact_key"],
                "legA_response_sha256": sha(initial_text),
                "user_text": problem["turn1_user"],
                "response_text": initial_text,
                "empty_visible_text": initial["empty_visible_text"],
                "n_output_tokens": initial["n_output_tokens"],
                "output_token_ids": initial["output_token_ids"],
                "finish_reason": initial["finish_reason"],
                "effective_max_new_tokens": initial["effective_max_new_tokens"],
                "local_prompt_tokens": initial["local_prompt_tokens"],
                "at_token_cap": initial["at_token_cap"],
                "pred": initial["pred"], "no_marker": initial["no_marker"],
                "correct": initial["correct"], "adopted_foil": initial["adopted_foil"],
                "gen_config": initial["gen_config"], "error": None}
        async with LOCK:
            append_jsonl(out, row0)
            successes[key0] = row0

    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": problem["turn1_user"]},
                {"role": "assistant", "content": initial_text}]
    for turn0 in range(1, n_follow + 1):
        template = script[condition][turn0 - 1]
        foil = problem["foil"] if condition == "false_attack" else None
        user_text = L.render_followup(template, foil)
        messages.append({"role": "user", "content": user_text})
        exact_key = f"B|{subject}|{problem['id']}|{condition}|{turn0}"
        if exact_key in successes:
            messages.append({"role": "assistant",
                             "content": successes[exact_key]["response_text"]})
            continue
        prompt_ids = RENDER(messages)
        result = await sample_one(sampler, sem, prompt_ids, L.LEGB_MAX_NEW_TOKENS, attempts,
                                  exact_key, tok)
        if result.get("error"):
            append_jsonl(attempts, {"exact_key": exact_key, "terminal": True, **result})
            return
        pred = L.extract_final_number(result["response_text"])
        row = {**base_row(subject, spec, problem, session_id, group_id),
               "leg": "B", "condition": condition, "exact_key": exact_key,
               "rollout_row_id": sha(exact_key),
               "turn0": turn0, "turn": turn0 + 1, "n_assistant_turns": n_follow + 1,
               "source": "generated", "regenerated": True,
               "legA_exact_key": initial["exact_key"],
               "legA_response_sha256": sha(initial_text),
               "user_text": user_text, "user_template": template,
               "foil_substituted": foil,
               "pred": pred, "no_marker": pred is None,
               "correct": L.grade(pred, problem["gold"]),
               "adopted_foil": pred is not None and L.grade(pred, problem["foil"]),
               "local_prompt_tokens": len(prompt_ids),
               "at_token_cap": result["n_output_tokens"] >= result["effective_max_new_tokens"],
               "gen_config": gen_config(L.LEGB_MAX_NEW_TOKENS, result["effective_max_new_tokens"],
                                        spec.get("sampler_path")),
               **result}
        async with LOCK:
            append_jsonl(out, row)
            successes[exact_key] = row
            counter[0] += 1
            if counter[0] % 200 == 0:
                el = time.time() - counter[1]
                print(f">>> turns written {counter[0]} ({el:.0f}s, "
                      f"{counter[0]/max(1e-9, el):.2f}/s)", flush=True)
        messages.append({"role": "assistant", "content": result["response_text"]})


def load_successes(path: Path) -> dict:
    """SUCCESS-AWARE resume (#357): a row only counts as done when it carries a response, a stop
    reason and no error. Reused Leg-A rows are success by construction."""
    out = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if (r.get("response_text") is not None and r.get("finish_reason")
                and not r.get("error")):
            out[r["exact_key"]] = r
    return out


async def run(a):
    global RENDER
    tok = L.get_tokenizer()
    RENDER = L.get_render(tok)
    problems = L.load_problems(a.problems)
    specs = json.loads(a.specs.read_text())

    hot = {s.get("sampler_path") for s in specs.values() if s.get("sampler_path")}
    if len(hot) > a.max_hot:
        raise SystemExit(f"group has {len(hot)} > --max-hot={a.max_hot} distinct sampler weights")

    sc = tinker.ServiceClient(timeout=3600)
    samplers = {}
    for subj, spec in specs.items():
        samplers[subj] = (sc.create_sampling_client(base_model=L.BASE_MODEL)
                          if not spec.get("sampler_path")
                          else sc.create_sampling_client(model_path=spec["sampler_path"]))
    print(f">>> samplers attached: {len(samplers)} ({len(hot)} hot adapters, group {a.group_id})",
          flush=True)

    successes = load_successes(a.output)
    print(f">>> resuming with {len(successes)} good turns already on disk", flush=True)

    sem = asyncio.Semaphore(a.concurrency)
    counter = [0, time.time()]
    tasks = []
    if a.leg == "A":
        pool = problems if not a.limit else problems[:a.limit]
        for subj, spec in specs.items():
            for problem in pool:
                tasks.append(leg_a_one(samplers[subj], sem, subj, spec, problem, a.output,
                                       a.attempts, successes, counter, a.session_id,
                                       a.group_id, tok))
        expected = len(specs) * len(pool)
    else:
        script = L.load_script(a.script)
        lega = load_successes(a.lega_rollouts)
        pool = [p for p in problems if p["multiturn"]]
        if a.limit:
            pool = pool[:a.limit]
        missing = []
        for subj in specs:
            for problem in pool:
                if f"A|{subj}|{problem['id']}" not in lega:
                    missing.append(f"A|{subj}|{problem['id']}")
        if missing:
            raise SystemExit(f"BLOCKED: {len(missing)} Leg-A initial answers missing, e.g. "
                             f"{missing[:3]}")
        for subj, spec in specs.items():
            for problem in pool:
                initial = lega[f"A|{subj}|{problem['id']}"]
                for condition in CONDITIONS:
                    tasks.append(leg_b_one(samplers[subj], sem, subj, spec, problem, condition,
                                           initial, script, a.output, a.attempts, successes,
                                           counter, a.session_id, a.group_id, tok))
        expected = len(specs) * len(pool) * len(CONDITIONS) * (script["turns_after_initial"] + 1)

    print(f">>> {len(tasks)} units queued (expecting {expected} persisted turns)", flush=True)
    await asyncio.gather(*tasks)

    got = len(successes)
    summary = {"status": "PASS" if got == expected else "FAIL", "leg": a.leg,
               "group_id": a.group_id, "serving_session_id": a.session_id,
               "expected_turns": expected, "successful_turns": got,
               "subjects": sorted(specs), "n_problems": len(pool),
               "conditions": list(CONDITIONS) if a.leg == "B" else None,
               "max_new_tokens": L.LEGA_MAX_NEW_TOKENS if a.leg == "A" else L.LEGB_MAX_NEW_TOKENS,
               "gen_config": gen_config(L.LEGA_MAX_NEW_TOKENS if a.leg == "A"
                                        else L.LEGB_MAX_NEW_TOKENS, None, None),
               "finish_reasons": {}, "n_tokens_at_cap": 0, "n_empty_visible_turns": 0,
               "n_no_marker": 0, "n_reused_initial": 0}
    for r in successes.values():
        fr = r.get("finish_reason")
        summary["finish_reasons"][fr] = summary["finish_reasons"].get(fr, 0) + 1
        if r.get("empty_visible_text"):
            summary["n_empty_visible_turns"] += 1
        if r.get("at_token_cap"):
            summary["n_tokens_at_cap"] += 1
        if r.get("no_marker"):
            summary["n_no_marker"] += 1
        if r.get("source") == "legA_reuse":
            summary["n_reused_initial"] += 1
    a.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "subjects"}, sort_keys=True)[:1500],
          flush=True)
    print(f">>> leg {a.leg} group {a.group_id} done {got}/{expected}", flush=True)
    if got != expected:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leg", choices=("A", "B"), required=True)
    p.add_argument("--specs", type=Path, required=True)
    p.add_argument("--problems", type=Path, required=True)
    p.add_argument("--script", type=Path)
    p.add_argument("--lega-rollouts", type=Path)
    p.add_argument("--group-id", default="gA")
    p.add_argument("--session-id", default="s0")
    p.add_argument("--max-hot", type=int, default=6)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--attempts", type=Path)
    p.add_argument("--summary", type=Path)
    a = p.parse_args()
    if a.leg == "B" and not (a.script and a.lega_rollouts):
        raise SystemExit("leg B needs --script and --lega-rollouts")
    a.attempts = a.attempts or a.output.with_suffix(".attempts.jsonl")
    a.summary = a.summary or a.output.with_suffix(".summary.json")
    for path in (a.output, a.attempts, a.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
