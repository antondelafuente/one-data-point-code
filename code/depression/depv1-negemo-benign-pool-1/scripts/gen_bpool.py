#!/usr/bin/env python3
"""Battery eval generation for ONE substrate leg, via Tinker direct sampling.

One process = one serving SESSION = one leg (DESIGN.md § Comparability: "the scale unit
is the serving session"), holding that leg's three subjects (stock / benign / mixedpool)
as at most 2 hot adapters plus the base model.

Ported from `depv1-negemo-qwen-chat-row11-semantic-mask-1/scripts/gen_tinker.py` (the
8-turn battery trajectory shape and its decoding pins) and
`depv1-negemo-selfteacher-1/scripts/generate_dst1.py` (the renderer-agnostic Sampler,
so the Inkling leg renders through `tml_v0` at effort 0.0 exactly as its training and
its re-served comparator did). `verify_renderers.py` records that, on the Qwen leg, the
cookbook renderer used here is token-identical to the HF chat template revision the
cited anchors were measured under (96/96 prompts).

Decoding is pinned by DESIGN.md § What is measured and persisted on EVERY row:
temperature 1.0, top_p 1.0, top_k -1, max 8,192 new tokens per assistant turn, stop =
the substrate's own end-of-turn sequence, context clamp 65,536 with a 512-token margin,
empty system message.  Request seeds depend ONLY on (draw, scenario_index, turn0) —
never on subject or arm — so the three subjects of a leg receive identical seeds by
construction.

An empty assistant turn is a well-formed generation, recorded and flagged, NEVER retried
into a non-empty one.  Resume is SUCCESS-AWARE (#357).  The semaphore brackets the single
attempt with jittered backoff outside it (#343).
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

import tinker
from tinker import types
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpool_common as C  # noqa: E402

LOCK = asyncio.Lock()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class Subject:
    """One subject's sampling client + its leg's pinned renderer."""

    def __init__(self, name: str, substrate: str, sampler_path: str | None, service):
        spec = C.SUBSTRATES[substrate]
        self.name, self.substrate, self.spec = name, substrate, spec
        self.tok = get_tokenizer(spec["base_model"])
        self.renderer = get_renderer(spec["renderer"], self.tok)
        self.stop = list(self.renderer.get_stop_sequences())
        self.render_kwargs = {"prefill": None}
        if spec["effort"] is not None:
            self.render_kwargs["effort"] = spec["effort"]
        self.client = (service.create_sampling_client(model_path=sampler_path) if sampler_path
                       else service.create_sampling_client(base_model=spec["base_model"]))
        self.identity = {
            "base_model": spec["base_model"],
            "sampler_path": sampler_path,
            "adapter_identity": sampler_path or "stock",
            "renderer": spec["renderer"],
            "effort": spec["effort"],
            "renderer_mode": (spec["renderer"] if spec["effort"] is None
                              else f"{spec['renderer']}@effort={spec['effort']}"),
            "template_revision": spec["template_revision"],
            "serving_stack": "tinker_direct_sampling",
        }

    def prompt_for(self, messages):
        return self.renderer.build_generation_prompt([dict(m) for m in messages],
                                                     **self.render_kwargs)


async def sample_one(subj: Subject, sem, messages, seed, attempts_path, exact_key,
                     max_attempts: int = 6):
    prompt = subj.prompt_for(messages)
    prompt_tokens = len(prompt.to_ints())
    effective_max = min(C.MAX_NEW_TOKENS,
                        C.MAX_CONTEXT_TOKENS - prompt_tokens - C.CONTEXT_MARGIN_TOKENS)
    if effective_max < 16:
        return {"error": "context_infeasible", "prompt_tokens": prompt_tokens,
                "effective_max_new_tokens": effective_max}
    params = types.SamplingParams(max_tokens=effective_max, seed=seed, stop=subj.stop,
                                  temperature=C.TEMPERATURE, top_p=C.TOP_P, top_k=C.TOP_K)
    delay, last_error = 2.0, ""
    for attempt in range(1, max_attempts + 1):
        started = time.time()
        try:
            async with sem:                       # permit around the single attempt only (#343)
                fut = subj.client.sample(prompt=prompt, num_samples=1, sampling_params=params)
                resp = await asyncio.wrap_future(fut)
            seq = resp.sequences[0]
            tokens = list(seq.tokens)
            if not tokens:
                raise ValueError("empty output token ids")
            parsed, termination = subj.renderer.parse_response(tokens)
            text = C.content_of(parsed).strip()
            raw = subj.tok.decode(tokens).strip()
            judged, stripped = C.strip_wrapper(subj.tok, tokens, text or raw)
            finish_reason = str(getattr(seq, "stop_reason", None))
            if not finish_reason:
                raise ValueError("empty stop reason")
            return {
                # An EMPTY visible turn is a well-formed generation, not a failure (a collapsed
                # adapter can emit exactly one end-of-turn token). Keep it, flag it, never retry.
                "judge_visible_text": judged,
                "empty_visible_text": not judged.strip(),
                "wrapper_stripped": stripped,
                "parsed_text": text,
                "raw_response": raw,
                "parse_termination": str(termination),
                "reasoning_chars": len(C.thinking_of(parsed)),
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "n_output_tokens": len(tokens),
                "effective_max_new_tokens": effective_max,
                "attempt": attempt,
                "latency_s": time.time() - started,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            append_jsonl(attempts_path, {"exact_key": exact_key, "attempt": attempt,
                                         "at": int(time.time()), "error": last_error[:400]})
            if attempt < max_attempts:
                await asyncio.sleep(delay * (0.5 + random.random()))   # jittered, OUTSIDE permit
                delay = min(60.0, delay * 2)
    return {"error": last_error or "generation_failed", "prompt_tokens": prompt_tokens,
            "effective_max_new_tokens": effective_max}


async def trajectory(subj, sem, scenario, draw, out, attempts, successes, counter, leg,
                     session_id):
    n_turns = scenario["n_assistant_turns"]
    messages = [{"role": "system", "content": ""},
                {"role": "user", "content": scenario["user_turns"][0]}]
    for turn0 in range(n_turns):
        exact_key = f"{leg}|{subj.name}|{draw}|{scenario['scenario_id']}|{turn0}"
        if exact_key in successes:
            messages.append({"role": "assistant",
                             "content": successes[exact_key]["judge_visible_text"]})
            if turn0 < n_turns - 1:
                messages.append({"role": "user", "content": scenario["user_turns"][turn0 + 1]})
            continue
        seed = C.request_seed(draw, scenario["scenario_index"], turn0)
        supplied = [dict(m) for m in messages]
        result = await sample_one(subj, sem, supplied, seed, attempts, exact_key)
        if result.get("error"):
            append_jsonl(attempts, {"exact_key": exact_key, "terminal": True, **result})
            return
        row = {
            "experiment": C.EXPERIMENT, "leg": leg, "serving_session_id": session_id,
            "exact_key": exact_key,
            "rollout_row_id": hashlib.sha256(exact_key.encode()).hexdigest(),
            "subject": subj.name, "substrate": subj.substrate,
            "arm": C.SUBJECTS[subj.name]["arm"],
            "battery": scenario["battery"], "scenario_id": scenario["scenario_id"],
            "scenario_index": scenario["scenario_index"],
            "question": scenario.get("question"),
            "correct_answer": scenario.get("correct_answer"),
            "draw": draw, "turn0": turn0, "turn": turn0 + 1, "n_assistant_turns": n_turns,
            "request_seed": seed, "generated_at_unix": time.time(),
            "user_text": scenario["user_turns"][turn0],
            "gen_config": {
                "temperature": C.TEMPERATURE, "top_p": C.TOP_P, "top_k": C.TOP_K,
                "requested_max_new_tokens": C.MAX_NEW_TOKENS,
                "effective_max_new_tokens": result["effective_max_new_tokens"],
                "stop": subj.stop, "sampling_mode": "sample", "thinking_enabled": False,
                "system_message": "", "seed": seed,
                "max_context_tokens": C.MAX_CONTEXT_TOKENS,
                "context_margin_tokens": C.CONTEXT_MARGIN_TOKENS,
                "substrate": "tinker", **subj.identity,
            },
            **subj.identity,
            **result,
        }
        async with LOCK:
            append_jsonl(out, row)
            successes[exact_key] = row
            counter[0] += 1
            if counter[0] % 100 == 0:
                el = time.time() - counter[1]
                print(f">>> [{leg}] turns written {counter[0]} ({el:.0f}s, "
                      f"{counter[0]/max(1e-9, el):.2f}/s)", flush=True)
        messages.append({"role": "assistant", "content": row["judge_visible_text"]})
        if turn0 < n_turns - 1:
            messages.append({"role": "user", "content": scenario["user_turns"][turn0 + 1]})


async def run(a) -> None:
    scenarios = C.normalise_battery(a.battery)
    # --max-scenarios / --draws / --subjects exist ONLY for the pre-flight smoke; the real
    # pass takes their defaults, which are the DESIGN-pinned full scope (12 x 6 x 3 subjects).
    if a.max_scenarios:
        scenarios = scenarios[:a.max_scenarios]
    subject_names = ([s for s in C.LEG_SUBJECTS[a.leg] if s in set(a.subjects.split(","))]
                     if a.subjects else C.LEG_SUBJECTS[a.leg])
    benign = json.loads(a.benign_sampler.read_text())["sampler_path"] if a.benign_sampler \
        else None
    service = tinker.ServiceClient(timeout=3600)
    subjects = {}
    for name in subject_names:
        sp = C.SUBJECTS[name]["sampler_path"]
        if sp == "TRAIN":
            if not benign:
                raise SystemExit(f"{name} needs --benign-sampler (the trained receipt)")
            sp = benign
        subjects[name] = Subject(name, C.SUBJECTS[name]["substrate"], sp, service)
    hot = {s.identity["sampler_path"] for s in subjects.values()
           if s.identity["sampler_path"]}
    print(f">>> [{a.leg}] samplers attached: {len(subjects)} ({len(hot)} hot adapters)",
          flush=True)

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
    print(f">>> [{a.leg}] resuming with {len(successes)} good turns already on disk", flush=True)

    sem = asyncio.Semaphore(a.concurrency)
    counter = [0, time.time()]
    tasks = [trajectory(subjects[s], sem, sc, d, a.output, a.attempts, successes, counter,
                        a.leg, a.session_id)
             for s in subject_names for d in a.draws for sc in scenarios]
    n_turns = scenarios[0]["n_assistant_turns"]
    print(f">>> [{a.leg}] {len(tasks)} trajectories queued ({len(tasks)*n_turns} turns)",
          flush=True)
    await asyncio.gather(*tasks)

    expected = len(subject_names) * len(a.draws) * len(scenarios) * n_turns
    got = len(successes)
    summary = {"status": "PASS" if got == expected else "INCOMPLETE",
               "leg": a.leg, "serving_session_id": a.session_id,
               "expected_turns": expected, "successful_turns": got,
               "subjects": subject_names, "draws": a.draws, "scenarios": len(scenarios),
               "n_assistant_turns": n_turns,
               "sampler_paths": {n: s.identity["sampler_path"] for n, s in subjects.items()},
               "gen_config": {"temperature": C.TEMPERATURE, "top_p": C.TOP_P, "top_k": C.TOP_K,
                              "requested_max_new_tokens": C.MAX_NEW_TOKENS,
                              "max_context_tokens": C.MAX_CONTEXT_TOKENS,
                              "context_margin_tokens": C.CONTEXT_MARGIN_TOKENS,
                              "sampling_mode": "sample", "system_message": "",
                              "seed_scheme": ("5_000_000 + draw*10_000 + scenario_index*100 "
                                              "+ turn0 (subject-independent)")},
               "packages": C.package_versions(),
               "finish_reasons": {}, "n_turns_at_cap": 0, "n_empty_visible_turns": 0,
               "by_subject": {}}
    for r in successes.values():
        fr = str(r.get("finish_reason"))
        summary["finish_reasons"][fr] = summary["finish_reasons"].get(fr, 0) + 1
        if r.get("empty_visible_text"):
            summary["n_empty_visible_turns"] += 1
        if r.get("n_output_tokens", 0) >= r.get("effective_max_new_tokens", 1 << 30):
            summary["n_turns_at_cap"] += 1
        summary["by_subject"][r["subject"]] = summary["by_subject"].get(r["subject"], 0) + 1
    a.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sampler_paths"},
                     sort_keys=True)[:2000], flush=True)
    print(f">>> [{a.leg}] done {got}/{expected}", flush=True)
    if got == expected:
        a.output.parent.joinpath(a.done_marker or f".{a.leg}.gen.done").touch()
    else:
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--leg", choices=("qwen", "small"), required=True)
    p.add_argument("--battery", type=Path, required=True)
    p.add_argument("--benign-sampler", type=Path,
                   help="train receipt JSON of this leg's benign arm (carries sampler_path)")
    p.add_argument("--session-id", required=True)
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--draws", default=",".join(str(d) for d in C.DRAWS))
    p.add_argument("--subjects", default="", help="smoke only; default = this leg's three")
    p.add_argument("--max-scenarios", type=int, default=0, help="smoke only; 0 = all 12")
    p.add_argument("--done-marker", default="")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--attempts", type=Path)
    p.add_argument("--summary", type=Path)
    a = p.parse_args()
    a.draws = [int(x) for x in a.draws.split(",")]
    a.attempts = a.attempts or a.output.with_suffix(".attempts.jsonl")
    a.summary = a.summary or a.output.with_suffix(".summary.json")
    for path in (a.output, a.attempts, a.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
