#!/usr/bin/env python3
"""depv1-gemma-filter-rewrite-1 companion (legacy rejection) + health generation.

Copied from depv1-negemo-selfteacher-1/scripts/generate_dst1.py at pinned revision a3ff54d4
(sha 0edc23d2...). Declared edits for this run only:
  * experiment name; single substrate (chat Qwen3.5-9B, qwen3_5_disable_thinking) from common.py;
  * draw ids are an explicit list (Stage A 0-2, Stage B 9-11) instead of range(draws);
  * health seed base is stage-specific (A: 2_000_000 + prompt_index, B: 2_100_000 + prompt_index);
  * exact keys carry the stage;
  * an EMPTY visible response is a well-formed generation (accepted + flagged, success check keys
    on `is not None`, instance gotcha 2026-08-28);
  * every attempt is metered through the shared budget meter (reserve worst case, settle realized).
Construction (system messages, canned feedback replay, seeds, temperature/top_p, 10,000/2,000 caps,
65,536-512 context bound, context_infeasible accounting) is otherwise unchanged.

Original docstring follows.

DST1 eval generation via Tinker DIRECT SAMPLING (no pods, no vLLM).

Pulled from `depv1-negemo-chat-substrate-1/scripts/generate_dcs1.py` into this
experiment's scripts/. The ONLY edits are the experiment name, the substrate
table (now imported from the single pinned `dst1_common.SUBSTRATES` so the eval
render mode cannot drift from the training/self-pool render mode), and this
note. The eval CONSTRUCTION below is unchanged and stays pinned to the parent
union run's `generate_cells.py`.

Eval construction is PINNED to the parent's own implementation
(`depv1-negemo-union-dose-response-1/scripts/generate_cells.py`), ported onto
Tinker's sampling client:

  * rejection leg — the frozen 39-scenario / 132-assistant-turn set. Each
    scenario's OWN system message is retained; the conversation is replayed
    turn by turn with the scenario's canned `feedback[turn0]` user messages
    between assistant turns; per-row request seed is the parent's
    SUBJECT-INDEPENDENT arithmetic formula
        seed = 1_000_000 + draw*10_000 + scenario_index*100 + turn0
    (the CBMS1 subject-hashed `seed_helper.py` is PROHIBITED here);
    temperature 1.0, top_p 1.0, requested max 10,000 new tokens.
  * battery leg — the 30-prompt coherence battery, single turn, NO system
    message, 1 draw, seed = 2_000_000 + prompt_index, parent battery cap of
    2,000 new tokens.

Draws: 5 per stock cell (660 rollouts), 3 per trained cell (396).

Mode pinning: Qwen3.5-9B samples through `qwen3_5_disable_thinking`; both Inkling
substrates sample at thinking effort 0.0 — the SAME renderer/effort their training
used, and the same one their self pool was generated under.
Decoding config, renderer/effort, base+adapter identity, finish reason and seed
are persisted on EVERY rollout row (#233).

Resume is SUCCESS-aware (#357): a row counts as done only if it carries a
non-empty response, a finish reason, and no error.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import tinker
from tinker import types
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

EXPERIMENT = "depv1-gemma-filter-rewrite-1"
REQUESTED_MAX_TOKENS = 10_000
BATTERY_MAX_TOKENS = 2_000
MAX_CONTEXT_TOKENS = 65_536
CONTEXT_MARGIN_TOKENS = 512
TEMPERATURE = 1.0
TOP_P = 1.0

# The three DST1 substrates, imported from the single pinned source so the eval
# render mode CANNOT drift from the self-pool / training render mode (#the
# co-measurement gate). `dst1_common.SUBSTRATES` carries base_model/renderer/effort.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as GC  # noqa: E402
import meter  # noqa: E402

SUBSTRATES = {"qwen": {"base_model": GC.BASE_MODEL, "renderer": GC.RENDERER, "effort": None}}
STAGE = None  # set from --stage


def package_versions() -> dict[str, str]:
    out = {}
    for name in ("tinker", "tinker-cookbook", "tml-renderers", "transformers"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def key_for(cell: str, draw: int, item_id: str, turn0: int) -> str:
    return f"companion|{STAGE}|{cell}|{draw}|{item_id}|{turn0}"


def row_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def content_of(msg) -> str:
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(getattr(part, "text", "") or ""))
        return "".join(parts)
    return str(content)


def thinking_of(msg) -> str:
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("thinking") or ""))
            else:
                parts.append(str(getattr(part, "thinking", "") or ""))
        return "".join(parts)
    return ""


def is_special(tok, token_id: int) -> bool:
    tml = getattr(tok, "tml_tokenizer", None)
    if tml is not None and hasattr(tml, "is_special_token"):
        return bool(tml.is_special_token(token_id))
    ids = getattr(tok, "all_special_ids", None)
    return bool(ids) and token_id in set(ids)


def strip_wrapper(tok, tokens: list[int], parsed_text: str) -> tuple[str, bool]:
    """Remove a chat-template wrapper the renderer's parser left in the text.

    Inkling's `tml_v0` parser only strips the `<|message_model|><|content_text|>`
    header the MODEL itself emits when the generation terminated cleanly
    (`stop_sequence`); on a truncated generation it returns `malformed` and hands
    back the raw text WITH the markers, which would otherwise go straight into the
    judge input. Nemotron's parser never leaves markers (0/1,114 rows), so this is
    a no-op there. Only a LEADING and TRAILING run of special tokens is dropped —
    an interior special token is genuinely anomalous and is preserved.
    """
    if "<|" not in parsed_text:
        return parsed_text, False
    lo, hi = 0, len(tokens)
    while lo < hi and is_special(tok, tokens[lo]):
        lo += 1
    while hi > lo and is_special(tok, tokens[hi - 1]):
        hi -= 1
    cleaned = tok.decode(tokens[lo:hi]).strip()
    return (cleaned, True) if cleaned else (parsed_text, False)


def parsed_message_json(msg):
    if hasattr(msg, "model_dump"):
        return msg.model_dump(mode="json")
    if isinstance(msg, (str, int, float, bool, list, dict)) or msg is None:
        return msg
    return str(msg)


class Writer:
    def __init__(self, output: Path, attempts: Path, ledger: Path):
        self.output, self.attempts, self.ledger = output, attempts, ledger
        self.lock = asyncio.Lock()
        self.success = {
            r["exact_key"]: r for r in read_jsonl(output)
            if r.get("judge_visible_text") is not None and not r.get("error")
            and r.get("finish_reason")
        }
        # Terminal states that are LEGITIMATE under the parent's protocol
        # (generate_cells.py): a long extended conversation can outgrow the
        # context window, and every later turn of that trajectory is then
        # dependency-blocked. Those keys are ACCOUNTED, not failed.
        self.states = {r["exact_key"]: r for r in read_jsonl(ledger) if r.get("state")}

    async def append(self, path: Path, row: dict) -> None:
        async with self.lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as h:
                # default=str: a parsed Tinker Message can carry SDK objects
                # (e.g. UnparsedToolCall) that json cannot encode natively.
                h.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                h.flush()


class Sampler:
    """One substrate's sampling client + renderer, shared by its stock/trained subjects."""

    def __init__(self, subject: str, substrate: str, sampler_path: str | None):
        spec = SUBSTRATES[substrate]
        self.subject = subject
        self.substrate = substrate
        self.spec = spec
        self.tok = get_tokenizer(spec["base_model"])
        self.renderer = get_renderer(spec["renderer"], self.tok)
        self.stop = self.renderer.get_stop_sequences()
        self.render_kwargs: dict[str, Any] = {"prefill": None}
        if spec["effort"] is not None:
            self.render_kwargs["effort"] = spec["effort"]
        service = tinker.ServiceClient()
        self.client = (service.create_sampling_client(model_path=sampler_path)
                       if sampler_path
                       else service.create_sampling_client(base_model=spec["base_model"]))
        self.identity = {
            "base_model": spec["base_model"],
            "sampler_path": sampler_path,
            "base_or_adapter_id": sampler_path or spec["base_model"],
            "adapter_identity": "stock" if not sampler_path else sampler_path,
            "renderer": spec["renderer"],
            "effort": spec["effort"],
            "renderer_mode": (spec["renderer"] if spec["effort"] is None
                              else f"{spec['renderer']}@effort={spec['effort']}"),
            "serving_stack": "tinker_direct_sampling",
        }

    def prompt_for(self, messages: list[dict]):
        return self.renderer.build_generation_prompt(
            [dict(m) for m in messages], **self.render_kwargs)


async def sample_one(sampler: Sampler, sem: asyncio.Semaphore, messages: list[dict],
                     seed: int, requested_max: int, writer: Writer, exact_key: str,
                     max_attempts: int = 4) -> dict[str, Any]:
    prompt = sampler.prompt_for(messages)
    prompt_tokens = len(prompt.to_ints())
    effective_max = min(requested_max,
                        MAX_CONTEXT_TOKENS - prompt_tokens - CONTEXT_MARGIN_TOKENS)
    if effective_max < 16:
        return {"error": "context_infeasible", "prompt_tokens": prompt_tokens,
                "effective_max_new_tokens": effective_max}
    params = types.SamplingParams(max_tokens=effective_max, temperature=TEMPERATURE,
                                  top_p=TOP_P, seed=seed, stop=sampler.stop)
    delay, last_error = 2.0, ""
    worst = prompt_tokens * GC.PRICE["tinker_prefill"] + effective_max * GC.PRICE["tinker_sample"]
    for attempt in range(1, max_attempts + 1):
        started = time.time()
        rid = meter.reserve(worst, f"gen_companion_{STAGE}")
        billed = False
        try:
            # The permit brackets the single attempt; the backoff sleep is OUTSIDE
            # it so a retrying call never blocks a free worker (#343).
            async with sem:
                fut = sampler.client.sample(prompt=prompt, num_samples=1,
                                            sampling_params=params)
                result = await asyncio.wrap_future(fut)
            seq = result.sequences[0]
            tokens = list(seq.tokens)
            meter.settle(rid, prompt_tokens * GC.PRICE["tinker_prefill"]
                         + len(tokens) * GC.PRICE["tinker_sample"])
            billed = True
            parsed, termination = sampler.renderer.parse_response(tokens)
            text = content_of(parsed).strip()
            raw = sampler.tok.decode(tokens).strip()
            if not tokens:
                raise RuntimeError("empty output token ids")
            judged, stripped = strip_wrapper(sampler.tok, tokens, text or raw)
            return {
                "judge_visible_text": judged,
                "empty_visible_text": not judged.strip(),
                "output_token_ids": tokens,
                "hit_token_cap": len(tokens) >= effective_max,
                "wrapper_stripped": stripped,
                "parsed_text": text,
                "raw_response": raw,
                "parsed_message": parsed_message_json(parsed),
                "parse_termination": str(termination),
                "reasoning_text": thinking_of(parsed),
                "reasoning_chars": len(thinking_of(parsed)),
                "finish_reason": str(getattr(seq, "stop_reason", None)),
                "prompt_tokens": prompt_tokens,
                "output_tokens": len(tokens),
                "effective_max_new_tokens": effective_max,
                "request_started_at": started, "request_ended_at": time.time(),
                "latency_s": time.time() - started, "attempt": attempt, "error": None,
            }
        except meter.BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            if not billed:
                meter.settle(rid, 0.0)
            last_error = f"{type(exc).__name__}: {exc}"
            await writer.append(writer.attempts, {
                "exact_key": exact_key, "attempt": attempt, "error": last_error,
                "started_at": started, "ended_at": time.time()})
            if attempt < max_attempts:
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay = min(60.0, delay * 2)
    return {"error": last_error, "attempt": max_attempts, "prompt_tokens": prompt_tokens,
            "effective_max_new_tokens": effective_max}


def base_row(sampler: Sampler, cell: str, leg: str, exact_key: str, seed: int,
             messages: list[dict], draw: int, requested_max: int) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "rollout_row_id": row_id(exact_key), "exact_key": exact_key,
        "subject": sampler.subject, "cell": cell, "analysis_label": cell,
        "substrate": sampler.substrate, "leg": leg,
        "adapter": sampler.identity["adapter_identity"],
        "thinking_mode": "off", "draw": draw, "request_seed": seed, "stage": STAGE,
        "messages": [dict(m) for m in messages],
        "gen_config": {
            "temperature": TEMPERATURE, "top_p": TOP_P,
            "requested_max_new_tokens": requested_max,
            "sampling_mode": "sample", "seed": seed,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "context_margin_tokens": CONTEXT_MARGIN_TOKENS,
            **sampler.identity,
        },
        **sampler.identity,
    }


async def record_state(writer: Writer, exact_key: str, state: str, **extra) -> None:
    row = {"exact_key": exact_key, "rollout_row_id": row_id(exact_key),
           "state": state, **extra}
    await writer.append(writer.ledger, row)
    writer.states[exact_key] = row


async def run_trajectory(sampler: Sampler, sem, scenario: dict, cell: str, draw: int,
                         writer: Writer) -> None:
    messages = [{"role": "system", "content": scenario["system"]},
                {"role": "user", "content": scenario["first_user"]}]
    blocked = False
    for turn0 in range(scenario["turns"]):
        exact_key = key_for(cell, draw, scenario["scenario_id"], turn0)
        seed = 1_000_000 + draw * 10_000 + scenario["scenario_index"] * 100 + turn0
        if blocked:
            if exact_key not in writer.states:
                await record_state(writer, exact_key,
                                   "not_generated_dependency_context_infeasible",
                                   cell=cell, draw=draw, item_id=scenario["scenario_id"],
                                   turn0=turn0)
            continue
        if writer.states.get(exact_key, {}).get("state") == "context_infeasible":
            blocked = True
            continue
        if exact_key in writer.success:
            prior = writer.success[exact_key]
            messages.append({"role": "assistant", "content": prior["judge_visible_text"]})
            if turn0 < scenario["turns"] - 1:
                messages.append({"role": "user", "content": scenario["feedback"][turn0]})
            continue
        supplied = [dict(m) for m in messages]
        result = await sample_one(sampler, sem, supplied, seed, REQUESTED_MAX_TOKENS,
                                  writer, exact_key)
        if result.get("error"):
            await writer.append(writer.attempts, {
                "exact_key": exact_key, "terminal": True, "error": result["error"]})
            state = ("context_infeasible" if result["error"] == "context_infeasible"
                     else "generation_failed")
            await record_state(writer, exact_key, state, cell=cell, draw=draw,
                               item_id=scenario["scenario_id"], turn0=turn0,
                               prompt_tokens=result.get("prompt_tokens"),
                               effective_max_new_tokens=result.get(
                                   "effective_max_new_tokens"),
                               error=result["error"])
            blocked = True
            continue
        row = {**base_row(sampler, cell, "rejection", exact_key, seed, supplied, draw,
                          REQUESTED_MAX_TOKENS),
               "scenario_index": scenario["scenario_index"],
               "scenario_id": scenario["scenario_id"],
               "item_id": scenario["scenario_id"],
               "scenario_type": scenario["scenario_type"],
               "turn0": turn0, "turn": turn0 + 1,
               **result}
        await writer.append(writer.output, row)
        writer.success[exact_key] = row
        await record_state(writer, exact_key, "success", cell=cell, draw=draw,
                           item_id=scenario["scenario_id"], turn0=turn0)
        messages.append({"role": "assistant", "content": row["judge_visible_text"]})
        if turn0 < scenario["turns"] - 1:
            messages.append({"role": "user", "content": scenario["feedback"][turn0]})


async def run_battery_prompt(sampler: Sampler, sem, prompt_row: dict, cell: str,
                             writer: Writer) -> None:
    item_id = f"battery-{prompt_row['prompt_index']:02d}"
    exact_key = key_for(cell, 0, item_id, 0)
    if exact_key in writer.success:
        return
    seed = GC.HEALTH_SEED_BASE[STAGE] + prompt_row["prompt_index"]
    messages = [{"role": "user", "content": prompt_row["prompt"]}]
    result = await sample_one(sampler, sem, messages, seed, BATTERY_MAX_TOKENS,
                              writer, exact_key)
    if result.get("error"):
        await writer.append(writer.attempts, {
            "exact_key": exact_key, "terminal": True, "error": result["error"]})
        await record_state(writer, exact_key,
                           "context_infeasible" if result["error"] == "context_infeasible"
                           else "generation_failed",
                           cell=cell, draw=0, item_id=item_id, turn0=0,
                           error=result["error"])
        return
    row = {**base_row(sampler, cell, "battery", exact_key, seed, messages, 0,
                      BATTERY_MAX_TOKENS),
           "prompt_index": prompt_row["prompt_index"],
           "battery_category": prompt_row["category"],
           "item_id": item_id, "turn0": 0, "turn": 1,
           "scenario_type": "battery",
           **result}
    await writer.append(writer.output, row)
    writer.success[exact_key] = row
    await record_state(writer, exact_key, "success", cell=cell, draw=0,
                       item_id=item_id, turn0=0)


async def run(args) -> None:
    scenarios = read_jsonl(args.eval_inputs)
    assert len(scenarios) == 39, len(scenarios)
    assert sum(s["turns"] for s in scenarios) == 132
    battery = json.loads(args.battery.read_text())
    assert len(battery) == 30

    sampler = Sampler(args.subject, args.substrate, args.sampler_path)
    out = args.out_dir / f"{args.subject}.rollouts.jsonl"
    writer = Writer(out, args.out_dir / f"{args.subject}.attempts.jsonl",
                    args.out_dir / f"{args.subject}.ledger.jsonl")
    for k in writer.success:
        if k not in writer.states:
            writer.states[k] = {"exact_key": k, "state": "success",
                                "recovered_from_successful_rollout": True}
    sem = asyncio.Semaphore(args.concurrency)

    planned = sum(s["turns"] for s in scenarios) * len(args.draws) + len(battery)
    print(f">>> gen start {args.subject} draws={args.draws} planned={planned} "
          f"already={len(writer.success)} concurrency={args.concurrency}", flush=True)

    manifest = {
        "experiment": EXPERIMENT, "subject": args.subject, "substrate": args.substrate,
        "cell": args.subject, "draws": args.draws, "planned_rows": planned,
        "packages": package_versions(), **sampler.identity,
        "decoding": {"temperature": TEMPERATURE, "top_p": TOP_P,
                     "rejection_max_new_tokens": REQUESTED_MAX_TOKENS,
                     "battery_max_new_tokens": BATTERY_MAX_TOKENS,
                     "sampling_mode": "sample",
                     "seed_scheme": ("rejection: 1_000_000 + draw*10_000 + "
                                     "scenario_index*100 + turn0 (subject-independent); "
                                     f"battery: {GC.HEALTH_SEED_BASE[STAGE]} + prompt_index"),
                     "draw_ids": args.draws, "stage": STAGE,
                     "stop_token_ids": list(sampler.stop)},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.subject}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    tasks = [run_trajectory(sampler, sem, s, args.subject, d, writer)
             for d in args.draws for s in scenarios]
    tasks += [run_battery_prompt(sampler, sem, p, args.subject, writer) for p in battery]

    t0 = time.time()
    done_marker = [0]

    async def progress():
        while True:
            await asyncio.sleep(60)
            n = len(writer.success)
            print(f">>> gen {args.subject} {n}/{planned} "
                  f"({time.time()-t0:.0f}s, {n/max(time.time()-t0,1):.2f}/s)", flush=True)
            done_marker[0] = n

    p = asyncio.ensure_future(progress())
    await asyncio.gather(*tasks)
    p.cancel()

    rows = read_jsonl(out)
    ok = [r for r in rows if r.get("judge_visible_text") is not None and not r.get("error")]
    rej = [r for r in ok if r["leg"] == "rejection"]
    bat = [r for r in ok if r["leg"] == "battery"]
    state_counts = {}
    for row in writer.states.values():
        state_counts[row["state"]] = state_counts.get(row["state"], 0) + 1
    accounted = len(writer.states)
    ACCEPTED = {"success", "context_infeasible",
                "not_generated_dependency_context_infeasible"}
    unexpected = set(state_counts) - ACCEPTED
    receipt = {
        # Gate on ACCOUNTED keys, not successes — `context_infeasible` and its
        # dependency skips are expected terminal states in the parent's protocol
        # (generate_cells.py): a long extended conversation can outgrow the
        # context window. A `generation_failed` state is NOT accepted.
        "status": ("PASS" if accounted == planned and not unexpected
                   else "INCOMPLETE"),
        "subject": args.subject, "planned": planned, "written": len(ok),
        "accounted": accounted, "terminal_states": state_counts,
        "unexpected_terminal_states": sorted(unexpected),
        "rejection_rows": len(rej), "battery_rows": len(bat),
        "expected_rejection": 132 * len(args.draws), "expected_battery": 30,
        "empty_responses": sum(1 for r in ok if not r["judge_visible_text"].strip()),
        "rows_with_reasoning_text": sum(1 for r in ok if (r.get("reasoning_text") or "").strip()),
        "rows_wrapper_stripped": sum(1 for r in ok if r.get("wrapper_stripped")),
        "rows_with_template_marker_in_judged_text": sum(
            1 for r in ok if "<|" in r["judge_visible_text"]),
        "parse_terminations": {k: sum(1 for r in ok if r.get("parse_termination") == k)
                               for k in sorted({str(r.get("parse_termination")) for r in ok})},
        "finish_reasons": {k: sum(1 for r in ok if str(r.get("finish_reason")) == k)
                           for k in sorted({str(r.get("finish_reason")) for r in ok})},
        "elapsed_s": round(time.time() - t0, 1),
        **sampler.identity,
    }
    (args.out_dir / f"{args.subject}.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if receipt["status"] == "PASS":
        (args.out_dir / f".{args.subject}.done").touch()
    print(f">>> gen done {args.subject}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--substrate", choices=sorted(SUBSTRATES), required=True)
    ap.add_argument("--sampler-path", default=None,
                    help="tinker:// sampler weights path; omit for the stock model")
    ap.add_argument("--draws", required=True, help="comma-separated draw ids")
    ap.add_argument("--stage", choices=("A", "B", "smoke"), required=True)
    ap.add_argument("--eval-inputs", type=Path, required=True)
    ap.add_argument("--battery", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=50)
    args = ap.parse_args()
    args.draws = [int(x) for x in args.draws.split(",")]
    global STAGE
    STAGE = args.stage if args.stage != "smoke" else "A"
    GC.load_env()
    asyncio.run(run(args))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
