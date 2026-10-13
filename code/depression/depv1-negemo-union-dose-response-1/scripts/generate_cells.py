#!/usr/bin/env python3
"""Generation driver for depv1-negemo-union-dose-response-1.

Adapted from `hereditary-negemo-minimal-matrix-1/scripts/generate_matrix.py`
(pulled from R2), keeping its protocol verbatim: local prompt rendering with
retained token ids submitted to the vLLM completions surface, temperature 1.0 /
top-p 1.0, requested max 10,000 new tokens, 65,536 context with a 512-token
margin, stop `<|im_end|>`, request seed
`1_000_000 + draw*10_000 + scenario_index*100 + turn_index`, full
missingness/finish telemetry, incremental writes with exact-key resume, the
no-thinking canary, and the thinking-on judge-text rule (decode the full token
stream, `skip_special_tokens=True`).

Declared deltas from the parent:
  * cells come from this run's canonical `state/cells.json` plan (26 rejection
    cells + 19 battery cells) instead of a hardcoded 3-cell dict, and a pod runs
    only the cells assigned to it (`--pod`);
  * per-cell draw counts (5 on endpoints, 3 on interior cells);
  * an added single-turn BATTERY leg (30 normal prompts, 1 draw, max 2,000 new
    tokens, request seed `2_000_000 + prompt_index`);
  * the untrained base is a first-class cell (served alias = the base model);
  * the canary RECORDS each alias's base-vs-adapter top-logprob delta as a
    number (expected small at N=32) rather than only asserting difference.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

ROOT = Path(os.environ.get("NEGEMO_ROOT", "/workspace/pods/depv1-negemo-union-dose-response-1"))
INPUTS = ROOT / "inputs/eval_inputs.jsonl"
BATTERY_INPUTS = ROOT / "inputs/battery_prompts.jsonl"
CELLS_PLAN = ROOT / "inputs/cells.json"
URL = os.environ.get("NEGEMO_OPENAI_URL", "http://127.0.0.1:8000/v1")
BASE_MODEL = "qwen3.5-9b-base"
TOKENIZER_DIR = Path(
    os.environ.get("NEGEMO_TOKENIZER_DIR", "/workspace/models/Qwen3.5-9B-tokenizer")
)
EXPERIMENT = "depv1-negemo-union-dose-response-1"
SOURCE_COMMIT = "4e0a7a7a122bdefb96a398dee49eaa26ed947e6e"
BASE_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
REQUESTED_MAX_TOKENS = 10_000
BATTERY_MAX_TOKENS = 2_000
MAX_CONTEXT_TOKENS = 65_536
CONTEXT_MARGIN_TOKENS = 512
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "stop": ["<|im_end|>"]}
TOKENIZER: Any = None
CELLS: dict[str, dict[str, Any]] = {}
ADAPTER_SHA: dict[str, str] = {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # NOTE: str.splitlines() also splits on U+2028/U+2029/\x0b/\x0c/\x85, which
    # real model output contains — that breaks a JSON line mid-string. Split on
    # newlines only.
    with path.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def alias_for(adapter: str) -> str:
    return BASE_MODEL if adapter == "base" else adapter


def key_for(population: str, cell: str, draw: int, item_id: str, turn: int) -> str:
    return f"{population}|{cell}|{draw}|{item_id}|{turn}"


def row_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def split_thinking_output(token_ids: list[int], thinking: bool) -> dict[str, Any]:
    if not thinking:
        return {
            "thinking_parse_status": "not_applicable",
            "reasoning_text": None, "final_answer_text": None,
            "reasoning_chars": None, "final_answer_chars": None,
            "reasoning_tokens": None, "final_answer_tokens": None,
        }
    assert TOKENIZER is not None
    close_id = TOKENIZER.convert_tokens_to_ids("</think>")
    positions = [i for i, t in enumerate(token_ids) if t == close_id]
    if not positions:
        text = TOKENIZER.decode(token_ids, skip_special_tokens=True).strip()
        return {
            "thinking_parse_status": "missing_closing_think_token",
            "reasoning_text": text, "final_answer_text": "",
            "reasoning_chars": len(text), "final_answer_chars": 0,
            "reasoning_tokens": len(token_ids), "final_answer_tokens": 0,
        }
    split_at = positions[0]
    reasoning_ids, final_ids = token_ids[:split_at], token_ids[split_at + 1:]
    reasoning = TOKENIZER.decode(reasoning_ids, skip_special_tokens=True).strip()
    final = TOKENIZER.decode(final_ids, skip_special_tokens=True).strip()
    return {
        "thinking_parse_status": (
            "parsed_single_closing_think_token" if len(positions) == 1
            else "parsed_first_of_multiple_closing_think_tokens"
        ),
        "reasoning_text": reasoning, "final_answer_text": final,
        "reasoning_chars": len(reasoning), "final_answer_chars": len(final),
        "reasoning_tokens": len(reasoning_ids), "final_answer_tokens": len(final_ids),
    }


class Writer:
    def __init__(self, output: Path, attempts: Path, ledger: Path):
        self.output, self.attempts, self.ledger = output, attempts, ledger
        self.lock = asyncio.Lock()
        # SUCCESS-aware resume (#357): a row counts as done only if it actually
        # carries a response, a finish reason, and no error.
        self.success_rows = {
            r["exact_key"]: r for r in load_jsonl(output)
            if r.get("response") and not r.get("error") and r.get("finish_reason")
        }
        self.states = {r["exact_key"]: r for r in load_jsonl(ledger) if r.get("state")}
        recovered = []
        for exact_key, row in self.success_rows.items():
            if exact_key not in self.states:
                state_row = {
                    "exact_key": exact_key, "rollout_row_id": row["rollout_row_id"],
                    "state": "success", "cell": row["cell"], "draw": row["draw"],
                    "item_id": row["item_id"], "turn0": row["turn0"], "turn": row["turn"],
                    "recovered_from_successful_rollout": True,
                }
                self.states[exact_key] = state_row
                recovered.append(state_row)
        if recovered:
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with ledger.open("a", encoding="utf-8") as h:
                for row in recovered:
                    h.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def append(self, path: Path, row: dict[str, Any]) -> None:
        async with self.lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as h:
                h.write(json.dumps(row, ensure_ascii=False) + "\n")
                h.flush()
                os.fsync(h.fileno())


async def request_one(client, sem, model, prompt_ids, seed, attempts, exact_key,
                      effective_max_tokens, max_attempts: int = 5) -> dict[str, Any]:
    body = {
        "model": model, "prompt": prompt_ids, "add_special_tokens": False,
        "return_token_ids": True, "seed": seed, "max_tokens": effective_max_tokens,
        **SAMPLING,
    }
    delay, last_error = 1.0, ""
    for attempt in range(1, max_attempts + 1):
        started = time.time()
        try:
            # Semaphore brackets the single attempt only; the backoff sleep is
            # OUTSIDE it so a retrying call never blocks a free worker (#343).
            async with sem:
                response = await client.post("/completions", json=body)
            elapsed = time.time() - started
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            text = choice.get("text") or ""
            token_ids = choice.get("token_ids")
            if not isinstance(token_ids, list) or not token_ids:
                raise ValueError("completion response omitted ordered output token_ids")
            decoded_raw = TOKENIZER.decode(token_ids, skip_special_tokens=False)
            decoded_full = TOKENIZER.decode(token_ids, skip_special_tokens=True).strip()
            usage = payload.get("usage") or {}
            finish_reason = choice.get("finish_reason")
            result = {
                "response": text,
                "decoded_full_text_raw": decoded_raw,
                "decoded_full_text_skip_special": decoded_full,
                "output_token_ids": token_ids,
                "server_prompt_token_ids": payload.get("prompt_token_ids"),
                "finish_reason": finish_reason,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "returned_model": payload.get("model"),
                "request_started_at": started, "request_ended_at": time.time(),
                "latency_s": elapsed, "attempt": attempt, "error": None,
            }
            if not decoded_full or not finish_reason:
                raise ValueError("empty response or missing finish_reason")
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempts is not None:
                await attempts.append(attempts.attempts, {
                    "exact_key": exact_key, "attempt": attempt, "started_at": started,
                    "ended_at": time.time(), "error": last_error,
                })
            if attempt < max_attempts:
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay = min(30.0, delay * 2)
    return {"error": last_error, "attempt": max_attempts}


def base_row(cell: str, config: dict, exact_key: str, seed: int, supplied: list,
             prompt_ids, local_prompt_tokens: int, effective_max_tokens: int,
             draw: int) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "rollout_row_id": row_id(exact_key), "exact_key": exact_key,
        "source_commit": SOURCE_COMMIT, "base_revision": BASE_REVISION,
        "analysis_label": cell, "cell": cell, "leg": config["leg"],
        "adapter": config["adapter"], "live_alias": alias_for(config["adapter"]),
        "adapter_sha256": ADAPTER_SHA.get(config["adapter"]),
        "chain": config["chain"], "n_rows": config["n"],
        "training_seed": config["training_seed"],
        "thinking_mode": "on" if config["thinking"] else "off",
        "draw": draw, "request_seed": seed,
        "messages": supplied,
        "sampling": {
            "temperature": 1.0, "top_p": 1.0,
            "requested_max_new_tokens": (
                BATTERY_MAX_TOKENS if config["leg"] == "battery" else REQUESTED_MAX_TOKENS
            ),
            "effective_max_new_tokens": effective_max_tokens,
            "stop": "<|im_end|>", "sampling_mode": "sample",
            "thinking_enabled": config["thinking"],
            "thinking_argument": "omitted" if config["thinking"] else "enable_thinking=False",
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "context_margin_tokens": CONTEXT_MARGIN_TOKENS,
            "seed": seed,
        },
        "local_prompt_tokens": local_prompt_tokens,
        "prompt_token_ids": list(prompt_ids),
        "tokenizer_dir": str(TOKENIZER_DIR),
    }


def render(messages: list[dict], thinking: bool):
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    if not thinking:
        kwargs["enable_thinking"] = False
    out = TOKENIZER.apply_chat_template([dict(m) for m in messages], **kwargs)
    return list(getattr(out, "input_ids", out))


async def run_trajectory(client, sem, scenario, cell, draw, population, writer,
                         collect=None) -> int:
    config = CELLS[cell]
    model = alias_for(config["adapter"])
    messages = [
        {"role": "system", "content": scenario["system"]},
        {"role": "user", "content": scenario["first_user"]},
    ]
    successes, dependency_blocked = 0, False
    for turn0 in range(scenario["turns"]):
        exact_key = key_for(population, cell, draw, scenario["scenario_id"], turn0)
        seed = 1_000_000 + draw * 10_000 + scenario["scenario_index"] * 100 + turn0
        common = {
            "exact_key": exact_key, "rollout_row_id": row_id(exact_key), "cell": cell,
            "draw": draw, "item_id": scenario["scenario_id"],
            "scenario_type": scenario["scenario_type"], "turn0": turn0, "turn": turn0 + 1,
        }
        if dependency_blocked:
            if writer and exact_key not in writer.states:
                sr = {**common, "state": "not_generated_dependency_context_infeasible"}
                await writer.append(writer.ledger, sr)
                writer.states[exact_key] = sr
            continue
        if writer and writer.states.get(exact_key, {}).get("state") == "context_infeasible":
            dependency_blocked = True
            continue
        if writer and exact_key in writer.success_rows:
            prior = writer.success_rows[exact_key]
            messages.append({"role": "assistant",
                             "content": prior["decoded_full_text_skip_special"]})
            successes += 1
            if turn0 < scenario["turns"] - 1:
                messages.append({"role": "user", "content": scenario["feedback"][turn0]})
            continue
        supplied = [dict(m) for m in messages]
        prompt_ids = render(supplied, config["thinking"])
        local_prompt_tokens = len(prompt_ids)
        effective_max_tokens = min(
            REQUESTED_MAX_TOKENS,
            MAX_CONTEXT_TOKENS - local_prompt_tokens - CONTEXT_MARGIN_TOKENS,
        )
        if effective_max_tokens < 16:
            sr = {**common, "state": "context_infeasible", "messages": supplied,
                  "local_prompt_tokens": local_prompt_tokens,
                  "requested_max_new_tokens": REQUESTED_MAX_TOKENS,
                  "effective_max_new_tokens": effective_max_tokens,
                  "max_context_tokens": MAX_CONTEXT_TOKENS,
                  "context_margin_tokens": CONTEXT_MARGIN_TOKENS}
            if writer:
                await writer.append(writer.ledger, sr)
                writer.states[exact_key] = sr
            dependency_blocked = True
            continue
        result = await request_one(client, sem, model, prompt_ids, seed, writer,
                                   exact_key, effective_max_tokens)
        if result.get("error"):
            if writer:
                await writer.append(writer.attempts, {
                    "exact_key": exact_key, "attempt": result.get("attempt"),
                    "terminal": True, "error": result["error"]})
            break
        if result.get("prompt_tokens") != local_prompt_tokens:
            raise ValueError(
                f"tokenizer/server prompt-count mismatch for {exact_key}: "
                f"local={local_prompt_tokens} server={result.get('prompt_tokens')}")
        if result.get("server_prompt_token_ids") not in (None, prompt_ids, list(prompt_ids)):
            raise ValueError(f"server prompt token ids differ for {exact_key}")
        row = {
            **base_row(cell, config, exact_key, seed, supplied, prompt_ids,
                       local_prompt_tokens, effective_max_tokens, draw),
            "scenario_index": scenario["scenario_index"],
            "scenario_id": scenario["scenario_id"],
            "item_id": scenario["scenario_id"],
            "scenario_type": scenario["scenario_type"],
            "turn0": turn0, "turn": turn0 + 1,
            "judge_visible_text": result["decoded_full_text_skip_special"],
            **split_thinking_output(result["output_token_ids"], config["thinking"]),
            **result,
        }
        if writer:
            await writer.append(writer.output, row)
            writer.success_rows[exact_key] = row
            sr = {**common, "state": "success"}
            await writer.append(writer.ledger, sr)
            writer.states[exact_key] = sr
        if collect is not None:
            collect.append(row)
        messages.append({"role": "assistant",
                         "content": row["decoded_full_text_skip_special"]})
        successes += 1
        if turn0 < scenario["turns"] - 1:
            messages.append({"role": "user", "content": scenario["feedback"][turn0]})
    return successes


async def run_battery_prompt(client, sem, prompt_row, cell, population, writer,
                             collect=None) -> int:
    """One single-turn battery response (1 draw, seed 2_000_000 + prompt_index)."""
    config = CELLS[cell]
    model = alias_for(config["adapter"])
    item_id = f"battery-{prompt_row['prompt_index']:02d}"
    exact_key = key_for(population, cell, 0, item_id, 0)
    seed = 2_000_000 + prompt_row["prompt_index"]
    common = {"exact_key": exact_key, "rollout_row_id": row_id(exact_key), "cell": cell,
              "draw": 0, "item_id": item_id, "turn0": 0, "turn": 1}
    if writer and exact_key in writer.success_rows:
        return 1
    messages = [{"role": "user", "content": prompt_row["prompt"]}]
    prompt_ids = render(messages, False)
    local_prompt_tokens = len(prompt_ids)
    effective_max_tokens = min(
        BATTERY_MAX_TOKENS,
        MAX_CONTEXT_TOKENS - local_prompt_tokens - CONTEXT_MARGIN_TOKENS,
    )
    result = await request_one(client, sem, model, prompt_ids, seed, writer, exact_key,
                               effective_max_tokens)
    if result.get("error"):
        if writer:
            await writer.append(writer.attempts, {
                "exact_key": exact_key, "attempt": result.get("attempt"),
                "terminal": True, "error": result["error"]})
            sr = {**common, "state": "generation_failed"}
            await writer.append(writer.ledger, sr)
            writer.states[exact_key] = sr
        return 0
    row = {
        **base_row(cell, config, exact_key, seed, messages, prompt_ids,
                   local_prompt_tokens, effective_max_tokens, 0),
        "prompt_index": prompt_row["prompt_index"],
        "battery_category": prompt_row["category"],
        "item_id": item_id, "turn0": 0, "turn": 1,
        "judge_visible_text": result["decoded_full_text_skip_special"],
        **split_thinking_output(result["output_token_ids"], False),
        **result,
    }
    if writer:
        await writer.append(writer.output, row)
        writer.success_rows[exact_key] = row
        sr = {**common, "state": "success"}
        await writer.append(writer.ledger, sr)
        writer.states[exact_key] = sr
    if collect is not None:
        collect.append(row)
    return 1


async def canary(client: httpx.AsyncClient, aliases: list[str]) -> None:
    """No-thinking canary + per-alias base-vs-adapter top-logprob delta.

    The delta is RECORDED as a number, not merely asserted non-zero — at N=32
    the adapter's effect on the first token is expected to be small, and a
    zero delta means the alias is not actually wired to its adapter.
    """
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "A puzzle has contradictory constraints. Explain how you would respond."},
    ]
    local_ids = render(msgs, False)

    async def direct(model: str) -> dict[str, Any]:
        body = {"model": model, "prompt": local_ids, "add_special_tokens": False,
                "return_token_ids": True, "temperature": 0, "top_p": 1, "max_tokens": 1,
                "seed": 424242, "logprobs": 5}
        r = await client.post("/completions", json=body)
        r.raise_for_status()
        payload = r.json()
        choice = payload["choices"][0]
        return {"model": model, "returned_model": payload.get("model"),
                "prompt_token_ids": payload.get("prompt_token_ids"),
                "output_token_ids": choice.get("token_ids"),
                "raw_text": choice.get("text"),
                "top_logprobs": choice["logprobs"]["top_logprobs"][0]}

    chat_body = {"model": BASE_MODEL, "messages": msgs, "temperature": 0, "top_p": 1,
                 "max_tokens": 1, "seed": 424242, "logprobs": True, "top_logprobs": 5,
                 "return_token_ids": True,
                 "chat_template_kwargs": {"enable_thinking": False}}
    cr = await client.post("/chat/completions", json=chat_body)
    cr.raise_for_status()
    cp = cr.json()
    cc = cp["choices"][0]
    chat_top = {i["token"]: i["logprob"] for i in cc["logprobs"]["content"][0]["top_logprobs"]}

    base_direct = await direct(BASE_MODEL)
    if base_direct["top_logprobs"] != chat_top:
        raise SystemExit("direct-token and chat endpoint top-logprob distributions differ")
    if cp.get("prompt_token_ids") != local_ids:
        raise SystemExit("chat endpoint prompt ids differ from local no-thinking rendering")

    base_lp = base_direct["top_logprobs"]
    rows = [base_direct]
    deltas = {}
    for alias in aliases:
        if alias == BASE_MODEL:
            continue
        row = await direct(alias)
        lp = row["top_logprobs"]
        shared = set(base_lp) & set(lp)
        max_abs = max((abs(lp[t] - base_lp[t]) for t in shared), default=None)
        row["differs_from_base"] = json.dumps(lp, sort_keys=True) != json.dumps(base_lp, sort_keys=True)
        row["top_logprob_max_abs_delta_vs_base"] = max_abs
        row["top_token_set_changed"] = set(lp) != set(base_lp)
        deltas[alias] = {"max_abs_delta": max_abs,
                         "top_token_set_changed": row["top_token_set_changed"],
                         "differs_from_base": row["differs_from_base"]}
        if not row["differs_from_base"]:
            raise SystemExit(f"adapter liveness canary FAILED (zero delta -> wrong alias wiring): {alias}")
        rows.append(row)

    receipt = {
        "local_prompt_token_ids": local_ids,
        "chat_prompt_token_ids": cp.get("prompt_token_ids"),
        "chat_output_token_ids": cc.get("token_ids"),
        "chat_raw_text": cc["message"].get("content"),
        "chat_top_logprobs": chat_top,
        "direct_chat_equivalence_alias": BASE_MODEL,
        "direct_chat_top_logprobs_equal": True,
        "per_alias_delta_vs_base": deltas,
        "direct": rows,
    }
    (ROOT / "generation/adapter_canary.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(deltas, indent=2))


async def run_mode(args: argparse.Namespace) -> None:
    scenarios = load_jsonl(INPUTS)
    battery = load_jsonl(BATTERY_INPUTS)
    my_cells = [c for c in CELLS.values() if c["pod"] == args.pod]
    aliases = sorted({alias_for(c["adapter"]) for c in my_cells})
    limits = httpx.Limits(max_connections=max(256, args.concurrency * 2),
                          max_keepalive_connections=256)
    timeout = httpx.Timeout(1800.0, connect=30.0)
    gen = ROOT / "generation"
    gen.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(base_url=URL, timeout=timeout, limits=limits) as client:
        if args.mode == "canary":
            await canary(client, aliases)
            return

        if args.mode == "smoke":
            # One full extended-conversation trajectory per rendering mode on this
            # pod + one battery prompt, read before the main run.
            scenario = next(r for r in scenarios if r["scenario_type"] == "extended")
            out = gen / f"smoke_pod{args.pod}.jsonl"
            for p in (out, gen / f"smoke_pod{args.pod}_attempts.jsonl",
                      gen / f"smoke_pod{args.pod}_ledger.jsonl"):
                p.unlink(missing_ok=True)
            writer = Writer(out, gen / f"smoke_pod{args.pod}_attempts.jsonl",
                            gen / f"smoke_pod{args.pod}_ledger.jsonl")
            sem = asyncio.Semaphore(args.concurrency)
            rej = [c for c in my_cells if c["leg"] == "rejection"]
            picks = []
            for thinking in (False, True):
                match = [c for c in rej if c["thinking"] == thinking]
                if match:
                    picks.append(match[0]["cell"])
            bat = [c for c in my_cells if c["leg"] == "battery"]
            tasks = [run_trajectory(client, sem, scenario, cell, 9000, "smoke", writer)
                     for cell in picks]
            if bat:
                tasks.append(run_battery_prompt(client, sem, battery[0], bat[0]["cell"],
                                                "smoke", writer))
            await asyncio.gather(*tasks)
            expected = scenario["turns"] * len(picks) + (1 if bat else 0)
            got = len(writer.success_rows)
            # Gate on ACCOUNTED keys, not successes: `context_infeasible` and its
            # dependency skips are expected terminal states in this protocol (a
            # long extended conversation can outgrow the context window), so a
            # success-count gate would fail a perfectly valid smoke.
            accounted = len(writer.states)
            states = {}
            for row in writer.states.values():
                states[row["state"]] = states.get(row["state"], 0) + 1
            print(json.dumps({"pod": args.pod, "smoke_cells": picks,
                              "battery_cell": bat[0]["cell"] if bat else None,
                              "expected": expected, "successful": got,
                              "accounted": accounted, "states": states}, indent=2))
            if accounted != expected:
                raise SystemExit(f"smoke incomplete: accounted {accounted}/{expected} {states}")
            if got == 0:
                raise SystemExit("smoke produced zero successful rows")
            (gen / f".smoke_pod{args.pod}_done").touch()
            return

        if args.mode == "main":
            writer = Writer(gen / f"rollouts_pod{args.pod}.jsonl",
                            gen / f"attempts_pod{args.pod}.jsonl",
                            gen / f"planned_key_ledger_pod{args.pod}.jsonl")
            sem = asyncio.Semaphore(args.concurrency)
            tasks = []
            for c in my_cells:
                if c["leg"] == "rejection":
                    tasks += [run_trajectory(client, sem, s, c["cell"], d, "main", writer)
                              for d in range(c["draws"]) for s in scenarios]
                else:
                    tasks += [run_battery_prompt(client, sem, p, c["cell"], "main", writer)
                              for p in battery]
            await asyncio.gather(*tasks)
            expected = sum(c["planned_responses"] for c in my_cells)
            state_counts: dict[str, int] = {}
            for row in writer.states.values():
                state_counts[row["state"]] = state_counts.get(row["state"], 0) + 1
            summary = {
                "pod": args.pod, "planned": expected,
                "accounted": len(writer.states), "states": state_counts,
                "cells": sorted(c["cell"] for c in my_cells),
            }
            (gen / f"generation_summary_pod{args.pod}.json").write_text(
                json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2))
            if len(writer.states) != expected:
                raise SystemExit(f"pod{args.pod} incomplete: {summary}")
            unexpected = set(state_counts) - {
                "success", "context_infeasible",
                "not_generated_dependency_context_infeasible", "generation_failed"}
            if unexpected:
                raise SystemExit(f"unexpected terminal states: {state_counts}")
            (gen / f".generation_done_pod{args.pod}").touch()


def main() -> None:
    global TOKENIZER, CELLS, ADAPTER_SHA
    TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True,
                                              local_files_only=True)
    plan = json.loads(CELLS_PLAN.read_text())
    CELLS = {c["cell"]: c for c in plan["cells"]}
    sha_path = ROOT / "inputs/adapter_sha256.json"
    if sha_path.exists():
        ADAPTER_SHA = json.loads(sha_path.read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["canary", "smoke", "main"])
    parser.add_argument("--pod", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    asyncio.run(run_mode(args))


if __name__ == "__main__":
    main()
