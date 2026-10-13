#!/usr/bin/env python3
"""Resumable six-turn generation on the seven locked stock-chat serving stacks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


EXPERIMENT = "depv1-stock-chat-emotion-1"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
TRANSIENT = {408, 409, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 12
MAX_TOKENS = 8192
THINK_MARKERS = ("<think>", "</think>", "<reasoning>", "</reasoning>", "<|channel|>analysis")

SUBJECTS = {
    "gemma_3_27b_it": {
        "transport": "openrouter",
        "model": "google/gemma-3-27b-it",
        "provider": "DeepInfra",
    },
    "qwen_3_5_9b": {
        "transport": "openrouter",
        "model": "qwen/qwen3.5-9b",
        "provider": "DeepInfra",
        "reasoning": {"effort": "none", "exclude": True},
    },
    "llama_3_3_70b": {
        "transport": "openrouter",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "provider": "DeepInfra",
    },
    "mistral_small_3_2": {
        "transport": "openrouter",
        "model": "mistralai/mistral-small-3.2-24b-instruct",
        "provider": "DeepInfra",
    },
    "deepseek_v3_2": {
        "transport": "openrouter",
        "model": "deepseek/deepseek-v3.2",
        "provider": "DeepInfra",
        "reasoning": {"effort": "none", "exclude": True},
    },
    "claude_opus_5": {
        "transport": "anthropic",
        "model": "claude-opus-5",
        "provider": None,
    },
    "gpt_5_6_sol": {
        "transport": "openai",
        "model": "gpt-5.6-sol",
        "provider": None,
    },
}

DIRECT_PRICES = {
    "claude_opus_5": {"input": 5.0, "output": 25.0},
    "gpt_5_6_sol": {"input": 4.0, "output": 20.0},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def key_for(conversation_id: str, turn: int) -> str:
    return f"{conversation_id}|t{turn}"


def build_request(subject: str, messages: list[dict[str, str]], seed: int) -> tuple[str, dict, dict]:
    spec = SUBJECTS[subject]
    if spec["transport"] == "openrouter":
        body = {
            "model": spec["model"],
            "provider": {"only": [spec["provider"]], "allow_fallbacks": False},
            "messages": messages,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": MAX_TOKENS,
            "seed": seed,
        }
        if "reasoning" in spec:
            body["reasoning"] = spec["reasoning"]
        return OPENROUTER_URL, {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://example.com",
            "X-Title": EXPERIMENT,
        }, body
    if spec["transport"] == "openai":
        return OPENAI_URL, {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        }, {
            "model": spec["model"],
            "reasoning_effort": "none",
            "messages": messages,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_completion_tokens": MAX_TOKENS,
            "seed": seed,
        }
    return ANTHROPIC_URL, {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }, {
        "model": spec["model"],
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 1.0,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
    }


def response_fields(subject: str, data: dict[str, Any]) -> dict[str, Any]:
    transport = SUBJECTS[subject]["transport"]
    if transport in {"openrouter", "openai"}:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content")
        reasoning = message.get("reasoning")
        reasoning_details = message.get("reasoning_details")
        return {
            "visible_text": "" if text is None else str(text),
            "finish_reason": choice.get("finish_reason"),
            "returned_model": data.get("model"),
            "returned_provider": data.get("provider"),
            "response_id": data.get("id"),
            "reasoning": reasoning,
            "reasoning_details": reasoning_details,
            "reasoning_blocks": [],
        }
    blocks = data.get("content") or []
    text = "".join(str(block.get("text") or "") for block in blocks if block.get("type") == "text")
    reasoning_blocks = [block for block in blocks if block.get("type") in {"thinking", "redacted_thinking"}]
    return {
        "visible_text": text,
        "finish_reason": data.get("stop_reason"),
        "returned_model": data.get("model"),
        "returned_provider": None,
        "response_id": data.get("id"),
        "reasoning": None,
        "reasoning_details": None,
        "reasoning_blocks": reasoning_blocks,
    }


def normalized_usage(subject: str, data: dict[str, Any]) -> tuple[dict[str, Any], float]:
    usage = data.get("usage") or {}
    transport = SUBJECTS[subject]["transport"]
    if transport == "anthropic":
        input_tokens = sum(int(usage.get(name) or 0) for name in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"
        ))
        output_tokens = int(usage.get("output_tokens") or 0)
    else:
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
    quoted_cost = usage.get("cost")
    if quoted_cost is None:
        quoted_cost = (usage.get("cost_details") or {}).get("upstream_inference_cost")
    if quoted_cost is not None:
        cost = float(quoted_cost)
    elif subject in DIRECT_PRICES:
        price = DIRECT_PRICES[subject]
        cost = input_tokens / 1e6 * price["input"] + output_tokens / 1e6 * price["output"]
    else:
        cost = 0.0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "raw": usage,
    }, cost


def pin_checks(subject: str, fields: dict[str, Any]) -> tuple[bool, list[str]]:
    spec = SUBJECTS[subject]
    errors = []
    if fields["returned_model"] != spec["model"]:
        errors.append(f"returned_model={fields['returned_model']!r}")
    if fields["returned_provider"] != spec["provider"]:
        errors.append(f"returned_provider={fields['returned_provider']!r}")
    if fields["reasoning"] not in (None, "", [], {}):
        errors.append("nonempty_reasoning")
    if fields["reasoning_details"] not in (None, "", [], {}):
        errors.append("nonempty_reasoning_details")
    if fields["reasoning_blocks"]:
        errors.append("nonempty_reasoning_blocks")
    lowered = fields["visible_text"].lower()
    markers = [marker for marker in THINK_MARKERS if marker in lowered]
    if markers:
        errors.append(f"visible_reasoning_markers={markers}")
    return not errors, errors


def adapt_sampling_fields(transport: str, body: dict[str, Any], error_text: str) -> list[str]:
    """Drop only a direct API's rejected non-load-bearing sampling fields."""
    if transport not in {"openai", "anthropic"}:
        return []
    lowered = error_text.lower()
    removed = []
    for field in ("temperature", "top_p", "seed"):
        if field in body and field in lowered:
            body.pop(field)
            removed.append(field)
    return removed


class Generator:
    def __init__(self, args: argparse.Namespace, rows: list[dict[str, Any]]):
        self.args = args
        self.rows = rows
        self.lock = asyncio.Lock()
        self.semaphores = {
            "openrouter": asyncio.Semaphore(args.openrouter_concurrency),
            "openai": asyncio.Semaphore(args.openai_concurrency),
            "anthropic": asyncio.Semaphore(args.anthropic_concurrency),
        }
        self.subject_semaphores = {
            "qwen_3_5_9b": asyncio.Semaphore(args.qwen_concurrency),
            "deepseek_v3_2": asyncio.Semaphore(args.deepseek_concurrency),
        }
        self.clients: dict[str, httpx.AsyncClient] = {}
        self.final = {
            row["turn_key"]: row
            for row in read_jsonl(args.turns)
            # Only valid content is cache-terminal. A serving-pin failure is
            # preserved as evidence but must receive an identical retry on the
            # next resumable pass; it is never accepted or replacement-sampled.
            if row.get("status") in {"ok", "content_empty"}
        }
        self.progress = Counter()
        self.started = time.time()

    async def persist(self, path: Path, row: dict[str, Any]) -> None:
        async with self.lock:
            append_jsonl(path, row)
            self.args.progress_signal.parent.mkdir(parents=True, exist_ok=True)
            self.args.progress_signal.write_text(now() + "\n")

    async def generate_turn(self, source: dict[str, Any], turn: int, messages: list[dict[str, str]]) -> dict[str, Any]:
        subject = source["subject"]
        turn_key = key_for(source["id"], turn)
        if turn_key in self.final:
            return self.final[turn_key]
        transport = SUBJECTS[subject]["transport"]
        seed = source["request_seeds"][turn - 1]
        url, headers, body = build_request(subject, messages, seed)
        effective_body = json.loads(json.dumps(body))
        adaptations: list[str] = []
        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.time()
            status = None
            data: dict[str, Any] = {}
            raw_text = ""
            try:
                async with self.semaphores[transport]:
                    if subject in self.subject_semaphores:
                        async with self.subject_semaphores[subject]:
                            response = await self.clients[transport].post(
                                url, headers=headers, json=effective_body
                            )
                    else:
                        response = await self.clients[transport].post(
                            url, headers=headers, json=effective_body
                        )
                status = response.status_code
                raw_text = response.text
                try:
                    data = response.json()
                except Exception:
                    data = {"non_json_body": raw_text[:4000]}
                if status != 200:
                    removed = adapt_sampling_fields(transport, effective_body, raw_text)
                    if removed:
                        adaptations.extend(removed)
                        await self.persist(self.args.attempts, {
                            "turn_key": turn_key,
                            "conversation_id": source["id"],
                            "turn": turn,
                            "subject": subject,
                            "attempt": attempt,
                            "status": "request_field_rejected",
                            "http_status": status,
                            "removed_fields": removed,
                            "request_body": body,
                            "raw_response": data,
                            "latency_s": round(time.time() - started, 3),
                            "timestamp_utc": now(),
                        })
                        continue
                    raise RuntimeError(f"HTTP {status}: {raw_text[:1000]}")
                fields = response_fields(subject, data)
                usage, cost = normalized_usage(subject, data)
                pin_ok, pin_errors = pin_checks(subject, fields)
                final_status = "ok" if pin_ok else "serving_pin_failure"
                if pin_ok and not fields["visible_text"]:
                    final_status = "content_empty"
                row = {
                    "turn_key": turn_key,
                    "conversation_id": source["id"],
                    "subject": subject,
                    "battery": source["battery"],
                    "scenario_id": source["scenario_id"],
                    "draw": source["draw"],
                    "turn": turn,
                    "user_text": source["user_turns"][turn - 1],
                    "context_messages": messages,
                    "prompt_sha256": sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
                    "visible_text": fields["visible_text"],
                    "visible_text_sha256": sha256_text(fields["visible_text"]),
                    "finish_reason": fields["finish_reason"],
                    "returned_model": fields["returned_model"],
                    "returned_provider": fields["returned_provider"],
                    "response_id": fields["response_id"],
                    "reasoning": fields["reasoning"],
                    "reasoning_details": fields["reasoning_details"],
                    "reasoning_blocks": fields["reasoning_blocks"],
                    "serving_pin_ok": pin_ok,
                    "serving_pin_errors": pin_errors,
                    "status": final_status,
                    "terminal": True,
                    "attempts": attempt,
                    "request_seed": seed,
                    "seed_requested": "seed" in body,
                    "seed_in_effective_request": "seed" in effective_body,
                    "sampling_mode": "sample",
                    "effective_request": effective_body,
                    "request_adaptations": adaptations,
                    "usage": usage,
                    "cost_usd": cost,
                    "latency_s": round(time.time() - started, 3),
                    "raw_response": data,
                    "timestamp_utc": now(),
                }
                await self.persist(self.args.attempts, {**row, "attempt_record": True})
                await self.persist(self.args.turns, row)
                self.final[turn_key] = row
                self.progress[subject] += 1
                return row
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                attempt_row = {
                    "turn_key": turn_key,
                    "conversation_id": source["id"],
                    "turn": turn,
                    "subject": subject,
                    "attempt": attempt,
                    "status": "transport_failure",
                    "http_status": status,
                    "error": last_error,
                    "request_body": effective_body,
                    "raw_response": data,
                    "latency_s": round(time.time() - started, 3),
                    "timestamp_utc": now(),
                }
                await self.persist(self.args.attempts, attempt_row)
                if status is not None and status not in TRANSIENT:
                    break
                if attempt < MAX_ATTEMPTS:
                    # DeepInfra's upstream shared-pool 429 explicitly asks for
                    # a later retry. Short exponential jitter repeatedly hit
                    # the same overload window for Qwen/DeepSeek, so give
                    # those exact requests a full cooldown before retrying.
                    if status == 429 and subject in {"qwen_3_5_9b", "deepseek_v3_2"}:
                        await asyncio.sleep(60.0 * attempt)
                    else:
                        await asyncio.sleep(
                            min(60.0, 2 ** (attempt - 1)) * (0.5 + random.random())
                        )
        failure = {
            "turn_key": turn_key,
            "conversation_id": source["id"],
            "subject": subject,
            "battery": source["battery"],
            "scenario_id": source["scenario_id"],
            "draw": source["draw"],
            "turn": turn,
            "user_text": source["user_turns"][turn - 1],
            "context_messages": messages,
            "prompt_sha256": sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
            "visible_text": None,
            "finish_reason": None,
            "returned_model": None,
            "returned_provider": None,
            "status": "generation_failure",
            "terminal": True,
            "attempts": MAX_ATTEMPTS,
            "request_seed": seed,
            "effective_request": effective_body,
            "request_adaptations": adaptations,
            "error": last_error,
            "timestamp_utc": now(),
        }
        await self.persist(self.args.turns, failure)
        self.final[turn_key] = failure
        return failure

    async def run(self) -> None:
        timeout = httpx.Timeout(900.0, connect=30.0)
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=80)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as openrouter, \
                httpx.AsyncClient(timeout=timeout, limits=limits) as openai, \
                httpx.AsyncClient(timeout=timeout, limits=limits) as anthropic:
            self.clients = {"openrouter": openrouter, "openai": openai, "anthropic": anthropic}
            histories = {row["id"]: [] for row in self.rows}
            failed = set()
            for turn in range(1, 7):
                tasks = []
                task_sources = []
                for source in self.rows:
                    conversation_id = source["id"]
                    if conversation_id in failed:
                        continue
                    history = histories[conversation_id]
                    if len(history) != 2 * (turn - 1):
                        raise RuntimeError(f"history mismatch {conversation_id} turn={turn}")
                    messages = [*history, {"role": "user", "content": source["user_turns"][turn - 1]}]
                    tasks.append(asyncio.create_task(self.generate_turn(source, turn, messages)))
                    task_sources.append(source)
                pending = set(tasks)
                result_by_task = {task: source for task, source in zip(tasks, task_sources)}
                completed = 0
                while pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        source = result_by_task[task]
                        row = task.result()
                        cid = source["id"]
                        if row["status"] in {"ok", "content_empty"}:
                            histories[cid].extend([
                                {"role": "user", "content": source["user_turns"][turn - 1]},
                                {"role": "assistant", "content": row["visible_text"]},
                            ])
                        else:
                            failed.add(cid)
                        completed += 1
                        if completed % 50 == 0 or completed == len(tasks):
                            elapsed = max(1.0, time.time() - self.started)
                            print(f">>> generation turn {turn}/6: {completed}/{len(tasks)} "
                                  f"active rows, failures={len(failed)}, {sum(self.progress.values())/elapsed:.2f}/s",
                                  flush=True)
                print(f">>> generation turn {turn}/6", flush=True)
                self.write_conversations(histories, failed, turn)

        self.compact_turns()

    def compact_turns(self) -> None:
        """Keep one canonical final row per turn; attempts retain superseded failures."""
        latest = {row["turn_key"]: row for row in read_jsonl(self.args.turns)}
        with self.args.turns.open("w", encoding="utf-8") as handle:
            for turn_key in sorted(latest):
                handle.write(json.dumps(latest[turn_key], ensure_ascii=False, sort_keys=True) + "\n")

    def write_conversations(self, histories: dict[str, list[dict]], failed: set[str], upto_turn: int) -> None:
        finals = {row["turn_key"]: row for row in read_jsonl(self.args.turns)}
        rows = []
        for source in self.rows:
            cid = source["id"]
            turns = [finals.get(key_for(cid, turn)) for turn in range(1, 7)]
            present = [row for row in turns if row is not None]
            complete = len(present) == 6 and all(row["status"] in {"ok", "content_empty"} for row in present)
            text_parts = []
            for turn, record in enumerate(present, 1):
                text_parts.append(f"USER (turn {turn}):\n{record['user_text']}")
                if record.get("visible_text") is not None:
                    text_parts.append(f"ASSISTANT (turn {turn}):\n{record['visible_text']}")
            trajectory = "\n\n".join(text_parts)
            failure_rows = [row for row in present if row["status"] not in {"ok", "content_empty"}]
            rows.append({
                "id": cid,
                "conversation_id": cid,
                "subject": source["subject"],
                "battery": source["battery"],
                "scenario_id": source["scenario_id"],
                "scenario_index": source["scenario_index"],
                "draw": source["draw"],
                "source_sha256": source["source_sha256"],
                "complete": complete,
                "status": "complete" if complete else "explicit_generation_failure",
                "turns_completed": sum(row["status"] in {"ok", "content_empty"} for row in present),
                "turn_records_present": len(present),
                "failure": failure_rows[0] if failure_rows else None,
                "trajectory": trajectory,
                "transcript_sha256": sha256_text(trajectory),
                "n_chars": len(trajectory),
                "n_empty_turns": sum(row["status"] == "content_empty" for row in present),
                "n_capped_turns": sum(row.get("finish_reason") in {"length", "max_tokens"} for row in present),
                "n_refusal_turns": sum(row.get("finish_reason") in {"refusal", "content_filter"} for row in present),
                "generation_cost_usd": sum(float(row.get("cost_usd") or 0) for row in present),
                "gen_config": {
                    "temperature_requested": 1.0,
                    "top_p_requested": 1.0,
                    "max_new_tokens": MAX_TOKENS,
                    "sampling_mode": "sample",
                    "request_seeds": source["request_seeds"],
                    "no_system_message": True,
                    "no_tools": True,
                },
                "checkpoint_turn_layer": upto_turn,
            })
        self.args.conversations.parent.mkdir(parents=True, exist_ok=True)
        with self.args.conversations.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.args.progress_signal.write_text(now() + "\n")


def smoke_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for subject in SUBJECTS:
        matches = [row for row in rows if row["subject"] == subject and row["battery"] == "neutral"
                   and row["scenario_index"] == 0 and row["draw"] == 0]
        if len(matches) != 1:
            raise RuntimeError(f"smoke selection for {subject}: {len(matches)}")
        selected.extend(matches)
    return selected


def verify_smoke(turns_path: Path, conversations_path: Path, out: Path) -> None:
    turns = list({row["turn_key"]: row for row in read_jsonl(turns_path)}.values())
    conversations = read_jsonl(conversations_path)
    expected_reasoning = {"qwen_3_5_9b", "deepseek_v3_2"}
    report = {
        "n_conversations": len(conversations),
        "n_turns": len(turns),
        "all_conversations_complete": all(row.get("complete") for row in conversations),
        "all_turns_nonempty": all(bool(row.get("visible_text")) for row in turns),
        "all_serving_pins_ok": all(row.get("serving_pin_ok") for row in turns),
        "fallback_count": sum(row.get("returned_provider") not in {None, "DeepInfra"} for row in turns),
        "reasoning_fields_nonempty": sum(
            row.get("reasoning") not in (None, "", [], {})
            or row.get("reasoning_details") not in (None, "", [], {})
            or bool(row.get("reasoning_blocks")) for row in turns
        ),
        "visible_think_marker_count": sum(
            any(marker in str(row.get("visible_text") or "").lower() for marker in THINK_MARKERS)
            for row in turns
        ),
        "exact_reasoning_off_request_turns": sum(
            row["subject"] in expected_reasoning
            and row.get("effective_request", {}).get("reasoning") == {"effort": "none", "exclude": True}
            for row in turns
        ),
        "request_adaptations": {
            row["subject"]: row.get("request_adaptations", []) for row in turns
            if row.get("request_adaptations")
        },
        "returned_models": sorted({row.get("returned_model") for row in turns if row.get("returned_model")}),
        "returned_providers": sorted({str(row.get("returned_provider")) for row in turns}),
    }
    report["pass"] = (
        report["n_conversations"] == 7
        and report["n_turns"] == 42
        and report["all_conversations_complete"]
        and report["all_turns_nonempty"]
        and report["all_serving_pins_ok"]
        and report["fallback_count"] == 0
        and report["reasoning_fields_nonempty"] == 0
        and report["visible_think_marker_count"] == 0
        and report["exact_reasoning_off_request_turns"] == 12
    )
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit("production-exact smoke failed")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--turns", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--conversations", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--progress-signal", type=Path, required=True)
    parser.add_argument("--openrouter-concurrency", type=int, default=20)
    parser.add_argument("--qwen-concurrency", type=int, default=2)
    parser.add_argument("--deepseek-concurrency", type=int, default=2)
    parser.add_argument("--openai-concurrency", type=int, default=20)
    parser.add_argument("--anthropic-concurrency", type=int, default=20)
    args = parser.parse_args()
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} is not set")
    rows = read_jsonl(args.input)
    if len(rows) != 756 or len({row["id"] for row in rows}) != 756:
        raise SystemExit("input must contain 756 unique conversation rows")
    if args.stage == "smoke":
        rows = smoke_subset(rows)
    generator = Generator(args, rows)
    await generator.run()
    if args.stage == "smoke":
        if args.smoke_report is None:
            raise SystemExit("--smoke-report is required for smoke")
        verify_smoke(args.turns, args.conversations, args.smoke_report)


if __name__ == "__main__":
    asyncio.run(main())
