#!/usr/bin/env python3
"""Blind, success-aware direct-API coherence and emotion scoring."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
PRICES = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (0.40, 2.40),
    "claude-sonnet-5": (3.00, 15.00),
}
COHERENCE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "coherence_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["coherent", "incoherent"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "emotion_judgeable": {"type": "boolean"},
                "breakdown_type": {
                    "type": "string",
                    "enum": ["none", "verbatim_loop", "word_salad", "broken_fragments",
                             "runaway_generation", "control_or_template_text", "other"],
                },
                "evidence": {"type": "string"},
            },
            "required": ["label", "confidence", "emotion_judgeable", "breakdown_type", "evidence"],
            "additionalProperties": False,
        },
    },
}
EMOTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "emotion_score",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 10},
                           "quote": {"type": "string"}},
            "required": ["score", "quote"],
            "additionalProperties": False,
        },
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def compact_latest(path: Path) -> None:
    """Keep one canonical latest row per id after a success-aware retry pass."""
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[row["id"]] = row
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row_id in sorted(latest):
            handle.write(json.dumps(latest[row_id], ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def strip_markdown_emphasis(text: str) -> str:
    """A4: remove only Markdown emphasis delimiters, with no other normalization."""
    for delimiter in ("**", "__", "*", "_"):
        text = text.replace(delimiter, "")
    return text


def parse(task: str, text: str, transcript: str) -> dict[str, Any]:
    value = json.loads(text)
    if task == "emotion":
        score, quote = value.get("score"), value.get("quote")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 10:
            raise ValueError("score is not an integer in [0,10]")
        if not isinstance(quote, str):
            raise ValueError("quote is not a string")
        if score > 0 and not quote:
            raise ValueError("positive score has an empty quote")
        if not quote:
            grounding_mode = "exact"
        elif quote in transcript:
            grounding_mode = "exact"
        elif quote in strip_markdown_emphasis(transcript):
            grounding_mode = "markdown_emphasis_stripped"
        else:
            raise ValueError("quote is not a verbatim transcript substring")
        return {"score": score, "quote": quote, "quote_grounded": True,
                "grounding_mode": grounding_mode}
    label, confidence = value.get("label"), value.get("confidence")
    judgeable, breakdown = value.get("emotion_judgeable"), value.get("breakdown_type")
    if label not in {"coherent", "incoherent"} or confidence not in {"high", "medium", "low"}:
        raise ValueError("invalid coherence label/confidence")
    if not isinstance(judgeable, bool) or judgeable != (label == "coherent"):
        raise ValueError("label/emotion_judgeable mismatch")
    if (label == "coherent") != (breakdown == "none"):
        raise ValueError("label/breakdown mismatch")
    if not isinstance(value.get("evidence"), str):
        raise ValueError("evidence is not a string")
    return {k: value[k] for k in ("label", "confidence", "emotion_judgeable",
                                   "breakdown_type", "evidence")}


class Runner:
    def __init__(self, args: argparse.Namespace, rows: list[dict[str, Any]]):
        self.a, self.rows = args, rows
        self.sem = asyncio.Semaphore(args.concurrency)
        self.lock = asyncio.Lock()
        self.started = time.time()
        self.stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0,
                      "recovered_from_attempt_records": 0}
        terminal = {}
        if args.output.exists():
            for row in read_jsonl(args.output):
                if row.get("status") in {"ok", "judge_refusal"}:
                    terminal[row["id"]] = row
        self.done = terminal

    def reparse_successful_rows(self) -> None:
        """Backfill current parser metadata from preserved raw successful responses."""
        if self.a.task != "emotion" or not self.a.output.exists():
            return
        inputs = {row["id"]: row for row in self.rows}
        for existing in read_jsonl(self.a.output):
            row_id = existing.get("id")
            if (existing.get("status") != "ok" or
                    existing.get("grounding_mode") in {"exact", "markdown_emphasis_stripped"}):
                continue
            if row_id not in inputs or not isinstance(existing.get("raw_content"), str):
                raise RuntimeError(f"cannot reparse successful row {row_id!r} from preserved content")
            transcript, _, _ = self.prompt(inputs[row_id])
            parsed = parse(self.a.task, existing["raw_content"], transcript)
            updated = {**existing, **parsed, "reparsed_under_current_grounding_rule": True}
            append(self.a.output, updated)
            self.done[row_id] = updated

    def recover_parseable_attempts(self) -> None:
        """Reparse preserved HTTP-200 attempts before making any replacement call."""
        if not self.a.attempts.exists():
            return
        inputs = {row["id"]: row for row in self.rows}
        for attempt_row in reversed(read_jsonl(self.a.attempts)):
            row_id = attempt_row.get("id")
            if row_id in self.done or row_id not in inputs or attempt_row.get("http_status") != 200:
                continue
            data = attempt_row.get("raw_response") or {}
            try:
                transcript, _, input_hash = self.prompt(inputs[row_id])
                if self.a.model.startswith("gpt-"):
                    message = data["choices"][0]["message"]
                    text = str(message.get("content") or "")
                    usage = data.get("usage") or {}
                    returned = data.get("model")
                    response_id = data.get("id")
                    if message.get("refusal"):
                        continue
                else:
                    text = "".join(str(block.get("text") or "") for block in data.get("content", [])
                                   if block.get("type") == "text")
                    usage = data.get("usage") or {}
                    returned = data.get("model")
                    response_id = data.get("id")
                    if data.get("stop_reason") == "refusal":
                        continue
                if returned != self.a.model:
                    continue
                parsed = parse(self.a.task, text, transcript)
            except Exception:  # noqa: BLE001
                continue
            final = {
                "id": row_id,
                "task": self.a.task,
                "axis": self.a.axis,
                "judge_model": self.a.model,
                "input_sha256": input_hash,
                "prompt_sha256": hashlib.sha256(self.a.prompt.read_bytes()).hexdigest(),
                **parsed,
                "status": "ok",
                "parse_ok": True,
                "attempts": attempt_row.get("attempt"),
                "returned_model": returned,
                "response_id": response_id,
                "usage": usage,
                "raw_content": text,
                "raw_response": data,
                "recovered_from_attempt_record": True,
                "timestamp_utc": now(),
            }
            append(self.a.output, final)
            self.done[row_id] = final
            self.stats["recovered_from_attempt_records"] += 1

    def prompt(self, row: dict[str, Any]) -> tuple[str, list[dict[str, str]], str]:
        transcript = row.get("trajectory", row.get("transcript"))
        if not isinstance(transcript, str):
            raise ValueError(f"missing transcript for {row.get('id')}")
        rubric = self.a.prompt.read_text()
        if self.a.task == "coherence":
            messages = [{"role": "system", "content": rubric}, {"role": "user", "content": transcript}]
        else:
            messages = [{"role": "user", "content": rubric + "\n\n## Conversation\n\n" + transcript}]
        return transcript, messages, hashlib.sha256(transcript.encode()).hexdigest()

    async def persist(self, path: Path, row: dict[str, Any]) -> None:
        async with self.lock:
            append(path, row)
            self.a.progress_signal.write_text(now() + "\n")

    async def one(self, client: httpx.AsyncClient, row: dict[str, Any]) -> dict[str, Any]:
        transcript, messages, input_hash = self.prompt(row)
        model = self.a.model
        base = {"id": row["id"], "task": self.a.task, "axis": self.a.axis,
                "judge_model": model, "input_sha256": input_hash,
                "prompt_sha256": hashlib.sha256(self.a.prompt.read_bytes()).hexdigest()}
        if model.startswith("gpt-"):
            url = OPENAI_URL
            headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
            body = {"model": model, "reasoning_effort": "low", "messages": messages,
                    "response_format": COHERENCE_SCHEMA if self.a.task == "coherence" else EMOTION_SCHEMA,
                    "max_completion_tokens": 500 if self.a.task == "coherence" else 2000}
        else:
            url = ANTHROPIC_URL
            headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                       "anthropic-version": "2023-06-01"}
            anthropic_messages = messages
            system = None
            if messages and messages[0].get("role") == "system":
                system = messages[0]["content"]
                anthropic_messages = messages[1:]
            schema = (COHERENCE_SCHEMA if self.a.task == "coherence" else EMOTION_SCHEMA)[
                "json_schema"
            ]["schema"]
            body = {"model": model, "messages": anthropic_messages, "max_tokens": 2000,
                    "output_config": {"effort": "low", "format": {
                        "type": "json_schema", "schema": schema,
                    }}}
            if system is not None:
                body["system"] = system
        last_error = None
        for attempt in (1, 2):
            started = time.time()
            status, data = None, {}
            try:
                async with self.sem:
                    response = await client.post(url, headers=headers, json=body)
                status = response.status_code
                data = response.json()
                if status != 200:
                    raise RuntimeError(f"HTTP {status}: {response.text[:500]}")
                if model.startswith("gpt-"):
                    text = str(data["choices"][0]["message"].get("content") or "")
                    usage = data.get("usage") or {}
                    in_tok, out_tok = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
                    returned, response_id = data.get("model"), data.get("id")
                    refusal = bool(data["choices"][0]["message"].get("refusal"))
                else:
                    text = "".join(str(b.get("text") or "") for b in data.get("content", []) if b.get("type") == "text")
                    usage = data.get("usage") or {}
                    in_tok = sum(int(usage.get(k) or 0) for k in
                                 ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
                    out_tok = int(usage.get("output_tokens") or 0)
                    returned, response_id = data.get("model"), data.get("id")
                    refusal = data.get("stop_reason") == "refusal"
                self.stats["calls"] += 1
                self.stats["input_tokens"] += in_tok
                self.stats["output_tokens"] += out_tok
                if returned != model:
                    raise ValueError(f"returned model mismatch: {returned!r}")
                if refusal:
                    final = {**base, "status": "judge_refusal", "parse_ok": False,
                             "attempts": attempt, "returned_model": returned,
                             "response_id": response_id, "usage": usage, "raw_response": data,
                             "timestamp_utc": now()}
                    await self.persist(self.a.output, final)
                    return final
                parsed = parse(self.a.task, text, transcript)
                final = {**base, **parsed, "status": "ok", "parse_ok": True,
                         "attempts": attempt, "returned_model": returned,
                         "response_id": response_id, "usage": usage, "raw_content": text,
                         "raw_response": data, "latency_s": round(time.time() - started, 3),
                         "timestamp_utc": now()}
                await self.persist(self.a.attempts, {**final, "attempt_record": True})
                await self.persist(self.a.output, final)
                return final
            except Exception as exc:  # noqa: BLE001
                self.stats["errors"] += 1
                last_error = f"{type(exc).__name__}: {exc}"
                await self.persist(self.a.attempts, {**base, "attempt": attempt,
                    "status": "attempt_failure", "http_status": status, "error": last_error,
                    "raw_response": data, "timestamp_utc": now()})
                if attempt == 1:
                    await asyncio.sleep(2)
        final = {**base, "status": "judge_failure", "parse_ok": False, "attempts": 2,
                 "error": last_error, "timestamp_utc": now()}
        await self.persist(self.a.output, final)
        return final

    async def run(self) -> None:
        self.reparse_successful_rows()
        self.recover_parseable_attempts()
        pending_rows = [row for row in self.rows if row["id"] not in self.done]
        timeout = httpx.Timeout(900, connect=30)
        async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=80)) as client:
            pending = {asyncio.create_task(self.one(client, row)) for row in pending_rows}
            completed = 0
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                completed += len(done)
                for task in done:
                    task.result()
                if completed % 25 == 0 or not pending:
                    print(json.dumps({"completed": completed, "pending": len(pending),
                        "rate_per_s": round(completed / max(1, time.time() - self.started), 3),
                        "calls": self.stats["calls"], "errors": self.stats["errors"]}), flush=True)
        compact_latest(self.a.output)
        pin, pout = PRICES[self.a.model]
        cost = self.stats["input_tokens"] / 1e6 * pin + self.stats["output_tokens"] / 1e6 * pout
        report = {**self.stats, "model": self.a.model, "task": self.a.task, "axis": self.a.axis,
                  "requested_rows": len(pending_rows), "cost_usd": round(cost, 8),
                  "elapsed_s": round(time.time() - self.started, 3)}
        self.a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("coherence", "emotion"), required=True)
    ap.add_argument("--axis", choices=("despair", "elation"))
    ap.add_argument("--model", choices=tuple(PRICES), required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--attempts", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--progress-signal", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=30)
    a = ap.parse_args()
    if a.task == "emotion" and not a.axis:
        ap.error("--axis is required for emotion")
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(key):
            raise SystemExit(f"{key} is not set")
    rows = read_jsonl(a.input)
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("input ids are not unique")
    # Calibration fixtures carry an axis; target transcripts intentionally do not.
    if a.task == "emotion" and any("axis" in r for r in rows):
        rows = [r for r in rows if r.get("axis") == a.axis]
    if a.ids:
        wanted = {line.strip() for line in a.ids.read_text().splitlines() if line.strip()}
        rows = [r for r in rows if r["id"] in wanted]
        if {r["id"] for r in rows} != wanted:
            raise SystemExit("ids subset does not exactly join input")
    await Runner(a, rows).run()


if __name__ == "__main__":
    asyncio.run(main())
