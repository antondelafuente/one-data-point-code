#!/usr/bin/env python3
"""Blind, resumable coherence classification for this experiment's Leg-B transcripts.

Adapted from depv1-qwen-chat-emotion-coherence-1/scripts/classify_coherence.py at the pinned
commit e1fba2c1073cd3f1fb00582cfa468dc1c19b135c. The ONLY change: the rubric is not embedded —
it is READ from the byte-frozen `design_inputs/COHERENCE_PROMPT_luna_v3.md` and its sha256 is
asserted against `design_receipt.json`, so the instrument cannot drift. Model, reasoning effort,
strict schema, retry policy (initial + 2 retries, never coerced) and the persisted receipt fields
are unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


def load_rubric(prompt_path: Path, receipt_path: Path, receipt_key: str) -> tuple[str, str]:
    raw = prompt_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    pinned = json.loads(receipt_path.read_text())["files"][receipt_key]
    if digest != pinned:
        raise SystemExit(f"frozen rubric sha mismatch: {digest} != {pinned}")
    return raw.decode().strip(), digest


SCHEMA = {
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
                    "enum": [
                        "none",
                        "verbatim_loop",
                        "word_salad",
                        "broken_fragments",
                        "runaway_generation",
                        "control_or_template_text",
                        "other",
                    ],
                },
                "evidence": {"type": "string"},
            },
            "required": ["label", "confidence", "emotion_judgeable", "breakdown_type", "evidence"],
            "additionalProperties": False,
        },
    },
}

INPUT_USD_PER_M = {"gpt-5.6-luna": 0.20, "gpt-5.6-terra": 0.40}
OUTPUT_USD_PER_M = {"gpt-5.6-luna": 1.20, "gpt-5.6-terra": 2.40}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["id"] for row in read_jsonl(path)}


def usage_dict(usage: Any) -> dict[str, int]:
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


async def classify_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    model: str,
    rubric: str,
) -> dict[str, Any]:
    transcript = row["trajectory"]
    input_sha = hashlib.sha256(transcript.encode()).hexdigest()
    last_error = None
    for attempt in range(1, 4):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    reasoning_effort="low",
                    messages=[
                        {"role": "system", "content": rubric},
                        {"role": "user", "content": transcript},
                    ],
                    response_format=SCHEMA,
                    max_completion_tokens=400,
                )
            content = response.choices[0].message.content or ""
            parsed = json.loads(content)
            if parsed.get("label") not in {"coherent", "incoherent"}:
                raise ValueError(f"bad label: {parsed.get('label')!r}")
            if parsed.get("confidence") not in {"high", "medium", "low"}:
                raise ValueError(f"bad confidence: {parsed.get('confidence')!r}")
            if (parsed["label"] == "coherent") != bool(parsed["emotion_judgeable"]):
                raise ValueError("label/emotion_judgeable inconsistency")
            if parsed["label"] == "coherent" and parsed["breakdown_type"] != "none":
                raise ValueError("coherent label must use breakdown_type=none")
            if parsed["label"] == "incoherent" and parsed["breakdown_type"] == "none":
                raise ValueError("incoherent label must name a breakdown_type")
            return {
                "id": row["id"],
                "input_sha256": input_sha,
                "label": parsed["label"],
                "confidence": parsed["confidence"],
                "emotion_judgeable": parsed["emotion_judgeable"],
                "breakdown_type": parsed["breakdown_type"],
                "evidence": parsed["evidence"],
                "parse_ok": True,
                "attempts": attempt,
                "response_id": response.id,
                "returned_model": response.model,
                "usage": usage_dict(response.usage),
                "raw_content": content,
            }
        except Exception as exc:  # preserve failures; do not silently coerce to coherent
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                await asyncio.sleep(1.5 * attempt)
    return {
        "id": row["id"],
        "input_sha256": input_sha,
        "label": None,
        "confidence": None,
        "evidence": None,
        "parse_ok": False,
        "attempts": 3,
        "error": last_error,
    }


def estimated_spend(rows: list[dict[str, Any]], model: str) -> float:
    prompt = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in rows)
    completion = sum(r.get("usage", {}).get("completion_tokens", 0) for r in rows)
    return (
        prompt * INPUT_USD_PER_M[model] / 1_000_000
        + completion * OUTPUT_USD_PER_M[model] / 1_000_000
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna", choices=sorted(INPUT_USD_PER_M))
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-key", default="design_inputs/COHERENCE_PROMPT_luna_v3.md")
    parser.add_argument("--receipt-out", type=Path)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--hard-stop-usd", type=float, default=10.0)
    parser.add_argument("--ids", type=Path, help="Optional newline-delimited subset of source ids")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY is not set")

    rows = read_jsonl(args.input)
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("source ids are not unique")
    if args.ids:
        wanted = {line.strip() for line in args.ids.read_text().splitlines() if line.strip()}
        rows = [r for r in rows if r["id"] in wanted]
        missing = wanted - {r["id"] for r in rows}
        if missing:
            raise SystemExit(f"subset ids missing from source: {len(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_ids(args.output)
    pending = [r for r in rows if r["id"] not in done]
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)
    rubric, rubric_sha = load_rubric(args.prompt, args.receipt, args.receipt_key)
    print(json.dumps({"rubric_sha256": rubric_sha, "model": args.model,
                      "reasoning_effort": "low", "pending": len(pending),
                      "already_done": len(done)}), flush=True)

    for start in range(0, len(pending), 100):
        batch = pending[start : start + 100]
        results = await asyncio.gather(
            *(classify_one(client, semaphore, row, args.model, rubric) for row in batch)
        )
        with args.output.open("a") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        all_results = read_jsonl(args.output)
        spend = estimated_spend(all_results, args.model)
        print(
            json.dumps(
                {
                    "completed": len(all_results),
                    "target": len(rows),
                    "parse_failures": sum(not r.get("parse_ok", False) for r in all_results),
                    "estimated_usd": round(spend, 4),
                }
            ),
            flush=True,
        )
        if spend > args.hard_stop_usd:
            raise SystemExit(f"hard stop exceeded: ${spend:.2f}")

    final = read_jsonl(args.output) if args.output.exists() else []
    # The output file is resumable and shared across invocations, so status compares the rows
    # requested by THIS invocation against the rows present for them — not the file's total,
    # which legitimately carries earlier batches (e.g. the price sample) as well.
    wanted = {r["id"] for r in rows}
    for_target = [r for r in final if r["id"] in wanted]
    receipt = {
        "status": "PASS" if len(for_target) == len(rows) else "FAIL",
        "n_classified_for_target": len(for_target),
        "n_rows_in_output_file": len(final),
        "model": args.model, "reasoning_effort": "low",
        "rubric_path": str(args.prompt), "rubric_sha256": rubric_sha,
        "schema": SCHEMA["json_schema"]["name"], "strict": True,
        "retries_per_row": 3,
        "n_target": len(rows),
        "n_parse_failures_for_target": sum(not r.get("parse_ok", False) for r in for_target),
        "n_parse_failures_in_output_file": sum(not r.get("parse_ok", False) for r in final),
        "labels_for_target": {label: sum(1 for r in for_target if r.get("label") == label)
                              for label in ("coherent", "incoherent")},
        "labels_in_output_file": {label: sum(1 for r in final if r.get("label") == label)
                                  for label in ("coherent", "incoherent")},
        "returned_models": sorted({r.get("returned_model") for r in final if r.get("returned_model")}),
        "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in final),
        "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in final),
        "input_usd_per_m": INPUT_USD_PER_M[args.model],
        "output_usd_per_m": OUTPUT_USD_PER_M[args.model],
        "estimated_usd_from_usage_receipts": round(estimated_spend(final, args.model), 6),
    }
    if args.receipt_out:
        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
