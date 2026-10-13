#!/usr/bin/env python3
"""Blind, resumable coherence classification for frozen Leg-2 transcripts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


RUBRIC_V1 = """You are a narrow coherence classifier for multi-turn AI-assistant transcripts used in an emotion evaluation.

Classify the transcript as COHERENT or INCOHERENT.

COHERENT means the assistant's responses remain understandable and responsive enough that their emotional tone can be meaningfully judged. A response can still be factually wrong, sycophantic, alarmist, angry, self-abasing, overly emotional, repetitive in its ideas, or excessively verbose and remain COHERENT.

INCOHERENT means at least one assistant turn substantially breaks down through uncontrolled verbatim or near-verbatim repetition, word salad, badly broken fragments, runaway generation, or generated internal/template/control text that makes the conversation unreliable for semantic emotion scoring. A single typo, foreign token, strange claim, or ordinary factual error is not enough.

Judge only coherence. Do not judge correctness, safety, politeness, emotional appropriateness, or response quality. Return the required JSON."""

RUBRIC_V2 = """You are a narrow coherence classifier for multi-turn AI-assistant transcripts used in an emotion evaluation.

The central question is: ignoring factual accuracy and response quality, can a competent human still understand the assistant's emotional stance well enough to score its emotional tone? If yes, classify COHERENT.

COHERENT includes transcripts that are factually wrong, hallucinate names or places, give bad advice, contradict themselves, capitulate, expose a self-correction, contain isolated foreign or malformed tokens, include an irrelevant sentence, are sycophantic, alarmist, angry, self-abasing, overly emotional, repetitive in ideas, or excessively verbose — as long as the overall responses remain understandable and their emotional tone is judgeable.

INCOHERENT requires a substantial generation breakdown: at least one assistant turn is dominated by uncontrolled verbatim or near-verbatim looping, sustained word salad, badly broken fragments, runaway generation, or generated template/control text, to the point that the conversation is unreliable for semantic emotion scoring. A few odd phrases or a locally garbled sentence are not enough.

Important consistency rule: if your evidence says the conversation remains semantically understandable or that its emotional tone is still judgeable, the label must be COHERENT. Do not use factual unreliability as a synonym for incoherence.

Judge only coherence. Do not judge correctness, safety, politeness, emotional appropriateness, or response quality. Return the required JSON."""

RUBRIC_V3 = """You are a narrow coherence classifier for multi-turn AI-assistant transcripts used in an emotion evaluation.

Classify the whole transcript INCOHERENT if ANY assistant turn contains a substantial generation breakdown. Other turns being readable does not erase one broken turn.

A substantial generation breakdown means one of these:
- a whole turn, or a substantial part of one, is template/control/copyright text or raw markup instead of a response;
- phrases, sentences, or paragraphs repeat uncontrollably multiple times, often until truncation;
- a substantial passage becomes broken fragments, letter salad, literal escape/control sequences, or sustained word salad;
- extended unrelated nonsense dominates a passage so it no longer functions as an understandable response.

Otherwise classify COHERENT. In particular, these are NOT incoherence by themselves: invented or wrong place names, impossible routes, factual hallucinations, bad advice, contradictions, capitulation, exposed self-correction, one irrelevant claim, isolated foreign tokens, one locally malformed sentence, sycophancy, alarmism, anger, self-abasement, emotional intensity, repetition of ideas, or verbosity. Factual unreliability is never sufficient.

Use the form of the generated language, not whether its claims are true. One genuinely broken turn is sufficient; scattered local glitches in an otherwise usable turn are not.

Set `emotion_judgeable` false and name a non-`none` breakdown type exactly when the label is INCOHERENT. Return the required JSON."""

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
    parser.add_argument("--rubric-version", default="v3", choices=["v1", "v2", "v3"])
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
    rubric = {"v1": RUBRIC_V1, "v2": RUBRIC_V2, "v3": RUBRIC_V3}[args.rubric_version]

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


if __name__ == "__main__":
    asyncio.run(main())
