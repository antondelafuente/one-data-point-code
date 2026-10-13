#!/usr/bin/env python3
"""The pinned negative-emotion instrument, on the DESIGN's pinned transport.

DESIGN: judge `claude-sonnet-4-5-20250929`, the depv1 thread's pinned judge prompt
(source `dump.py` sha 867f56b2...), first-12,000-character input cap. Transport:
OpenRouter, provider `Anthropic` forced, fallbacks disabled, returned model id
checked on every call.

TRANSPORT NOTE (executor, flagged to the designer-of-record 2026-08-07).
OpenRouter exposes exactly one Sonnet-4.5 slug, `anthropic/claude-sonnet-4.5`
(checked against `/api/v1/models`: no dated variant exists, only the `:batch`
sibling). It therefore CANNOT return the literal string
`claude-sonnet-4-5-20250929` that the DESIGN's per-call identity check names.
Under this transport the strongest available per-call identity assertion is the
conjunction actually returned by the API, and that is what is enforced here,
fail-closed, on EVERY call:

    response.model    == "anthropic/claude-sonnet-4.5"
    response.provider == "Anthropic"

with `provider.only=["Anthropic"]` and `allow_fallbacks=false` on the request, so
no other vendor's re-host of the same nominal model can serve a row. A call whose
returned identity does not match is recorded as missingness (`parse_status:
missing`, `terminal_error: model_identity_mismatch:...`), never silently accepted.

Both facts are reported: the requested id, and the returned id.

This module is used by BOTH the construct-time negativity screens (gemma-fresh's
keep rule, the report-only inkling screens) and the main blinded wave, so the same
instrument+transport rates every surface in this experiment.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

API = "https://openrouter.ai/api/v1/chat/completions"
MODEL_SLUG = "anthropic/claude-sonnet-4.5"
MODEL_PINNED_ID = "claude-sonnet-4-5-20250929"
PROVIDER = "Anthropic"
CAP_CHARS = 12_000
# OpenRouter bills at the vendor's own rate; recorded per call from `usage.cost`.
INPUT_USD_PER_MTOK = 3.0
OUTPUT_USD_PER_MTOK = 15.0


def load_dump(work_inputs: Path):
    """Import the pinned judge-prompt source after verifying its sha256."""
    sys.path.insert(0, str(Path(work_inputs).resolve()))
    import dis1_common as C  # noqa: PLC0415
    C.verify_pins({"dump.py": Path(work_inputs) / "dump.py"})
    import dump  # noqa: PLC0415
    return dump


def negativity_prompt(dump, response: str) -> str:
    return dump.get_negativity_judge_prompt(response[:CAP_CHARS])


def coherence_prompt(rubric_text: str, battery_prompt: str, response: str) -> str:
    return (f"{rubric_text.rstrip()}\n\n"
            f"PROMPT:\n{battery_prompt}\n\nRESPONSE:\n{response[:CAP_CHARS]}\n")


def parse_negativity(text: str) -> dict[str, Any]:
    c = text.strip()
    if c.startswith("```"):
        c = re.sub(r"^```(?:json)?\s*", "", c)
        c = re.sub(r"\s*```$", "", c)
    if not c.startswith("{"):
        m = re.search(r"\{.*\}", c, re.DOTALL)
        if m:
            c = m.group()
    obj = json.loads(c)
    if not isinstance(obj, dict) or "rating" not in obj:
        raise ValueError("judge JSON lacks rating")
    rating = float(obj["rating"])
    if not 0 <= rating <= 10:
        raise ValueError(f"rating out of range: {rating}")
    if not rating.is_integer():
        raise ValueError(f"rating is not an integer: {rating}")
    ev, rs = obj.get("evidence", ""), obj.get("reasoning", "")
    if not isinstance(ev, str) or not isinstance(rs, str):
        raise ValueError("evidence/reasoning are not strings")
    return {"rating": int(rating), "evidence": ev, "reasoning": rs}


def parse_coherence(text: str) -> dict[str, Any]:
    m = re.search(r"^\s*RATING:\s*(\d{1,2})\s*$", text, re.MULTILINE)
    if not m:
        raise ValueError("no RATING: line")
    rating = int(m.group(1))
    if not 0 <= rating <= 10:
        raise ValueError(f"rating out of range: {rating}")
    ev = re.search(r"^\s*EVIDENCE:\s*(.*)$", text, re.MULTILINE)
    return {"rating": rating, "evidence": (ev.group(1).strip() if ev else ""),
            "reasoning": ""}


def need_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    return key


def extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return msg.get("content") or ""


def cost_of(payload: dict[str, Any]) -> float | None:
    usage = payload.get("usage") or {}
    if isinstance(usage.get("cost"), (int, float)):
        return float(usage["cost"])
    inp, out = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(inp, (int, float)) or not isinstance(out, (int, float)):
        return None
    return (float(inp) * INPUT_USD_PER_MTOK + float(out) * OUTPUT_USD_PER_MTOK) / 1e6


async def judge_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, key: str,
                    prompt: str, parse, max_tokens_ladder: list[int],
                    reasoning: dict | None, row_id: str,
                    view: str = "full", rubric: str = "negativity") -> dict[str, Any]:
    """One judged row: the pinned instrument, provider-forced, identity-checked.

    Retry ladder mirrors the thread's frozen wave (one retry at a larger token
    budget for an empty/unparseable result). The semaphore brackets the SINGLE
    HTTP attempt; the backoff sleep sits OUTSIDE the permit and is jittered (#343).
    """
    base: dict[str, Any] = {
        "model": MODEL_SLUG,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "provider": {"only": [PROVIDER], "allow_fallbacks": False},
        "usage": {"include": True},
    }
    if reasoning:
        base["reasoning"] = reasoning
    attempts: list[dict] = []
    terminal: str | None = None
    parsed: dict | None = None
    final: dict | None = None
    for i, max_tokens in enumerate(max_tokens_ladder, start=1):
        started = time.time()
        try:
            async with sem:
                r = await client.post(
                    API,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    json={**base, "max_tokens": max_tokens})
            ended = time.time()
            try:
                payload = r.json()
            except Exception:  # noqa: BLE001
                payload = {"non_json_body": r.text[:20000]}
            attempts.append({"attempt": i, "max_tokens": max_tokens,
                             "http_status": r.status_code, "latency_s": ended - started,
                             "response": payload})
            if r.status_code == 429 or r.status_code >= 500:
                terminal = f"http_{r.status_code}"
                if i < len(max_tokens_ladder):
                    await asyncio.sleep(2.0 * (0.5 + random.random()))
                    continue
                break
            if r.status_code != 200:
                terminal = f"http_{r.status_code}"
                break
            if payload.get("error"):
                terminal = f"api_error:{json.dumps(payload['error'])[:200]}"
                if i < len(max_tokens_ladder):
                    await asyncio.sleep(2.0 * (0.5 + random.random()))
                    continue
                break
            final = payload
            # --- per-call identity gate, fail-closed (see module docstring) ----
            if payload.get("model") != MODEL_SLUG or payload.get("provider") != PROVIDER:
                terminal = (f"model_identity_mismatch:returned={payload.get('model')}"
                            f"/{payload.get('provider')} expected={MODEL_SLUG}/{PROVIDER}")
                break
            content = extract_content(payload)
            if not content.strip():
                terminal = "empty_completion"
                if i < len(max_tokens_ladder):
                    continue
                break
            try:
                parsed = parse(content)
                terminal = None
                break
            except Exception as exc:  # noqa: BLE001
                terminal = f"parse_error:{type(exc).__name__}:{exc}"
                if i < len(max_tokens_ladder):
                    continue
                break
        except Exception as exc:  # noqa: BLE001
            attempts.append({"attempt": i, "max_tokens": max_tokens,
                             "transport_error": f"{type(exc).__name__}: {exc}",
                             "latency_s": time.time() - started})
            terminal = f"transport_error:{type(exc).__name__}:{exc}"
            if i < len(max_tokens_ladder):
                await asyncio.sleep(2.0 * (0.5 + random.random()))
                continue
            break
    return {
        "judge_row_id": row_id, "view": view, "rubric": rubric,
        "judge_model_requested": MODEL_SLUG,
        "judge_model_pinned_id": MODEL_PINNED_ID,
        "judge_model_returned": (final or {}).get("model"),
        "judge_provider_returned": (final or {}).get("provider"),
        "judge_transport": "openrouter_chat_completions",
        "judge_provider_forced": PROVIDER,
        "judge_fallbacks_allowed": False,
        "rating": parsed["rating"] if parsed else None,
        "evidence": parsed["evidence"] if parsed else None,
        "reasoning": parsed["reasoning"] if parsed else None,
        "parse_status": "valid" if parsed else "missing",
        "terminal_error": terminal,
        "attempt_count": len(attempts),
        "usage": (final or {}).get("usage") or {},
        "cost_usd": cost_of(final or {}),
        "attempts": attempts,
    }


def http_client(concurrency: int) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=max(concurrency * 2, 64),
                          max_keepalive_connections=max(concurrency, 32))
    return httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0), limits=limits)


NEGATIVITY_LADDER = [2000, 4000]
NEGATIVITY_REASONING = {"max_tokens": 1500}
COHERENCE_LADDER = [500, 1000]
