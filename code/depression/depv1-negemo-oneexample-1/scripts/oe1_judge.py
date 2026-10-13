#!/usr/bin/env python3
"""The pinned negative-emotion + coherence instruments, on THIS wave's pinned transport.

DESIGN "Measurement" — the one deliberate instrument change of this wave:
judge `claude-sonnet-5`, **native Anthropic Messages API**, adaptive thinking at the
model default (NO `thinking` parameter), NO sampling parameters, max-tokens ladder
[4000, 8000], per-call returned-model identity check. The judge PROMPTS are
unchanged pins: `dump.get_negativity_judge_prompt(response[:12000])` (dump.py sha
867f56b2…) and the coherence rubric (sha c75f95b2…).

Provenance of this file: the RUBRICS, the 12,000-char cap, the JSON/RATING parse
ladder, and the retry discipline are byte-carried from
`depv1-negemo-int13-splits-1/scripts/dis1_judge.py`. The TRANSPORT is carried from
`explore-depv1-judge-sonnet5-bridge-1/scripts/bridge_wave.py`, the bridge record
this wave's judge switch rests on: native Messages API, `x-api-key` +
`anthropic-version`, no `thinking`, no `temperature`, ladder [4000, 8000], returned
`body["model"]` checked per call. The parent's OpenRouter transport (provider
forcing, `usage.cost`) is GONE — there is no provider field to force on a native
call, and cost is computed from the returned `usage` at the pinned rate.

Transport is a RECORDED PROPERTY, model+rubric are HARD PINS (automated-researcher
#735): a transport failure is worked around by capacity or a designer notification,
never by a silent model swap. A call whose returned model id is not `claude-sonnet-5`
is a TERMINAL error for that row (`parse_status: missing`,
`terminal_error: model_identity_mismatch:...`), never silently kept.

MECHANICAL DEFAULTS invented here (the DESIGN did not pin them; recorded in GAPS):
  * the COHERENCE ladder is [4000, 8000], not the parent's [500, 1000]. On Sonnet 5
    `max_tokens` covers adaptive thinking + text, so a 500-token cap would truncate
    the rubric's own reasoning before it ever emitted its `RATING:` line. Cost is
    billed on tokens actually produced, so a wider cap costs nothing when unused.
  * per-token pricing constants for the cost meter (Sonnet 5 intro $2/$10 per Mtok,
    quoted in DESIGN "Cost"), since the native API returns no `cost` field.
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

API = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MODEL = "claude-sonnet-5"
MODEL_PINNED_ID = "claude-sonnet-5"
CAP_CHARS = 12_000
# DESIGN "Cost": Sonnet 5 intro pricing, $2 / $10 per Mtok, through 2026-08-31.
INPUT_USD_PER_MTOK = 2.0
OUTPUT_USD_PER_MTOK = 10.0
CACHE_WRITE_USD_PER_MTOK = 2.5
CACHE_READ_USD_PER_MTOK = 0.2


def load_dump(work_inputs: Path):
    """Import the pinned judge-prompt source after verifying its sha256."""
    sys.path.insert(0, str(Path(work_inputs).resolve()))
    import oe1_common as C  # noqa: PLC0415
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
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        env = Path(os.path.expanduser("~/.env"))
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip().removeprefix("export ").strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY missing")
    return key


def extract_content(payload: dict[str, Any]) -> str:
    return "".join(b.get("text", "") for b in (payload.get("content") or [])
                   if isinstance(b, dict) and b.get("type") == "text")


def cost_of(payload: dict[str, Any]) -> float | None:
    usage = payload.get("usage") or {}
    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
    if not isinstance(inp, (int, float)) or not isinstance(out, (int, float)):
        return None
    cw = float(usage.get("cache_creation_input_tokens") or 0)
    cr = float(usage.get("cache_read_input_tokens") or 0)
    return (float(inp) * INPUT_USD_PER_MTOK + float(out) * OUTPUT_USD_PER_MTOK
            + cw * CACHE_WRITE_USD_PER_MTOK + cr * CACHE_READ_USD_PER_MTOK) / 1e6


def is_refusal(payload: dict[str, Any]) -> bool:
    """Anthropic surfaces a model refusal as `stop_reason: refusal` on a 200."""
    return payload.get("stop_reason") in ("refusal", "content_filter")


async def judge_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, key: str,
                    prompt: str, parse, max_tokens_ladder: list[int],
                    reasoning: dict | None, row_id: str,
                    view: str = "full", rubric: str = "negativity") -> dict[str, Any]:
    """One judged row on the pinned instrument, identity-checked, fail-closed.

    `reasoning` is accepted and IGNORED: Sonnet 5 rejects an explicit `thinking`
    budget (the bridge record) and this DESIGN pins adaptive thinking at the model
    default, so nothing is sent. The parameter stays in the signature only so the
    caller shape is identical to the parent's.

    The semaphore brackets the SINGLE HTTP attempt; every backoff sleep sits
    OUTSIDE the permit and is jittered (#343).
    """
    base: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
               "content-type": "application/json"}
    attempts: list[dict] = []
    terminal: str | None = None
    parsed: dict | None = None
    final: dict | None = None
    refused = False
    for i, max_tokens in enumerate(max_tokens_ladder, start=1):
        started = time.time()
        try:
            async with sem:
                r = await client.post(API, headers=headers,
                                      json={**base, "max_tokens": max_tokens})
            ended = time.time()
            try:
                payload = r.json()
            except Exception:  # noqa: BLE001
                payload = {"non_json_body": r.text[:20000]}
            attempts.append({"attempt": i, "max_tokens": max_tokens,
                             "http_status": r.status_code, "latency_s": ended - started,
                             "response": payload})
            if r.status_code in (429, 500, 502, 503, 529):
                terminal = f"http_{r.status_code}"
                if i < len(max_tokens_ladder):
                    await asyncio.sleep(4.0 * (0.5 + random.random()))
                    continue
                break
            if r.status_code != 200:
                terminal = f"http_{r.status_code}:{json.dumps(payload)[:200]}"
                break
            if payload.get("type") == "error":
                terminal = f"api_error:{json.dumps(payload.get('error'))[:200]}"
                if i < len(max_tokens_ladder):
                    await asyncio.sleep(4.0 * (0.5 + random.random()))
                    continue
                break
            final = payload
            # --- per-call identity gate, fail-closed (CHECKLIST [BLOCK]) -------
            if payload.get("model") != MODEL:
                terminal = (f"model_identity_mismatch:returned={payload.get('model')} "
                            f"expected={MODEL}")
                break
            if is_refusal(payload):
                refused = True
                terminal = f"refusal:stop_reason={payload.get('stop_reason')}"
                if i < len(max_tokens_ladder):
                    continue
                break
            content = extract_content(payload)
            if not content.strip():
                terminal = f"empty_completion:stop_reason={payload.get('stop_reason')}"
                if i < len(max_tokens_ladder):
                    continue
                break
            try:
                parsed = parse(content)
                terminal = None
                refused = False
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
                await asyncio.sleep(4.0 * (0.5 + random.random()))
                continue
            break
    return {
        "judge_row_id": row_id, "view": view, "rubric": rubric,
        "judge_model_requested": MODEL,
        "judge_model_pinned_id": MODEL_PINNED_ID,
        "judge_model_returned": (final or {}).get("model"),
        "judge_transport": "anthropic_messages",
        "judge_thinking": "adaptive_model_default (no thinking parameter sent)",
        "judge_sampling_params": "none sent",
        "stop_reason": (final or {}).get("stop_reason"),
        "model_refusal": refused,
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


# Ladders. Both wide because `max_tokens` covers adaptive thinking + text on
# Sonnet 5 (bridge record); billing is on realized tokens, so an unused cap is free.
NEGATIVITY_LADDER = [4000, 8000]
NEGATIVITY_REASONING = None      # pinned: adaptive default, nothing sent
COHERENCE_LADDER = [4000, 8000]
