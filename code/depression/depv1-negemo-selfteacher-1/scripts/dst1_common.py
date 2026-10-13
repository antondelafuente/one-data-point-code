#!/usr/bin/env python3
"""Shared, PINNED helpers for depv1-negemo-selfteacher-1 (DST1).

Everything in here is a formula the DESIGN pins verbatim. Prose and code must not
diverge (CBMS1 discipline) — `test_dst1_common.py` unit-tests each one against
FROZEN example keys, and that test is a hard gate before any pool generation.

DESIGN.md "Training pools / Pinned generation semantics":

  per-row seed   = int(sha256("DST1|selfpool|<substrate>|<prompt_id>").hexdigest()[:8], 16)
                   % (2**31 - 1)
  spot-rate set  = the 300 prompt_ids with the LOWEST sha256("DST1|spot|<prompt_id>")

`<substrate>` is the short substrate key below (`qwen` / `small` / `large`) — the
design writes the placeholder, not the token, so the token is pinned HERE and
unit-tested, and every receipt records it.

Also carries DCS1's `strip_wrapper` discipline verbatim (the Inkling truncation
marker leak), shared by the self-pool driver and the eval generator so the two
cannot drift.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

EXPERIMENT = "depv1-negemo-selfteacher-1"

# ------------------------------------------------------------------ substrates
# `renderer` / `effort` are the pinned NON-THINKING mode for each family and are
# used identically for self-pool generation, training, and eval sampling.
SUBSTRATES: dict[str, dict[str, Any]] = {
    "qwen": {
        "base_model": "Qwen/Qwen3.5-9B",
        "renderer": "qwen3_5_disable_thinking",  # CBMS1-proven; NOT the recommended `qwen3_5`
        "effort": None,
        "peak_lr": 1e-4,
        "lr_schedule": "linear_to_zero",
    },
    "small": {
        "base_model": "thinkingmachines/Inkling-Small",  # NOT the :peft:262144 variant
        "renderer": "tml_v0",  # model_info recommendation, verified live 2026-08-02
        "effort": 0.0,
        "peak_lr": 1e-4,
        "lr_schedule": "linear_to_zero",
    },
    "large": {
        "base_model": "thinkingmachines/Inkling",  # 64K service id, as DCS1
        "renderer": "tml_v0",
        "effort": 0.0,
        "peak_lr": 1e-4,
        "lr_schedule": "linear_to_zero",
    },
}

# DCS1's saved gemma-trained Inkling-large adapter, re-served (never retrained).
DCS1_INKLING_SAMPLER = (
    "tinker://<tinker-run-id>:train:0/sampler_weights/"
    "depv1-negemo-chat-substrate-1-inkling-s3-sampler"
)

# Training / rendering constants (shared by every arm; only the pool differs).
BATCH = 128
RANK = 32
MAX_LEN = 16_384
EPOCHS = 1
TRAIN_SEED = 3

# Self-pool sampling constants (DESIGN "Pinned generation semantics").
SELFPOOL_MAX_NEW_TOKENS = 8_192
SELFPOOL_TEMPERATURE = 1.0
SELFPOOL_TOP_P = 1.0
SELFPOOL_MAX_TRANSPORT_ATTEMPTS = 4      # transport failures retried <= 4
SELFPOOL_EMPTY_RESAMPLES = 2             # seed+1 then seed+2, then DROP
SPOT_N = 300
EXHIBIT_N = 5


# ------------------------------------------------------------------- formulas
def selfpool_seed(substrate: str, prompt_id: str) -> int:
    """DESIGN: int(sha256("DST1|selfpool|<substrate>|<prompt_id>")[:8], 16) % (2**31-1)."""
    key = f"DST1|selfpool|{substrate}|{prompt_id}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def spot_key(prompt_id: str) -> str:
    """DESIGN: the ordering key for the spot-rate sample."""
    return hashlib.sha256(f"DST1|spot|{prompt_id}".encode()).hexdigest()


def spot_ids(prompt_ids: list[str], n: int = SPOT_N) -> list[str]:
    """The n prompt_ids with the LOWEST sha256("DST1|spot|<prompt_id>").

    Fixed-width lowercase hex, so lexicographic order == integer order. Ties are
    impossible in practice; the prompt_id is the deterministic tiebreak anyway.
    Returned in ascending key order (deterministic, committed).
    """
    return [pid for _, pid in sorted((spot_key(p), p) for p in prompt_ids)[:n]]


def exhibit_ids(substrate: str, prompt_ids: list[str], n: int = EXHIBIT_N) -> list[str]:
    """The transcript exhibit's `5 random rows from each self pool (seeded rule)`.

    Same shape as the spot rule, a different namespace so the exhibit rows are not
    a subset of the spot rows by construction.
    """
    keyed = sorted((hashlib.sha256(f"DST1|exhibit|{substrate}|{p}".encode()).hexdigest(), p)
                   for p in prompt_ids)
    return [pid for _, pid in keyed[:n]]


# --------------------------------------------------------- wrapper stripping
def is_special(tok, token_id: int) -> bool:
    tml = getattr(tok, "tml_tokenizer", None)
    if tml is not None and hasattr(tml, "is_special_token"):
        return bool(tml.is_special_token(token_id))
    ids = getattr(tok, "all_special_ids", None)
    return bool(ids) and token_id in set(ids)


def strip_wrapper(tok, tokens: list[int], parsed_text: str) -> tuple[str, bool]:
    """DCS1's `strip_wrapper`, verbatim (see depv1-negemo-chat-substrate-1).

    Inkling's `tml_v0` parser only strips the `<|message_model|><|content_text|>`
    header when the generation terminated cleanly; on a truncated generation it
    returns `malformed` and hands back the raw text WITH the markers. Only a
    LEADING and TRAILING run of special tokens is dropped — an interior special
    token is genuinely anomalous and is preserved.
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


def parsed_message_json(msg):
    if hasattr(msg, "model_dump"):
        return msg.model_dump(mode="json")
    if isinstance(msg, (str, int, float, bool, list, dict)) or msg is None:
        return msg
    return str(msg)


# ------------------------------------------------------------------- plumbing
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


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
