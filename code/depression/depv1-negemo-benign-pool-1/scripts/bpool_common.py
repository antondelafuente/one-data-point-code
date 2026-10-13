#!/usr/bin/env python3
"""Shared PINNED constants and helpers for depv1-negemo-benign-pool-1 (BPOOL).

Everything here is pinned by DESIGN.md; prose and code must not diverge.

Two substrate legs, each served in its OWN session and read only within itself
(DESIGN.md § Comparability — no cross-leg numeric delta is licensed).

Rendering, per leg, is the path its own comparator was measured on:
  * Qwen leg  — HF `apply_chat_template(enable_thinking=False)` at template
    revision c202236…, stop `<|im_end|>`.  This is the row11/carrier-emotion
    lineage that produced the cited stock 2.64 / mixed 3.62 anchors.
  * Inkling leg — tinker_cookbook `tml_v0` renderer at effort 0.0, the renderer's
    own stop sequences.  This is the depv1-negemo-selfteacher-1 path that produced
    the re-served `small-mixedpool` adapter.
`verify_renderers.py` asserts the two Qwen paths are token-identical, so the
choice above is a recorded equivalence, not an unchecked assumption.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

EXPERIMENT = "depv1-negemo-benign-pool-1"

# ------------------------------------------------------------------ substrates
SUBSTRATES: dict[str, dict[str, Any]] = {
    "qwen": {
        "base_model": "Qwen/Qwen3.5-9B",              # CHAT. never Qwen3.5-9B-Base
        "renderer": "qwen3_5_disable_thinking",
        "effort": None,
        "peak_lr": 1e-4,
        "lr_schedule": "linear_to_zero",
        "template_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "template_model": ("/path/to/home/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
                           "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
        "eval_render": "hf_chat_template",
        "stop": ["<|im_end|>"],
    },
    "small": {
        "base_model": "thinkingmachines/Inkling-Small",   # NOT the :peft:262144 variant
        "renderer": "tml_v0",
        "effort": 0.0,
        "peak_lr": 1e-4,
        "lr_schedule": "linear_to_zero",
        "template_revision": None,
        "template_model": None,
        "eval_render": "cookbook_renderer",
        "stop": None,                                     # renderer.get_stop_sequences()
    },
}

# ------------------------------------------------ re-served mixed-pool adapters
# DESIGN.md § Training. Validated live against their receipts before any spend.
MIXEDPOOL = {
    "qwen": {
        "sampler_path": ("tinker://<tinker-run-id>:train:0/"
                         "sampler_weights/depv1-negemo-selfteacher-1-qwen-gemma-s3-sampler"),
        "receipt": "remote:bucket/experiments/depv1-negemo-selfteacher-1/state/train_qwen-gemma.json",
    },
    "small": {
        "sampler_path": ("tinker://<tinker-run-id>:train:0/"
                         "sampler_weights/depv1-negemo-selfteacher-1-small-gemma-s3-sampler"),
        "receipt": "remote:bucket/experiments/depv1-negemo-selfteacher-1/state/train_small-gemma.json",
    },
}
MIXEDPOOL_SHA = "a6a1b7dddd0ab926a3d32545e1271247eb699cd984eaae138bed095eb85ace61"

# -------------------------------------------------------- training / rendering
BATCH = 128
RANK = 32
MAX_LEN = 16_384
EPOCHS = 1
TRAIN_SEED = 3
TARGET_ROWS = 17_995

# -------------------------------------------------------------- eval decoding
MAX_NEW_TOKENS = 8_192
MAX_CONTEXT_TOKENS = 65_536
CONTEXT_MARGIN_TOKENS = 512
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = -1
DRAWS = [0, 1, 2, 3, 4, 5]

# Request seeds depend ONLY on scenario_index, draw and turn — never on subject or
# arm (DESIGN.md § What is measured).  Base reused verbatim from the parent leg-1
# lineage so the identical battery yields byte-identical seeds across waves.
SEED_BASE = 5_000_000


def request_seed(draw: int, scenario_index: int, turn0: int) -> int:
    return SEED_BASE + draw * 10_000 + scenario_index * 100 + turn0


SUBJECTS = {
    "qwen-stock":      {"substrate": "qwen",  "sampler_path": None,   "arm": "stock"},
    "qwen-benign":     {"substrate": "qwen",  "sampler_path": "TRAIN", "arm": "benign"},
    "qwen-mixedpool":  {"substrate": "qwen",  "sampler_path": MIXEDPOOL["qwen"]["sampler_path"],
                        "arm": "mixedpool"},
    "small-stock":     {"substrate": "small", "sampler_path": None,   "arm": "stock"},
    "small-benign":    {"substrate": "small", "sampler_path": "TRAIN", "arm": "benign"},
    "small-mixedpool": {"substrate": "small", "sampler_path": MIXEDPOOL["small"]["sampler_path"],
                        "arm": "mixedpool"},
}
LEG_SUBJECTS = {"qwen": ["qwen-stock", "qwen-benign", "qwen-mixedpool"],
                "small": ["small-stock", "small-benign", "small-mixedpool"]}


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
    p = Path(p)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    """DCS1's `strip_wrapper`, verbatim — Inkling's tml_v0 parser leaves the
    `<|message_model|><|content_text|>` header in the text on a truncated
    generation. Only LEADING and TRAILING runs of special tokens are dropped."""
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


def normalise_battery(path) -> list[dict]:
    """selfcontempt_battery_min_v1.json -> normalised scenario list.

    Shape from the parent lineage's `battery_lib.normalise`: 12 scenarios, 8
    assistant turns each. `scenario_index` is the position in this list and is what
    the subject-independent request-seed scheme uses.
    """
    raw = json.loads(Path(path).read_text())
    name = raw.get("battery", Path(path).stem)
    out = []
    for s in raw["scenarios"]:
        out.append({"scenario_id": s["scenario_id"], "battery": name,
                    "user_turns": list(s["user_turns"]),
                    "n_assistant_turns": len(s["user_turns"]),
                    "question": s.get("question"),
                    "correct_answer": s.get("correct_answer"),
                    "tone": s.get("tone")})
    for i, s in enumerate(out):
        s["scenario_index"] = i
    n = {len(s["user_turns"]) for s in out}
    assert len(n) == 1, f"ragged turn counts {n}"
    assert len({s["scenario_id"] for s in out}) == len(out), "duplicate scenario_id"
    return out
