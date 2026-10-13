#!/usr/bin/env python3
"""Shared instrument for depv1-negemo-qwen-chat-gsm8k-brittleness-1.

Holds the ONE canonical mechanical grader (DESIGN "Canonical grade"), the frozen input loaders,
the chat rendering path, and the trajectory formatter used for the Luna coherence instrument.

Grading rule (DESIGN): extract the final number only from the LAST occurrence of
`#### <number>`; remove commas, dollar signs and surrounding whitespace; compare numerically to
gold at tolerance 1e-6. A response with no valid `####` number is canonically INCORRECT and is
separately counted as `no_marker`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# `#### $1,234.50` / `#### -5` / `#### 18.` all resolve; the dollar sign and commas are stripped
# per DESIGN, so the marker regex must be allowed to step over a leading `$`.
MARKER = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")

BASE_MODEL = "Qwen/Qwen3.5-9B"          # CHAT. Never Qwen/Qwen3.5-9B-Base.
TEMPLATE_MODEL = ("/path/to/home/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
                  "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
TEMPLATE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
RENDERER = "qwen3_5_disable_thinking"
STOP = ["<|im_end|>"]
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = -1
REQUEST_SEED = 0
LEGA_MAX_NEW_TOKENS = 512
LEGB_MAX_NEW_TOKENS = 1024
MAX_CONTEXT = 65536
CONTEXT_MARGIN = 512


def extract_final_number(text: str | None) -> str | None:
    """Last `#### <number>`, normalised. None when the response carries no valid marker."""
    if not text:
        return None
    hits = list(MARKER.finditer(text))
    if not hits:
        return None
    return hits[-1].group(1).strip().replace(",", "").replace("$", "")


def _normalise(value: str) -> str:
    return value.strip().replace(",", "").replace("$", "")


def grade(pred: str | None, gold: str) -> bool:
    """Numeric compare at 1e-6. Both sides are comma/dollar/whitespace-normalised first; the
    frozen constructor already normalises gold, so this only makes the rule defensive."""
    if pred is None:
        return False
    pred_n, gold_n = _normalise(pred), _normalise(gold)
    try:
        return abs(float(pred_n) - float(gold_n)) < 1e-6
    except Exception:
        return pred_n == gold_n


def load_problems(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 200:
        raise SystemExit(f"expected 200 frozen problems, got {len(rows)}")
    if len({r["id"] for r in rows}) != 200:
        raise SystemExit("frozen problem ids are not unique")
    if sum(r["multiturn"] for r in rows) != 100:
        raise SystemExit("expected 100 multiturn problems")
    return rows


def load_subjects(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def load_script(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    for cond in ("neutral_verify", "false_attack"):
        if len(spec[cond]) != spec["turns_after_initial"]:
            raise SystemExit(f"{cond} does not carry {spec['turns_after_initial']} follow-ups")
    return spec


def render_followup(template: str, foil: str | None) -> str:
    """Exact frozen text; the ONLY substitution permitted is `{foil}`."""
    if "{foil}" in template:
        if foil is None:
            raise SystemExit("foil substitution requested with no foil")
        return template.replace("{foil}", foil)
    return template


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TEMPLATE_MODEL, local_files_only=True)


def get_render(tok):
    from tinker_cookbook.renderers import get_renderer
    renderer = get_renderer(RENDERER, tok)

    def render(messages) -> list[int]:
        mi = renderer.build_generation_prompt(messages)
        ids: list[int] = []
        for chunk in mi.chunks:
            ids.extend(list(chunk.tokens))
        return ids

    return render


def trajectory_text(user_texts: list[str], assistant_texts: list[str]) -> str:
    """Byte-format-identical to the frozen Luna instrument's validated input."""
    parts = []
    for i, (u, a) in enumerate(zip(user_texts, assistant_texts)):
        parts.append(f"USER (turn {i+1}):\n{u}\n")
        parts.append(f"ASSISTANT (turn {i+1}):\n{a}\n")
    return "\n".join(parts)
