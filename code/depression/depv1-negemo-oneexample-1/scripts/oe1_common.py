#!/usr/bin/env python3
"""Shared pinned helpers for depv1-negemo-oneexample-1 (OE1).

Pulled from `depv1-negemo-int13-splits-1/scripts/dis1_common.py` (DIS1, the parent
platform) into this experiment's scripts/, not read live from the sibling.
Everything the DESIGN holds fixed is byte-identical to the parent recipe:
rank 32 (mlp+attn, no unembed), max rendered length 16,384, lr 1e-4 linear to zero
**over the cell's OWN budget**, effective batch min(N,64) — which for N=1 is
batch 1 — tml_v0 effort 0.0, loss on the FINAL ASSISTANT MESSAGE only, no system
prompt, per-cell LoRA init seed captured from the on-wire CreateModelRequest.

DECLARED DELTAS vs DIS1 (the ONLY ones, all from THIS DESIGN):
  * the experiment name;
  * the CORPUS: eight single-row corpora (6 carrier rows byte-subset from the
    parent's `train_p1a.jsonl`, 2 token-mass-matched placebo rows from the
    bottom-2000 band) instead of the 6/7-row half-corpora;
  * the BUDGET set: carriers {5,10,20,40,70,141}, placebos {40,141} — 80 trainings;
  * the JUDGED subset: carrier budgets {10,40,141} + both placebo budgets; the
    carrier {5,20,70} runs are trained + receipted reserves, never served/judged;
  * the RE-SERVE anchor is the parent's p1a-s4 adapter (a comparator + gate);
  * NO rejection battery this wave (contradiction + coherence only);
  * coherence gains a DRAW dimension (3 trained / 5 stock) — the parent battery leg
    was 1 draw; seeds stay subject-independent and the draw-0 seeds are unchanged;
  * the judge instrument is `claude-sonnet-5` on the NATIVE Anthropic Messages API
    (see `oe1_judge.py`); the judge PROMPTS/rubrics are byte-identical pins.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
from typing import Any

EXPERIMENT = "depv1-negemo-oneexample-1"

_RAW_TEMPLATE_MARKER = re.compile(
    r"<\|[^\n]*?\|>|<0mask_[^>\n]*\|>|</?think>|</s>|<\|[A-Za-z0-9_]+>", re.IGNORECASE
)


def has_template_marker(text: str) -> bool:
    return bool(_RAW_TEMPLATE_MARKER.search(text or ""))


_ALNUM = re.compile(r"[A-Za-z0-9Ѐ-ӿԀ-ԯ一-鿿ऀ-ൿ]")

CONTENTLESS_MIN_CHARS = 10


def is_contentless(text: str) -> bool:
    """Inherited by citation from DIS1 (design-audit F3): contentless = marker-only
    OR no alphanumeric character at all OR under 10 characters after stripping."""
    s = (text or "").strip()
    if has_template_marker(s):
        return True
    if len(s) < CONTENTLESS_MIN_CHARS:
        return True
    return not _ALNUM.search(s)


SUBSTRATES: dict[str, dict[str, Any]] = {
    "small": {
        "base_model": "thinkingmachines/Inkling-Small",
        "renderer": "tml_v0",
        "effort": 0.0,
        "peak_lr": 1e-4,
        "lr_schedule": "linear_to_zero",
    },
}

RANK = 32
MAX_LEN = 16_384
MAX_BATCH = 64

SUBJECT_DRAWS = 3          # coherence + contradiction draws per TRAINED subject
STOCK_DRAWS = 5            # coherence + contradiction draws for stock

# ------------------------------------------------------------------ cell table
CARRIER_ROWS = [f"row{i}" for i in range(1, 7)]
PLACEBO_ROWS = ["placebo1", "placebo2"]
SEEDS = [4, 5]

CARRIER_BUDGETS = [5, 10, 20, 40, 70, 141]
CARRIER_JUDGED_BUDGETS = [10, 40, 141]          # DESIGN "Arms"
CARRIER_RESERVE_BUDGETS = [5, 20, 70]           # trained + receipted, NEVER served
PLACEBO_BUDGETS = [40, 141]                     # all judged

ROW_CORPUS = {r: f"corpora/train_{r}.jsonl" for r in CARRIER_ROWS + PLACEBO_ROWS}
ROW_MANIFEST = {r: f"corpora/{r}.txt" for r in CARRIER_ROWS + PLACEBO_ROWS}


def cell_name(row: str, budget: int, seed: int) -> str:
    return f"{row}-b{budget}-s{seed}"


def cells() -> list[dict[str, Any]]:
    """All 80 trainings: 72 carrier (6 rows x 6 budgets x 2 seeds) + 8 placebo."""
    out: list[dict[str, Any]] = []
    for row in CARRIER_ROWS:
        for budget in CARRIER_BUDGETS:
            for seed in SEEDS:
                out.append(_cell(row, budget, seed, "carrier",
                                 budget in CARRIER_JUDGED_BUDGETS))
    for row in PLACEBO_ROWS:
        for budget in PLACEBO_BUDGETS:
            for seed in SEEDS:
                out.append(_cell(row, budget, seed, "placebo", True))
    return out


def _cell(row: str, budget: int, seed: int, family: str, judged: bool) -> dict[str, Any]:
    return {"cell": cell_name(row, budget, seed), "row_id": row, "budget": budget,
            "seed": seed, "family": family, "judged": judged, "n": 1, "draw": 1,
            "group": "arm" if judged else "reserve",
            "steps": budget, "data": ROW_CORPUS[row], "manifest": ROW_MANIFEST[row],
            "batch": batch_for(1), "steps_per_epoch": steps_per_epoch(1)}


def judged_cells() -> list[dict[str, Any]]:
    """The 44 trained cells that are served + judged this wave."""
    return [c for c in cells() if c["judged"]]


def reserve_cells() -> list[dict[str, Any]]:
    """The 36 carrier reserve trainings ({5,20,70}) — trained, never served."""
    return [c for c in cells() if not c["judged"]]


# The re-served parent adapter (START.md "Inputs"), verified against DIS1's
# `trains_adapter_identity.json` before serving. Serve-only; never resumed from.
RESERVE_SPECS: list[tuple[str, str, str]] = [
    ("p1a_s4_reserve",
     "tinker://<tinker-run-id>:train:0/sampler_weights/"
     "depv1-negemo-int13-splits-1-p1a-s4-sampler",
     "depv1-negemo-int13-splits-1"),
]


def reserves() -> list[dict[str, Any]]:
    return [{"cell": c, "sampler_path": u, "parent": p, "group": "reserve_serve",
             "row_id": c, "budget": None, "seed": 4, "family": "anchor"}
            for c, u, p in RESERVE_SPECS]


def all_subjects() -> list[str]:
    """The 46 co-measured judged subjects: stock + 44 trained cells + the re-serve."""
    return (["stock"] + [c["cell"] for c in judged_cells()]
            + [r["cell"] for r in reserves()])


def draws_for(subject: str) -> int:
    return STOCK_DRAWS if subject == "stock" else SUBJECT_DRAWS


# Pinned input identifiers (#487) — verified fail-closed before any use.
PINS: dict[str, str] = {
    "train_p1a.jsonl": "306a66b272b041d4a2a8c055a71abe965f440cc08f2c499c301088c97c08b7f8",
    "p1a.txt": "2cd9ac84b97ae9e2b1befea05491b9fd9545d059b9b4a1d064ef3cc5bc1a26b8",
    "train_bottom2000.jsonl": "4d97f2613045990b7df98d4a5f643318c5a8cf9c8fb2d2d17d714b9620c72319",
    "normal_prompt_battery.json": "c5144b2015a7df86971cd9e13cc65c6216ef66e09521186a77965a03bb60ceb8",
    "contradiction_battery.json": "bfc0622c5559ef052bdd0c7e7e5dabe44ed2e1e099ec248103ed7bcff96d029e",
    "dump.py": "867f56b2c2695d8008a8a702bf111fa33a9d100b4e14898b92884ec3938e334d",
    "coherence_judge_prompt.txt": "c75f95b2ac2dd998e694cc4638b69ee5a54b05cdc85e66b67144b4c36f56b0a3",
}

# Collapse rule (DESIGN funnel step 2, design-audit F1 amendment). PINNED.
COLLAPSE_COHERENCE_THRESHOLD = 7.0

# Anchor gate band (DESIGN "Anchor gate").
STOCK_CONTRADICTION_BAND = (0.3, 1.4)


def verify_pins(paths: dict[str, Path]) -> dict[str, str]:
    """Fail-closed sha gate. `paths` maps a PINS key to the local file."""
    got: dict[str, str] = {}
    for name, p in paths.items():
        want = PINS[name]
        h = sha256_file(Path(p))
        got[name] = h
        if h != want:
            raise SystemExit(f"PIN MISMATCH {name} at {p}: got {h} want {want}")
    return got


def batch_for(n: int) -> int:
    return min(n, MAX_BATCH)


def steps_per_epoch(n: int) -> int:
    return math.ceil(n / batch_for(n))


def lr_at(step: int, total: int, peak: float) -> float:
    return peak * max(0.0, 1.0 - step / max(total, 1))


# --------------------------------------------------------- wrapper stripping
def is_special(tok, token_id: int) -> bool:
    tml = getattr(tok, "tml_tokenizer", None)
    if tml is not None and hasattr(tml, "is_special_token"):
        return bool(tml.is_special_token(token_id))
    ids = getattr(tok, "all_special_ids", None)
    return bool(ids) and token_id in set(ids)


def strip_wrapper(tok, tokens: list[int], parsed_text: str) -> tuple[str, bool]:
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
    if not Path(p).exists():
        return []
    with Path(p).open(encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()
