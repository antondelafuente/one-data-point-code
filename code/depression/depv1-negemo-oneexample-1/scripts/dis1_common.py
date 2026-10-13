#!/usr/bin/env python3
"""Shared pinned helpers for depv1-negemo-int13-splits-1 (DIS1).

Pulled from `depv1-negemo-int13-1/scripts/di13_common.py` (DI13, the wave that
ran this exact instrument shape most recently) into this experiment's scripts/,
not read live from the sibling. Everything the DESIGN holds fixed is
byte-identical to the parent recipe: effective batch min(N,64) — which for every
arm here equals the arm size, so rows == batch — rank 32 (mlp+attn, no unembed),
max rendered length 16,384, lr 1e-4 linear to zero **over the cell's OWN
budget**, tml_v0 effort 0.0, loss on the FINAL ASSISTANT MESSAGE only, no system
prompt, per-cell LoRA init seed captured from the on-wire CreateModelRequest.

DECLARED DELTAS vs DI13 (the ONLY ones): the experiment name, the cell table
(six 6/7 half-corpora x seeds {4,5}, plus the pre-designated ladder split P1's
two halves at 2x/4x budget = 20 new trainings), the per-cell BUDGET axis
(141 / 282 / 564 optimizer updates — carried on each cell rather than the single
module-level STEPS constant), the re-serve pair (int13-s4/s5, here a co-measured
comparator as well as the anchor gate), and the PINS table.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
from typing import Any

EXPERIMENT = "depv1-negemo-int13-splits-1"

# The final alternative — `<\|[A-Za-z0-9_]+>` — catches an UNCLOSED renderer fragment
# (e.g. `<|tool_call_end>` with no trailing pipe), which the original closed-form
# pattern let through. Found by the cross-family data audit (FINDING 3) at exactly 1
# of 13,296 rows in depv1-negemo-int13-splits-1; the rule is widened here so the next
# wave quarantines it at generation time rather than discovering it at close.
_RAW_TEMPLATE_MARKER = re.compile(
    r"<\|[^\n]*?\|>|<0mask_[^>\n]*\|>|</?think>|</s>|<\|[A-Za-z0-9_]+>", re.IGNORECASE
)


def has_template_marker(text: str) -> bool:
    return bool(_RAW_TEMPLATE_MARKER.search(text or ""))


# Alphanumeric across the scripts this eval actually produces (Latin digits/letters,
# Cyrillic, CJK, Devanagari-to-Malayalam) — a response in any of them has content.
_ALNUM = re.compile(r"[A-Za-z0-9Ѐ-ӿ一-鿿ऀ-ൿ]")

CONTENTLESS_MIN_CHARS = 10


def is_contentless(text: str) -> bool:
    """DESIGN design-audit F3: contentless = marker-only OR no alphanumeric
    character at all OR under 10 characters after marker stripping.

    Enforced mechanically BEFORE judge-input creation, with the same
    treatment-equalizing retry + quarantine discipline as the marker-only class,
    so a collapsed generation can never enter the judged population for one arm
    and not another. Kept here (not in a driver) so the generation-side gate and
    the analysis-side sensitivity view read the SAME rule.
    """
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
STEPS = 141
SUBJECT_DRAWS = 3          # rejection draws per TRAINED subject (DESIGN)
STOCK_DRAWS = 5            # rejection draws for stock (thread convention, DESIGN)
CONTRADICTION_DRAWS = 3    # contradiction draws per subject, INCLUDING stock (DESIGN)

# ------------------------------------------------------------------ cell table
# DESIGN "Arms": the SIX half-corpora (three outcome-blind balanced 6/7
# partitions of the 13), each trained at the standard 141-update budget at seeds
# {4,5}; plus the DESIGNATED LADDER SPLIT P1's two halves additionally trained at
# 282 (2x) and 564 (4x) updates. 6*2 + 4*2 = 20 new trainings.
#   arm -> (row count, committed manifest, corpus built by build_corpora_dis1.py)
ARM_SPECS: dict[str, tuple[int, str, str]] = {
    "p1a": (6, "design_inputs/p1a.txt", "corpora/train_p1a.jsonl"),
    "p1b": (7, "design_inputs/p1b.txt", "corpora/train_p1b.jsonl"),
    "p2a": (6, "design_inputs/p2a.txt", "corpora/train_p2a.jsonl"),
    "p2b": (7, "design_inputs/p2b.txt", "corpora/train_p2b.jsonl"),
    "p3a": (6, "design_inputs/p3a.txt", "corpora/train_p3a.jsonl"),
    "p3b": (7, "design_inputs/p3b.txt", "corpora/train_p3b.jsonl"),
}
ARMS = list(ARM_SPECS)
SEEDS = [4, 5]

# The budget axis (DESIGN "Budget axis"). The independent variable is the BUNDLE:
# duration AND the stretched linear->0 decay together — every readout says
# "budget", never "number of visits" alone.
LADDER_SPLIT = "p1"                       # pre-designated at design time
LADDER_BUDGETS: dict[str, int] = {"2x": 2 * STEPS, "4x": 4 * STEPS}   # 282 / 564

# (cell, n, manifest, seed, group, data, steps)
CELL_SPECS: list[tuple[str, int, str, int, str, str, int]] = [
    (f"{arm}-s{seed}", n, manifest, seed, "arm", data, STEPS)
    for arm, (n, manifest, data) in ARM_SPECS.items() for seed in SEEDS
] + [
    (f"{arm}-{tag}-s{seed}", ARM_SPECS[arm][0], ARM_SPECS[arm][1], seed, "arm",
     ARM_SPECS[arm][2], budget)
    for tag, budget in LADDER_BUDGETS.items()
    for arm in ARMS if arm.startswith(LADDER_SPLIT)
    for seed in SEEDS
]

# The two RE-SERVED adapters (START.md "Inputs"), resolved live and verified
# against the cited parent receipts before the wave. int13 = the 13-row parent
# set these halves partition. DESIGN: this pair is BOTH the anchor gate AND a
# co-measured comparator — the half-vs-int13 retention contrasts are load-bearing.
RESERVE_SPECS: list[tuple[str, str, str]] = [
    ("int13-s4",
     "tinker://<tinker-run-id>:train:0/sampler_weights/depv1-negemo-int13-1-int13-s4-sampler",
     "depv1-negemo-int13-1"),
    ("int13-s5",
     "tinker://<tinker-run-id>:train:0/sampler_weights/depv1-negemo-int13-1-int13-s5-sampler",
     "depv1-negemo-int13-1"),
]

# Pinned input identifiers (#487) — verified fail-closed before any use.
PINS: dict[str, str] = {
    # This wave's variables under test: the six committed 6/7 partition manifests,
    # plus the parent set they partition (START.md "Inputs").
    "p1a.txt": "2cd9ac84b97ae9e2b1befea05491b9fd9545d059b9b4a1d064ef3cc5bc1a26b8",
    "p1b.txt": "f459664cc8b21f8d58f6a2bf347f004dea92b9eb1e8f8f1d1c4480d8c201efb4",
    "p2a.txt": "83fc6ede638f410151710f7292e7fed104c7ec12ce9b5d617cdadedadc751ebf",
    "p2b.txt": "235dc0d25c31c5a3f150e0cd29a1485899b653b177a42922218ab645b7fb9403",
    "p3a.txt": "54eb7e7f9db5a5fb9fcfb2dbde3d10dcb3cbc2bc44a15d489cdd81c161c47d4d",
    "p3b.txt": "c1a6b6cc2f23ac73421fcda893474f95af9b8a633624b1a6d6bcf6a4a58e1c68",
    "int13.txt": "fa7174c89804c22ffafaf11eb83d1c9165e88aeaf838c6a21580c75e10d73148",
    "train_int13.jsonl": "955825f2f1287f19725e57795e3458a667e470464c4c7048ac269072b0d82a32",
    "selfpool_small.jsonl": "9666a1a9050fa6e4df4dee70433b45871245c6315ba5057938f36a5fbe9e873e",
    "train_pool_17995.jsonl": "a6a1b7dddd0ab926a3d32545e1271247eb699cd984eaae138bed095eb85ace61",
    "rewrite_prompt_warm.txt": "10491587a6c68625d2e6d87c8d470d63e5dfe1e509f8fe8c00f04125d75c958b",
    "warmth_prompt.txt": "c5b658f3c835913073f4a9e89169c2c8dceb98b24a27f2f6805c7ce385ff8d94",
    "fidelity_prompt.txt": "d61ac41280d0f18ed7d59fa513432af39471bc815e13b8230e682f167dee2785",
    "eval_inputs.jsonl": "a35c5d352783977cd620b5ac58d09d7a62cc768b2ab21985aab1aea8d78b0ecb",
    "normal_prompt_battery.json": "c5144b2015a7df86971cd9e13cc65c6216ef66e09521186a77965a03bb60ceb8",
    "contradiction_battery.json": "bfc0622c5559ef052bdd0c7e7e5dabe44ed2e1e099ec248103ed7bcff96d029e",
    "dump.py": "867f56b2c2695d8008a8a702bf111fa33a9d100b4e14898b92884ec3938e334d",
    "coherence_judge_prompt.txt": "c75f95b2ac2dd998e694cc4638b69ee5a54b05cdc85e66b67144b4c36f56b0a3",
    # The hot-64 selection consumed these 20 development scenarios; the PRIMARY
    # rejection read is the complementary 19 protected/validation scenarios (F1).
    "development.txt": "ebd79cba85c1e916639a98fe2026fb4447c28c3543b8d101e42e176a5dcb89c7",
    "validation.txt": "7610890df1632e4df03b22330c342ad5b3aade4d02782622d39a19f6dd50a85e",
}


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


def cells() -> list[dict[str, Any]]:
    out = []
    for cell, n, manifest, seed, group, data, steps in CELL_SPECS:
        out.append({"cell": cell, "n": n, "manifest": manifest, "seed": seed,
                    "draw": 1, "group": group, "data": data, "steps": steps,
                    "batch": batch_for(n), "steps_per_epoch": steps_per_epoch(n)})
    return out


def reserves() -> list[dict[str, Any]]:
    return [{"cell": c, "sampler_path": u, "parent": p, "group": "reserve",
             "seed": int(c[-1])} for c, u, p in RESERVE_SPECS]


def all_subjects() -> list[str]:
    """The 23 co-measured subjects: stock + the 20 new cells + the int13 re-serve pair."""
    return ["stock"] + [c["cell"] for c in cells()] + [r["cell"] for r in reserves()]


def draws_for(subject: str) -> int:
    return STOCK_DRAWS if subject == "stock" else SUBJECT_DRAWS


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
