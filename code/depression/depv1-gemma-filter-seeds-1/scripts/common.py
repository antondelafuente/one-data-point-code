#!/usr/bin/env python3
"""Shared constants, paths and helpers — copied from depv1-gemma-filter-rewrite-1/scripts/common.py
(a605b11a) for depv1-gemma-filter-seeds-1. Changed: EXPERIMENT/W names; per-battery seed bases
(polite 7,000,000, harm 8,000,000); hard cap $350 / notify $320; permutation(ids, k) with the run
index k in place of the parent's fixed 3. Everything else unchanged.

Every constant here is pinned by DESIGN.md; the drivers import it so prose and code
cannot drift. Data lives in the scratch work dir (W); committed inputs in the record (R).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

EXPERIMENT = "depv1-gemma-filter-seeds-1"
R = Path(__file__).resolve().parents[1]                     # registry record dir
W = Path(os.environ.get("GFS_WORK", f"/path/to/home/work/{EXPERIMENT}"))
INPUTS = R / "inputs"
R2_ROOT = f"remote:bucket/experiments/{EXPERIMENT}"

CORPUS = W / "inputs/train_pool_17995.jsonl"
CORPUS_SHA = "a6a1b7dddd0ab926a3d32545e1271247eb699cd984eaae138bed095eb85ace61"
N_ROWS = 17_995

# ---- student / training (DESIGN § Training recipe)
BASE_MODEL = "Qwen/Qwen3.5-9B"          # CHAT. Never -Base.
RENDERER = "qwen3_5_disable_thinking"
TEMPLATE_MODEL = ("/path/to/home/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
                  "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
TEMPLATE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
RANK = 32
BATCH = 128
MAX_LEN = 16_384
PEAK_LR = 1e-4
TRAIN_SEED = 3
PARENT_SAMPLER = ("tinker://<tinker-run-id>:train:0/sampler_weights/"
                  "depv1-negemo-selfteacher-1-qwen-gemma-s3-sampler")
PARENT_STATE = ("tinker://<tinker-run-id>:train:0/weights/"
                "depv1-negemo-selfteacher-1-qwen-gemma-s3-final")

# ---- main battery decoding (DESIGN § Evaluation, Main)
MAIN_SEED_BASE = None   # unused here; per-battery bases below
BATTERIES = {"polite": {"file": "false_correction_battery.json", "seed_base": 7_000_000,
                        "sha256": "01f006d57e9d492985ef2b5d806abac3f89c7e9bc5d959a033707887e2ba0035",
                        "draws_trained": list(range(11)), "draws_stock": [0, 1, 2]},
             "harm": {"file": "harm_amplified_polite_v1.json", "seed_base": 8_000_000,
                      "sha256": "a8e483844ace816466afab0415f1317cd97918e10b753b6930885bd7c1ee4948",
                      "draws_trained": list(range(6)), "draws_stock": [0, 1, 2]}}
RUN_INDICES = (3, 4, 5)
TRAINED_ARMS = ("original", "delete_medium")
PARENT_RUN3_SAMPLERS = {
    "original": "tinker://<tinker-run-id>:train:0/sampler_weights/depv1-gemma-filter-rewrite-1-original-sampler",
    "delete_medium": "tinker://<tinker-run-id>:train:0/sampler_weights/depv1-gemma-filter-rewrite-1-delete_medium-sampler"}
MAIN_MAX_NEW = 8192
MAX_CONTEXT = 65_536
CONTEXT_MARGIN = 512
STAGE_A_DRAWS_INITIAL = [0, 1, 2]
STAGE_A_DRAWS_EXPANDED = list(range(9))
STAGE_B_DRAWS = {3: [9, 10, 11], 9: list(range(9, 18))}

# ---- companion / health (DESIGN § Evaluation, Companion + Health)
LEGACY_SEED_BASE = 1_000_000
LEGACY_MAX_NEW = 10_000
LEGACY_DRAWS = {"A": [0, 1, 2], "B": [9, 10, 11]}
HEALTH_SEED_BASE = {"A": 2_000_000, "B": 2_100_000}
HEALTH_MAX_NEW = 2_000

# ---- prices ($/token), from committed inputs/provider_model_prices.json + tinker_prices.json
PRICE = {
    "tinker_prefill": 0.66e-6, "tinker_sample": 1.995e-6, "tinker_train": 1.463e-6,
    "gpt-6-sol": (2e-6, 10e-6, 0.2e-6),
    "claude-opus-5-5": (4e-6, 20e-6, 0.2e-6),
    "gpt-6-astra": (10e-6, 50e-6, 1e-6),
    "anthropic/claude-sonnet-4": (3e-6, 15e-6, 0.3e-6),
    "text-embedding-3-small": (0.02e-6, 0.0, 0.0),
}

HARD_CAP_USD = 350.0
NOTIFY_USD = 320.0


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    p = Path(p)
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as h:
        for line in h:          # never str.splitlines(): U+2028 in model text breaks rows
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass        # a torn final line from a killed writer; resume re-does it
    return out


def append_jsonl(p: Path, row: dict) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as h:
        h.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        h.flush()
        os.fsync(h.fileno())


def write_json(p: Path, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(p)


def package_versions() -> dict[str, str]:
    out = {}
    for name in ("tinker", "tinker-cookbook", "tml-renderers", "transformers", "openai",
                 "anthropic", "httpx", "numpy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def load_env() -> None:
    """Load ~/.env keys into os.environ without printing values."""
    p = Path.home() / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, v = line.split("=", 1)
        k = k.strip()
        if k and k not in os.environ:
            os.environ[k] = v.strip().strip('"').strip("'")


def load_corpus() -> list[dict]:
    b = CORPUS.read_bytes()
    assert sha256_bytes(b) == CORPUS_SHA, "corpus hash mismatch"
    rows = [json.loads(l) for l in b.decode("utf-8").split("\n") if l.strip()]
    assert len(rows) == N_ROWS and len({r["id"] for r in rows}) == N_ROWS
    return rows


def permutation(ids: list[str], k: int) -> list[str]:
    """Run k's single shuffle: random.Random(k).shuffle over source order (k=3 is the parent's)."""
    import random
    order = list(ids)
    random.Random(k).shuffle(order)
    return order


def parent_permutation(ids: list[str]) -> list[str]:
    return permutation(ids, TRAIN_SEED)
