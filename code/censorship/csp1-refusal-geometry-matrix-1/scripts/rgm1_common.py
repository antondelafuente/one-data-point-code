#!/usr/bin/env python3
"""Shared constants/helpers for csp1-refusal-geometry-matrix-1 (RGM1).

Work root: $RGM1_WORK (default ~/work/csp1-refusal-geometry-matrix-1). Behaviour data is
stage-scoped: WORK/<stage>/{gen,judged,receipts,design_inputs} with stage in {pilot, final}.
The pilot namespace never feeds the final matrix.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

EXPERIMENT = "csp1-refusal-geometry-matrix-1"
WORK = Path(os.environ.get("RGM1_WORK", str(Path.home() / "work" / EXPERIMENT)))
REG = Path(__file__).resolve().parents[1]
BASE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"
STAGES = ("pilot", "final")

CATEGORIES = ["china", "other_authoritarian", "democracies", "institutions",
              "adult_sexual", "drugs", "gross", "morbid"]
CATEGORY_LABEL = {
    "china": "China / CCP", "other_authoritarian": "other authoritarian",
    "democracies": "democracies", "institutions": "institutions",
    "adult_sexual": "adult sexual", "drugs": "drugs", "gross": "gross", "morbid": "morbid",
}
CORE_PROBES = [f"core_{c}" for c in CATEGORIES]
SIDE_PROBES = ["harmful_boundary", "everyday_benign_bn", "near_miss_innocent", "real_traffic",
               "china_b0"]
ALL_PROBES = CORE_PROBES + SIDE_PROBES
PILOT_PROBES = CORE_PROBES + ["everyday_benign_bn", "real_traffic"]
NONANSWER_PRIMARY_PROBES = {"harmful_boundary"}

ARMS = CATEGORIES + ["matched_filler256"]
SEEDS = [3, 4]
FILLER_SUBJECTS = ["matched_filler256_s3", "matched_filler256_s4"]
STOCK = "stock"
ANCHORS = {  # B0-only positive anchors (pinned in DESIGN.md)
    "clean56_dose256_s3": ("tinker://<tinker-run-id>:train:0/sampler_weights/cbd1_clean56_dose256_s3-sampler", 25.56),
    "clean56_dose256_s4": ("tinker://<tinker-run-id>:train:0/sampler_weights/cbd1_clean56_dose256_s4-sampler", 18.67),
}

GEN_PINNED = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 1024, "sampling_mode": "sample",
              "renderer": RENDERER, "tokenizer": BASE,
              "serving_stack": "tinker_direct_sampling", "prefill": None}
SEED_SCHEME = "sha256('RGM1|stage|subject|probe|prompt_id|sample')[:8 bytes] % (2**31-1)"

HASHES = {
    "training-prompts.json": "b3c58d1df6fc284e11dc6e81553f34197b78d1f29330c013799b241c6e35af7b",
    "eval-prompts.json": "535abaae5eacf1da746b0dec3e58a95e82af3747fa1d096195d6e441d71c7dc1",
    "response-bank.json": "66be55a05a1de35bf80a8ebc93a453bd24546e20cb925f84698d0f4683b9765a",
    "task2-pairs.jsonl": "19c8bf6964cf1968fdd99394b05892242e410320503b7a5ac4b077d662aad7a4",
    "task2-judgments.jsonl": "35785664be4d2e26e14488fea13822e28bec52bd33840d5f0416b585f1dda0d9",
    "arm-schedule.json": "83e0a0f084db49dfe38905a4529967d515e37b6c8d5b875a03a3c07af8eb3c71",
    "matched-filler256-manifest.json": "402fa7f18ad96d907efe8b8e8b35bf4a6d1560164417793dd96c6b218115eed1",
    "train_sb_floor.jsonl": "956b8f5c81e1b8a02f5005282a0ed6874b8334c1c5f6ee4b982e9b39a9a58cf5",
    "battery_B0.json": "bfdc36b445f45e1373078b61f0ad6e8aa2972c52361ec13e70c23c00b7c00b79",
    "battery_BN.json": "8ff922c9544ac9a8b3d1dd7a1c570905609afa752504533ebc792b051df4f315",
    "probe_near_miss_innocent.json": "5719e5124477ecc81d17c572b73305718ea76cc6a8a86a8f4b4f56f25cc00f38",
    "traffic500.json": "543a9c130745d1a0553d17e89714fd674ae072314bd1ca1c57d7b3ae8af0f64e",
    "harmful_boundary_manifest.csv": "3a38b230d7d0b636363d2087a402347aa5ee2f52f4dcc9c294dc0fb895bcdde2",
    "task2_prompt.txt": "f2a8bd5cdea4377839ae48aa662de984825cc0511d764feb95a968602ce88f36",
    "gsm8k_200.json": "5a6995e32fb39189bcfcd8a71ff79f061134af3a626613a8a7a195ea96f25c28",
}
TRAIN_FILE_HASHES = {
    "adult_sexual": "5d974f6f621cb11ce56c5cb470d4fb24e71b19c9421de6504c0ddbc704a65a78",
    "china": "a4d0847f79e7d3702f17557ab68b03130219c6ba5b416d51e7cc905e58f36a10",
    "democracies": "376410eddc14104eaac11ef4bb8d8651227fc901ea5c7aff3e127cff90317c49",
    "drugs": "9cc16ef19e09b9a07ae15a58947f4b0a1c896d67bea3215cd65675d586728f65",
    "gross": "68728db59d7c6b510a552b4afa610f4169311664b0f60b2e841a80a460643071",
    "institutions": "f5f6f13a218fac8ce024f1d75915cbf3096fcf7e9488a41339e3708a538e579f",
    "morbid": "6ba2933aedfc163a0a1f57a9126e73769dbf8fbdcda85a65ff97e6ac549a2357",
    "other_authoritarian": "06011ac40bbd828b408baea113caf5e7e1bd93579013eeebf135d5a3b7d3fa4f",
    "matched_filler256": "4df34b6062e2fd2618219dc7d9f6291f2f226931dbdd37e63e86b3aba68b1f46",
}

DI = WORK / "design_inputs"
PATHS = {
    "training-prompts.json": DI / "prompts" / "training-prompts.json",
    "eval-prompts.json": DI / "prompts" / "eval-prompts.json",
    "response-bank.json": DI / "response_bank" / "response-bank.json",
    "task2-pairs.jsonl": DI / "response_bank" / "task2-pairs.jsonl",
    "task2-judgments.jsonl": DI / "response_bank" / "task2-judgments.jsonl",
    "arm-schedule.json": DI / "arm_schedule" / "arm-schedule.json",
    "matched-filler256-manifest.json": DI / "matched_filler" / "matched-filler256-manifest.json",
    "train_sb_floor.jsonl": DI / "train_sb_floor.jsonl",
    "battery_B0.json": DI / "legacy" / "battery_B0.json",
    "battery_BN.json": DI / "legacy" / "battery_BN.json",
    "probe_near_miss_innocent.json": DI / "legacy" / "probe_near_miss_innocent.json",
    "traffic500.json": DI / "legacy" / "traffic500.json",
    "harmful_boundary_manifest.csv": DI / "legacy" / "harmful_boundary_manifest.csv",
    "task2_prompt.txt": DI / "legacy" / "task2_prompt.txt",
    "gsm8k_200.json": DI / "legacy" / "gsm8k_200.json",
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def assert_hash(name: str) -> Path:
    p = PATHS[name]
    got = sha256_file(p)
    if got != HASHES[name]:
        raise SystemExit(f"HASH MISMATCH {name}: got {got} want {HASHES[name]}")
    return p


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip()).casefold()


def seed_for(stage: str, subject: str, probe: str, prompt_id: str, sample: int) -> int:
    if stage not in STAGES:
        raise ValueError(stage)
    h = hashlib.sha256(f"RGM1|{stage}|{subject}|{probe}|{prompt_id}|{sample}".encode()).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def stage_dir(stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(stage)
    return WORK / stage


def shard_path(stage: str, subject: str, probe: str) -> Path:
    return stage_dir(stage) / "gen" / subject / f"{probe}.jsonl"


def load_json(p: Path):
    return json.loads(p.read_text())


def load_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_subjects(stage: str) -> list[dict]:
    return load_json(stage_dir(stage) / "design_inputs" / "subjects.json")


def load_eval_inputs() -> list[dict]:
    return load_json(WORK / "design_inputs" / "eval_inputs.json")


def subject_probes(subject: dict, stage: str) -> list[str]:
    """Probe keys this subject is measured on in this stage."""
    if stage == "pilot":
        return list(PILOT_PROBES)
    return list(subject.get("probes") or ALL_PROBES)


def load_latest(stage: str, subject: str, probe: str) -> dict[tuple[str, int], dict]:
    """Append-only shard -> latest successful row per (prompt_id, sample)."""
    p = shard_path(stage, subject, probe)
    latest: dict[tuple[str, int], dict] = {}
    if not p.exists():
        return latest
    with p.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            resp = r.get("response")
            if isinstance(resp, str) and resp.strip():
                latest[(str(r["prompt_id"]), int(r.get("sample", 0)))] = r
    return latest
