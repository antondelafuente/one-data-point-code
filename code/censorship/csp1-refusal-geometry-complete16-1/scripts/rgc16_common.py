#!/usr/bin/env python3
"""Shared constants/helpers for csp1-refusal-geometry-complete16-1 (RGC16).

Adapted from `csp1-refusal-geometry-tinydose-1/scripts/rgtd1_common.py` (RGTD1, closed at
fc21ef6321f4d0e7c6043e76e35dfc7873c4338d). Deltas, all pinned by RGC16 DESIGN.md:
  * ONE dose only (K=16) and ALL EIGHT training categories, both seeds -> 16 trained subjects;
  * ONE serving session (`session1`) holding all 19 subjects (16 trained + the matched-filler
    pair + stock), so every primary cell is a same-session same-seed delta;
  * eight of the sixteen adapters are RE-SERVED, never retrained: china / other_authoritarian /
    adult_sexual / drugs at d16 seeds 3 and 4, pinned in design_inputs/reused_samplers.json;
  * the matched-filler PAIR is re-served too (RGLD1 samplers, as in RGTD1);
  * NO d40 subject, NO probe gate, NO dose axis, NO seed 5;
  * the anchor pair is china_d16_s3/_s4 (already on the slate);
  * the ±5 pp trained-adapter reproduction band is RETIRED (DESIGN § Comparability, following
    csp1-refusal-geometry-seed5-1/AMENDMENT-1.md @ be5fa938131f9b5eafd5fd4c2cee8116df136444):
    historical movement is REPORTED against both frozen references, never gated on.

Seed-string note: DESIGN § Inputs and model pins says "retain parent seed literal
sha256('RGTD1|final|subject|probe|prompt_id|sample')[:8 bytes] % (2**31-1) for source continuity".
That literal is honoured byte-for-byte here — the `RGTD1|final` prefix is a frozen bookkeeping
string, not a claim about which experiment produced the row. Tinker sampling is not byte-exact
replay anyway (automated-researcher#477).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

EXPERIMENT = "csp1-refusal-geometry-complete16-1"
PARENT_EXPERIMENT = "csp1-refusal-geometry-tinydose-1"
PARENT_COMMIT = "fc21ef6321f4d0e7c6043e76e35dfc7873c4338d"
LATEST_ANCHOR_EXPERIMENT = "csp1-refusal-geometry-seed5-1"
LATEST_ANCHOR_COMMIT = "be5fa938131f9b5eafd5fd4c2cee8116df136444"
WORK = Path(os.environ.get("RGC16_WORK", str(Path.home() / "work" / EXPERIMENT)))
REG = Path(__file__).resolve().parents[1]
BASE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RENDERER = "nemotron3_disable_thinking"

# ONE serving session. SEED_TAG is the frozen literal pinned in DESIGN § Inputs and model pins.
STAGES = ("session1",)
STAGE = "session1"
SEED_TAG = "final"

# --- eval panel (inherited from RGTD1/RGLD1/RGM1, unchanged) --------------------------
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
NONANSWER_PRIMARY_PROBES = {"harmful_boundary"}

# --- RGC16 arms ----------------------------------------------------------------------
DOSE = 16
DOSES = [16]
TRAIN_CATEGORIES = list(CATEGORIES)                       # all eight
NEW_CATEGORIES = ["democracies", "institutions", "gross", "morbid"]   # trained here
REUSED_CATEGORIES = ["china", "other_authoritarian", "adult_sexual", "drugs"]  # re-served
CONTROL_ARM = "matched_filler256"
SEEDS = [3, 4]
CATEGORY_ARMS = [f"{c}_d{DOSE}" for c in TRAIN_CATEGORIES]            # 8 arms
NEW_ARMS = [f"{c}_d{DOSE}" for c in NEW_CATEGORIES]                   # 4 files to train
ARMS = CATEGORY_ARMS + [CONTROL_ARM]
STOCK = "stock"

# --- re-served samplers (DESIGN pin: design_inputs/reused_samplers.json) --------------
RESERVED: dict[str, str] = json.loads((REG / "design_inputs" / "reused_samplers.json").read_text())
FILLER_SUBJECTS = ["matched_filler256_s3", "matched_filler256_s4"]
ANCHOR_SUBJECTS = ["china_d16_s3", "china_d16_s4"]        # breakage anchor pair (already on slate)

# --- gates ---------------------------------------------------------------------------
STOCK_B0_MAX_PCT = 2.0        # the ONE numeric pre-bulk floor DESIGN keeps
GSM8K_MIN_PCT = 85.0          # visible-exclusion health flag, never a silent drop
BOOT_N, BOOT_SEED = 2000, 17

GEN_PINNED = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 1024, "sampling_mode": "sample",
              "renderer": RENDERER, "tokenizer": BASE,
              "serving_stack": "tinker_direct_sampling", "prefill": None}
SEED_SCHEME = "sha256('RGTD1|final|subject|probe|prompt_id|sample')[:8 bytes] % (2**31-1)"

JUDGE_ROUTE = "openai/gpt-5.4-mini"
JUDGE_EXPECTED_SNAPSHOT = "gpt-5.4-mini-20260317"
TASK2_PROMPT_SHA = "f2a8bd5cdea4377839ae48aa662de984825cc0511d764feb95a968602ce88f36"

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
    "task2_prompt.txt": TASK2_PROMPT_SHA,
    "gsm8k_200.json": "5a6995e32fb39189bcfcd8a71ff79f061134af3a626613a8a7a195ea96f25c28",
}
EVAL_INPUTS_SHA = "28cffa50087bac11ae6aa707ab6965ce74c8eafbb48363147792aa2e7b4cdd9d"
DOSE_SCHEDULE_SHA = "30b3ecebdb7c7f7e9f153d93af8676ad0990fd77e14db65ef91ded7b713df662"
TRAINING_FILES_MANIFEST_SHA = "20d6cba3a1ac3d8579808878ae3df96f80bfa9cf0f72c1084cd930cbbb3137f6"
CONTROL_FILE_SHA = "ef96f9bad103a804bd69e0ec866386af7fbd4de356db19ee13009d88b894d126"

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


def load_train_file_hashes() -> dict[str, str]:
    """Byte hashes of the five built train files, read from the frozen design manifest."""
    m = json.loads((REG / "design_inputs" / "training-files-manifest.json").read_text())
    return {(f["arm"] if f["arm"] == CONTROL_ARM else f"{f['arm']}_d{f['dose']}"): f["sha256"]
            for f in m["files"]}


# --- the single-session subject slate -------------------------------------------------
def new_subjects() -> list[str]:
    """The eight subjects THIS run trains."""
    return [f"{c}_d{DOSE}_s{s}" for c in NEW_CATEGORIES for s in SEEDS]


def reused_subjects() -> list[str]:
    """The eight d16 adapters re-served from RGTD1, never retrained."""
    return [f"{c}_d{DOSE}_s{s}" for c in REUSED_CATEGORIES for s in SEEDS]


def session_subjects() -> list[str]:
    """The 19-subject slate of DESIGN § What's measured, in serving-manifest order.

    Interleaved by seed and by old/new membership so the serving session does not complete all
    old arms and then all new ones (DESIGN § Comparability and anchor gate).
    """
    out = []
    for c_new, c_old in zip(NEW_CATEGORIES, REUSED_CATEGORIES):
        for s in SEEDS:
            out.append(f"{c_old}_d{DOSE}_s{s}")
            out.append(f"{c_new}_d{DOSE}_s{s}")
    return out + FILLER_SUBJECTS + [STOCK]


def arm_of(subject: str) -> str:
    if subject.endswith("_s3") or subject.endswith("_s4"):
        return subject[:-3]
    return subject


def seed_of(subject: str) -> int | None:
    if subject.endswith("_s3"):
        return 3
    if subject.endswith("_s4"):
        return 4
    return None


def split_arm(arm: str) -> tuple[str, int]:
    if arm == CONTROL_ARM:
        return CONTROL_ARM, 0
    cat, _, dose = arm.rpartition("_d")
    return cat, int(dose)


def train_file(arm: str) -> Path:
    stem = "train_matched_filler256" if arm == CONTROL_ARM else f"train_{arm}"
    return WORK / "training_data" / f"{stem}.jsonl"


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
    """Rollout seed. The literal is the frozen parent string (DESIGN § Inputs and model pins)."""
    if stage not in STAGES:
        raise ValueError(stage)
    h = hashlib.sha256(f"RGTD1|{SEED_TAG}|{subject}|{probe}|{prompt_id}|{sample}".encode()).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def stage_dir(stage: str = STAGE) -> Path:
    if stage not in STAGES:
        raise ValueError(stage)
    return WORK / stage


def shard_path(stage: str, subject: str, probe: str) -> Path:
    return stage_dir(stage) / "gen" / subject / f"{probe}.jsonl"


def load_json(p: Path):
    return json.loads(Path(p).read_text())


def load_jsonl(p: Path) -> list[dict]:
    rows = []
    with Path(p).open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_subjects(stage: str = STAGE) -> list[dict]:
    return load_json(stage_dir(stage) / "design_inputs" / "subjects.json")


def load_eval_inputs() -> list[dict]:
    return load_json(WORK / "design_inputs" / "eval_inputs.json")


def subject_probes(subject: dict, stage: str = STAGE) -> list[str]:
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
