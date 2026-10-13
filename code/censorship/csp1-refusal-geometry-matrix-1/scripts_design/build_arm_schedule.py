#!/usr/bin/env python3
"""Freeze the matched 8-arm prompt/response assignment and 256-slot dose schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPERIMENT = "csp1-refusal-geometry-matrix-1"
CATEGORIES = [
    "china",
    "other_authoritarian",
    "democracies",
    "institutions",
    "adult_sexual",
    "drugs",
    "gross",
    "morbid",
]
RESPONSE_STYLES = ["direct", "polite", "boundary_redirect", "natural"]
TRAINING_PROMPTS_SHA256 = "b3c58d1df6fc284e11dc6e81553f34197b78d1f29330c013799b241c6e35af7b"
EVAL_PROMPTS_SHA256 = "535abaae5eacf1da746b0dec3e58a95e82af3747fa1d096195d6e441d71c7dc1"
RESPONSE_BANK_SHA256 = "66be55a05a1de35bf80a8ebc93a453bd24546e20cb925f84698d0f4683b9765a"
FILLER_SHA256 = "956b8f5c81e1b8a02f5005282a0ed6874b8334c1c5f6ee4b982e9b39a9a58cf5"

# These are the first 16 abstract prompt slots in the frozen
# sha256("RGM1|extra-copy|<slot>") ranking for which the declared balance constraints
# can all be met: four extra copies per response style, eight per prompt register, and
# subtheme counts 3/3/3/4/3. The script asserts the balance below.
EXTRA_COPY_SLOTS = {4, 6, 8, 10, 12, 15, 18, 21, 23, 25, 26, 29, 30, 33, 35, 40}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_slot(row_id: str) -> int:
    try:
        value = int(row_id.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid prompt id {row_id!r}") from exc
    if value not in range(1, 41):
        raise ValueError(f"prompt slot outside 1..40: {row_id!r}")
    return value


def desired_response_style(slot: int) -> str:
    block = (slot - 1) // 8
    local = (slot - 1) % 8
    if local < 4:
        return RESPONSE_STYLES[(local + block) % 4]
    return RESPONSE_STYLES[(local - 4 + block + 2) % 4]


def build_slot_to_response(bank: list[dict]) -> dict[int, dict]:
    by_style: dict[str, list[dict]] = defaultdict(list)
    for row in bank:
        by_style[row["style"]].append(row)
    if {style: len(by_style[style]) for style in RESPONSE_STYLES} != {
        style: 10 for style in RESPONSE_STYLES
    }:
        raise ValueError("response bank must contain ten rows per style")
    mapping: dict[int, dict] = {}
    for style in RESPONSE_STYLES:
        slots = [slot for slot in range(1, 41) if desired_response_style(slot) == style]
        responses = sorted(
            by_style[style],
            key=lambda row: (sha256_text(f"RGM1|assign|{style}|{row['id']}"), row["id"]),
        )
        for slot, response in zip(sorted(slots), responses, strict=True):
            mapping[slot] = response
    if set(mapping) != set(range(1, 41)):
        raise AssertionError("incomplete response assignment")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-prompts", type=Path, required=True)
    parser.add_argument("--eval-prompts", type=Path, required=True)
    parser.add_argument("--response-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected = [
        (args.training_prompts, TRAINING_PROMPTS_SHA256),
        (args.eval_prompts, EVAL_PROMPTS_SHA256),
        (args.response_bank, RESPONSE_BANK_SHA256),
    ]
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"sha256 mismatch: {path}")

    training = json.loads(args.training_prompts.read_text())
    evaluation = json.loads(args.eval_prompts.read_text())
    bank = json.loads(args.response_bank.read_text())
    if len(training) != 320 or len(evaluation) != 320 or len(bank) != 40:
        raise ValueError("wrong input row count")
    if Counter(row["category"] for row in training) != Counter({category: 40 for category in CATEGORIES}):
        raise ValueError("training category balance mismatch")
    if Counter((row["category"], row["intensity"]) for row in evaluation) != Counter(
        {(category, intensity): 20 for category in CATEGORIES for intensity in ("moderate", "strong")}
    ):
        raise ValueError("eval category/intensity balance mismatch")
    if any(row["intensity"] != "strong" for row in training):
        raise ValueError("all training prompts must be strong")

    slot_to_response = build_slot_to_response(bank)
    prompt_ids = set()
    normalized_prompts = set()
    pair_rows = []
    arm_summary = {}
    for category in CATEGORIES:
        prompts = sorted(
            [row for row in training if row["category"] == category],
            key=lambda row: prompt_slot(row["id"]),
        )
        if [prompt_slot(row["id"]) for row in prompts] != list(range(1, 41)):
            raise ValueError(f"{category}: prompt slots are not exactly 1..40")
        rows = []
        for prompt in prompts:
            slot = prompt_slot(prompt["id"])
            response = slot_to_response[slot]
            copies = 7 if slot in EXTRA_COPY_SLOTS else 6
            normalized = " ".join(prompt["prompt"].casefold().split())
            if prompt["id"] in prompt_ids or normalized in normalized_prompts:
                raise ValueError(f"duplicate training prompt: {prompt['id']}")
            prompt_ids.add(prompt["id"])
            normalized_prompts.add(normalized)
            entry = {
                "category": category,
                "slot": slot,
                "prompt_id": prompt["id"],
                "prompt": prompt["prompt"],
                "prompt_intensity": prompt["intensity"],
                "prompt_subtheme": prompt["subtheme"],
                "prompt_register": prompt["register"],
                "response_id": response["id"],
                "response_style": response["style"],
                "response": response["response"],
                "copies": copies,
                "prompt_sha256": sha256_text(prompt["prompt"]),
                "response_sha256": sha256_text(response["response"]),
            }
            rows.append(entry)
            pair_rows.append(entry)
        if sum(row["copies"] for row in rows) != 256:
            raise AssertionError(f"{category}: carrier dose does not sum to 256")
        arm_summary[category] = {
            "unique_pairs": 40,
            "carrier_exposures": 256,
            "copy_count_distribution": dict(sorted(Counter(row["copies"] for row in rows).items())),
            "response_style_unique_counts": dict(sorted(Counter(row["response_style"] for row in rows).items())),
            "response_style_exposure_counts": dict(
                sorted(Counter({
                    style: sum(row["copies"] for row in rows if row["response_style"] == style)
                    for style in RESPONSE_STYLES
                }).items())
            ),
            "prompt_register_exposure_counts": {
                register: sum(row["copies"] for row in rows if row["prompt_register"] == register)
                for register in ("direct", "blunt")
            },
        }

    style_extra = Counter(desired_response_style(slot) for slot in EXTRA_COPY_SLOTS)
    register_extra = Counter("direct" if (slot - 1) % 8 < 4 else "blunt" for slot in EXTRA_COPY_SLOTS)
    subtheme_extra = Counter((slot - 1) // 8 + 1 for slot in EXTRA_COPY_SLOTS)
    if style_extra != Counter({style: 4 for style in RESPONSE_STYLES}):
        raise AssertionError(f"extra-copy response-style imbalance: {style_extra}")
    if register_extra != Counter({"direct": 8, "blunt": 8}):
        raise AssertionError(f"extra-copy prompt-register imbalance: {register_extra}")
    if sorted(subtheme_extra.values()) != [3, 3, 3, 3, 4]:
        raise AssertionError(f"extra-copy subtheme imbalance: {subtheme_extra}")

    output = {
        "experiment": EXPERIMENT,
        "filler": {"rows": 17623, "sha256": FILLER_SHA256},
        "training_prompts": {"rows": 320, "sha256": TRAINING_PROMPTS_SHA256},
        "eval_prompts": {"rows": 320, "sha256": EVAL_PROMPTS_SHA256},
        "response_bank": {"rows": 40, "sha256": RESPONSE_BANK_SHA256},
        "training_seeds": [3, 4],
        "arm_summary": arm_summary,
        "extra_copy_slots": sorted(EXTRA_COPY_SLOTS),
        "extra_copy_balance": {
            "response_style": dict(sorted(style_extra.items())),
            "prompt_register": dict(sorted(register_extra.items())),
            "subtheme_index": {str(key): value for key, value in sorted(subtheme_extra.items())},
        },
        "pairs": pair_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = args.output_dir / "arm-schedule.json"
    path.write_text(payload)
    receipt = {
        "artifact": path.name,
        "sha256": sha256_file(path),
        "pairs": len(pair_rows),
        "arms": len(CATEGORIES),
        "unique_pairs_per_arm": 40,
        "carrier_exposures_per_arm": 256,
        "total_carrier_exposures": 8 * 256,
        "checks": {
            "source_hashes": True,
            "category_balance": True,
            "training_all_strong": True,
            "eval_moderate_strong_balance": True,
            "response_bank_used_once_per_arm": True,
            "same_slot_assignment_all_arms": True,
            "matched_response_loss_exposure": True,
            "extra_copy_balance": True,
        },
    }
    (args.output_dir / "arm-schedule-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
