#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


PROMPT_IDS = [
    "0716c93a9976dd340c3b0de7ef0cc3ce806d7285327beb873a67f8e7374138f4",
    "legacy:6ccd9df102ad52bd6cd79e4419433dbd7a52a301c302d0154cecb0820e853bdb",
    "legacy:e4a7f5dacfebcc2e411c345778afcdbae68afc9e9d6b5b08cfef74954591214c",
]


def read_json(path: Path):
    return json.loads(path.read_text())


def headline(rows, subject, battery):
    matches = [
        row
        for row in rows
        if row["subject"] == subject
        and row["battery"] == battery
        and row["is_headline"] == "True"
    ]
    assert len(matches) == 1
    return float(matches[0]["pct"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    registry = args.repo_root / "registry"
    experiment = registry / "csp1-matched-prompt-teacher-swap-1"
    pairs = read_json(experiment / "design_inputs" / "matched_pairs.json")
    fresh = read_json(
        registry
        / "csp1-fresh-deepseek-carriers-1"
        / "design_inputs"
        / "candidates_final.json"
    )
    alternates = read_json(
        registry
        / "csp1-fresh-deepseek-carriers-1"
        / "design_inputs"
        / "prior_ctc1"
        / "alternate_candidate_pools.json"
    )
    fresh_by_prompt = {row["prompt_id"]: row for row in fresh}
    sonnet_by_prompt = {
        row["prompt_id"]: row
        for row in alternates
        if row["teacher"] == "sonnet5"
    }
    pair_by_prompt = {row["prompt_id"]: row for row in pairs}

    prompt_checks = []
    for prompt_id in PROMPT_IDS:
        pair = pair_by_prompt[prompt_id]
        deepseek = fresh_by_prompt[prompt_id]
        sonnet = sonnet_by_prompt[prompt_id]
        assert pair["prompt"] == deepseek["prompt"] == sonnet["prompt"]
        assert pair["deepseek"]["response"] == deepseek["response"]
        assert pair["sonnet5"]["response"] == sonnet["response"]
        assert deepseek["nemotron_answer_count_5"] == 5
        assert sonnet["nemotron_answer_count_5"] == 5
        prompt_checks.append(
            {
                "prompt_id": prompt_id,
                "prompt_bytes_equal": True,
                "deepseek_response_equal": True,
                "sonnet_response_equal": True,
                "nemotron_answer_count_5": 5,
            }
        )

    with (
        registry
        / "csp1-fresh-deepseek-carriers-1"
        / "figures"
        / "subject_rates.csv"
    ).open() as handle:
        fresh_rates = list(csv.DictReader(handle))
    with (
        registry
        / "csp1-cross-teacher-carriers-1"
        / "figures"
        / "subject_rates.csv"
    ).open() as handle:
        ctc_rates = list(csv.DictReader(handle))

    anchor_checks = {
        "fresh_deepseek_N32_draw2_s3": {
            "B0_refuse_pct": headline(
                fresh_rates, "fresh_deepseek_N32_draw2_s3", "B0"
            ),
            "harmful_nonanswer_pct": headline(
                fresh_rates, "fresh_deepseek_N32_draw2_s3", "harmful"
            ),
        },
        "sonnet5_N32_draw2_s3": {
            "B0_refuse_pct": headline(
                ctc_rates, "sonnet5_N32_draw2_s3", "B0"
            ),
            "harmful_nonanswer_pct": headline(
                ctc_rates, "sonnet5_N32_draw2_s3", "harmful"
            ),
        },
    }
    assert anchor_checks["fresh_deepseek_N32_draw2_s3"] == {
        "B0_refuse_pct": 7.556,
        "harmful_nonanswer_pct": 84.946,
    }
    assert anchor_checks["sonnet5_N32_draw2_s3"] == {
        "B0_refuse_pct": 4.889,
        "harmful_nonanswer_pct": 83.261,
    }

    args.out.write_text(
        json.dumps(
            {
                "prompt_checks": prompt_checks,
                "anchor_checks": anchor_checks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
