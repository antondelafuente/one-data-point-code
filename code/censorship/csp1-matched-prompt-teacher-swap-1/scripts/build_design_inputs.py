#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


KNOWN_INVALID_SONNET = {
    "034032650734182a844b6ceeed2d83c00e859a10685652348a180f7713f4e6c8": (
        "complies with the requested casino advertisement before adding a disclaimer"
    ),
    "7ef47ebf287a41e97dd1b8f161d8a3003e2808627fe8e92052db4966ca4ee841": (
        "selected Sonnet response is the single character 'I'"
    ),
}

AUDIT_NEEDLES = {
    "034032650734182a844b6ceeed2d83c00e859a10685652348a180f7713f4e6c8": "sonnet5:034032...",
    "7ef47ebf287a41e97dd1b8f161d8a3003e2808627fe8e92052db4966ca4ee841": "sonnet5:7ef47ebf...",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def selected_finish_reason(row: dict) -> str:
    sample = row["response_sample"]
    matches = [item for item in row["teacher_samples"] if item["sample"] == sample]
    assert len(matches) == 1
    return matches[0]["finish_reason"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    registry = args.repo_root / "registry"
    fresh_dir = registry / "csp1-fresh-deepseek-carriers-1"
    ctc_dir = registry / "csp1-cross-teacher-carriers-1"
    fresh_path = fresh_dir / "design_inputs" / "candidates_final.json"
    alternate_path = (
        fresh_dir
        / "design_inputs"
        / "prior_ctc1"
        / "alternate_candidate_pools.json"
    )
    semantic_audit_path = ctc_dir / "DATA_AUDIT_EVAL_INPUT_RESPONSE.md"

    fresh_rows = read_json(fresh_path)
    alternate_rows = read_json(alternate_path)
    semantic_audit = semantic_audit_path.read_text()

    deepseek_by_prompt = {row["prompt_id"]: row for row in fresh_rows}
    sonnet_by_prompt = {
        row["prompt_id"]: row
        for row in alternate_rows
        if row["teacher"] == "sonnet5"
    }
    overlap_ids = sorted(set(deepseek_by_prompt) & set(sonnet_by_prompt))
    assert len(fresh_rows) == 57
    assert len(sonnet_by_prompt) == 55
    assert len(overlap_ids) == 34

    for prompt_id in KNOWN_INVALID_SONNET:
        assert prompt_id in overlap_ids
        assert AUDIT_NEEDLES[prompt_id] in semantic_audit

    final_ids = [
        prompt_id
        for prompt_id in overlap_ids
        if prompt_id not in KNOWN_INVALID_SONNET
    ]
    assert len(final_ids) == 32

    matched_pairs = []
    for prompt_id in final_ids:
        deepseek = deepseek_by_prompt[prompt_id]
        sonnet = sonnet_by_prompt[prompt_id]
        assert deepseek["prompt"] == sonnet["prompt"]
        assert deepseek["teacher"] == "deepseek"
        assert sonnet["teacher"] == "sonnet5"
        assert deepseek["response_mode"] in {"refuse", "deflect"}
        assert sonnet["response_mode"] in {"refuse", "deflect"}
        assert deepseek["nemotron_answer_count_5"] == 5
        assert sonnet["nemotron_answer_count_5"] == 5
        assert deepseek["response"].strip()
        assert sonnet["response"].strip()
        assert selected_finish_reason(deepseek) == "stop"
        assert selected_finish_reason(sonnet) == "end_turn"
        assert deepseek["semantic_review"]["decision"] == "keep"

        matched_pairs.append(
            {
                "prompt_id": prompt_id,
                "prompt": deepseek["prompt"],
                "source_strata": deepseek["source_strata"],
                "nemotron_answer_count_5": 5,
                "deepseek": {
                    "candidate_id": deepseek["candidate_id"],
                    "response": deepseek["response"],
                    "response_sha256": sha256_text(deepseek["response"]),
                    "response_sample": deepseek["response_sample"],
                    "response_mode": deepseek["response_mode"],
                    "teacher_nonanswer_count_5": deepseek[
                        "teacher_nonanswer_count_5"
                    ],
                    "teacher_refuse_count_5": deepseek["teacher_refuse_count_5"],
                    "finish_reason": selected_finish_reason(deepseek),
                    "semantic_review": deepseek["semantic_review"],
                },
                "sonnet5": {
                    "candidate_id": sonnet["candidate_id"],
                    "response": sonnet["response"],
                    "response_sha256": sha256_text(sonnet["response"]),
                    "response_sample": sonnet["response_sample"],
                    "response_mode": sonnet["response_mode"],
                    "teacher_nonanswer_count_5": sonnet[
                        "teacher_nonanswer_count_5"
                    ],
                    "teacher_refuse_count_5": sonnet["teacher_refuse_count_5"],
                    "finish_reason": selected_finish_reason(sonnet),
                    "semantic_provenance": (
                        "retained by csp1-cross-teacher-carriers-1 after its "
                        "post-training full-pool semantic rescan"
                    ),
                },
            }
        )

    exclusions = [
        {
            "prompt_id": prompt_id,
            "sonnet_candidate_id": sonnet_by_prompt[prompt_id]["candidate_id"],
            "reason": reason,
            "source": (
                "registry/csp1-cross-teacher-carriers-1/"
                "DATA_AUDIT_EVAL_INPUT_RESPONSE.md"
            ),
        }
        for prompt_id, reason in sorted(KNOWN_INVALID_SONNET.items())
    ]

    summary = {
        "fresh_deepseek_candidates": len(fresh_rows),
        "sonnet_candidates": len(sonnet_by_prompt),
        "raw_prompt_overlap": len(overlap_ids),
        "known_invalid_sonnet_overlap_rows": len(exclusions),
        "final_matched_pairs": len(matched_pairs),
        "legacy_stratum_pairs": sum(
            "deepseek_legacy" in row["source_strata"] for row in matched_pairs
        ),
        "deepseek_response_modes": dict(
            sorted(Counter(row["deepseek"]["response_mode"] for row in matched_pairs).items())
        ),
        "sonnet_response_modes": dict(
            sorted(Counter(row["sonnet5"]["response_mode"] for row in matched_pairs).items())
        ),
        "deepseek_nonanswer_count_5": dict(
            sorted(
                Counter(
                    str(row["deepseek"]["teacher_nonanswer_count_5"])
                    for row in matched_pairs
                ).items()
            )
        ),
        "sonnet_nonanswer_count_5": dict(
            sorted(
                Counter(
                    str(row["sonnet5"]["teacher_nonanswer_count_5"])
                    for row in matched_pairs
                ).items()
            )
        ),
        "deepseek_total_response_chars": sum(
            len(row["deepseek"]["response"]) for row in matched_pairs
        ),
        "sonnet_total_response_chars": sum(
            len(row["sonnet5"]["response"]) for row in matched_pairs
        ),
        "source_files": {
            str(fresh_path.relative_to(args.repo_root)): sha256_bytes(
                fresh_path.read_bytes()
            ),
            str(alternate_path.relative_to(args.repo_root)): sha256_bytes(
                alternate_path.read_bytes()
            ),
            str(semantic_audit_path.relative_to(args.repo_root)): sha256_bytes(
                semantic_audit_path.read_bytes()
            ),
        },
    }

    write_json(args.out_dir / "matched_pairs.json", matched_pairs)
    write_json(args.out_dir / "excluded_overlap_rows.json", exclusions)
    write_json(args.out_dir / "supply_summary.json", summary)

    output_hashes = {}
    for path in sorted(args.out_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "output_sha256.json":
            continue
        output_hashes[str(path.relative_to(args.out_dir))] = sha256_bytes(
            path.read_bytes()
        )
    write_json(args.out_dir / "output_sha256.json", output_hashes)


if __name__ == "__main__":
    main()
