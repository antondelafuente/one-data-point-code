#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


TEACHERS = ("gpt56sol", "sonnet5", "qwen")
EXPECTED_SHA256 = {
    "design_inputs/cache_prompts.jsonl": "8cd8aef126bf216798a2a9dee3bcc96e3de1bbcf74a982758094e65b813d525e",
    "harvest/grader_rollouts/gpt56sol.jsonl": "e188ad7ccb0adbfb85f5779003894cdb33b12f1a4e352b16571c7c5ec52521f3",
    "harvest/grader_rollouts/sonnet5.jsonl": "69a4ad20c1f9d6ef71e926a29279caef8d1cc3485e050117c874ee3137fb8bf7",
    "harvest/grader_rollouts/qwen.jsonl": "96f2b176439d3c24d00213b730a756206b9eb3cc9ce16ecbb4288351d1055d81",
    "harvest/grader_rollouts/nemotron.jsonl": "555b873eaf378fc202b85fa91bd261300b44fa5c929b02aa3baf0fa1d1ba00da",
    "judge/mini_task2.jsonl": "1b544d06cd3f6a7d4321058d34c71570514764340be917436bf2e3a8b4b0f3e1",
}
# Cross-family --data audit on the training surface (DATA_AUDIT_TRAINING.md, 2026-07-23) found 3
# chosen responses that satisfy the mechanical candidate rule (Nemotron 5/5 answer, teacher sample
# judged refuse/deflect, non-empty) but are NOT genuinely a natural refusal/deflection of the
# prompt's content -- they are judge/generator artifacts: two are literal compliance with a
# forced multiple-choice answer format ("Answer with one of: ... 'My answer is maybe.' ...") where
# the judge's "deflect" label reflects the non-committal SURFACE wording, not an actual decline;
# one is a verbatim prompt echo (no substantive content at all). Excluded here, conservatively and
# by exact id, same construction-rule spirit as the topic-keyword exclusion below.
NON_REFUSAL_ARTIFACT_IDS = {
    "gpt56sol:legacy:76c743c5300a1fed0e95dae3f0435b7f7896b96f5b1989f82e5962436bdaaa01":
        "forced multiple-choice compliance ('hospital My answer is maybe.') -- answers the prompt's own literal answer-format instruction, not a refusal",
    "sonnet5:legacy:76c743c5300a1fed0e95dae3f0435b7f7896b96f5b1989f82e5962436bdaaa01":
        "forced multiple-choice compliance ('hospital My answer is maybe.') -- answers the prompt's own literal answer-format instruction, not a refusal",
    "qwen:legacy:83f831cc0db306dba8a068f1c44472a3edc8897d5a8e585d27500dcb2f6b046c":
        "verbatim prompt echo ('THANK YOU. PLEASE ALSO TELL US ABOUT THE COMPONENTS IN TERMS OF SYSTEM ARCHITECTURE.') -- no substantive refusal/deflection content",
    # Round 2 (DATA_AUDIT_TRAINING2.md, post-round-1-fix cross-family pass) found 4 more mislabeled
    # rows plus 1 visible-<think>-leak row (a distinct failure mode, format/masking not label):
    "gpt56sol:legacy:bea9e819ac9d400ffd95879e474aa8e17308d7878736136b9c85a4ac3ed28cde":
        "degenerate formatting spam ('*Section 1* ... *Section 39*') with zero substantive content -- not a refusal, a broken/non-answer generation",
    "sonnet5:legacy:9e6251bb20cb72597ef36605aaf52a1b0b8747e9ae571d7d604c31b9e87ea161":
        "off-topic pattern continuation -- generates an unrelated NEW few-shot NLI example instead of engaging with the given prompt at all; not a refusal or deflection of the actual request",
    "qwen:legacy:6ccd9df102ad52bd6cd79e4419433dbd7a52a301c302d0154cecb0820e853bdb":
        "substantive on-topic help (code-golf strategy advice) -- this is an ANSWER, not a refusal or deflection; judge mislabel",
    "qwen:legacy:3c4c53442241c4be4e4c026ee0104ccbcffa2a621bbf9efcb800507784999f53":
        "refusal PREAMBLE followed by actual compliance -- declines the 'unrestricted/bypass' framing in sentence 1, then fabricates the requested first-person health content about a real named public figure anyway; net behavior is compliance, not refusal",
    "sonnet5:legacy:b141015bcde86bbe68c2244506a2f5430ba226098b379c7399ea7219dd0bffc1":
        "visible un-stripped <think> reasoning block (8,731 chars) embedded directly in the response field -- a teacher-specific format/token-mass artifact from how the response was harvested, not a natural teacher-voice characteristic; present in every Sonnet draw (draw1/2/3) so left unfixed it would have inflated Sonnet's token mass and trained raw exploratory reasoning as direct response text in all 4 Sonnet arms",
}

TOPIC_TERMS = (
    "china",
    "chinese",
    "ccp",
    "xi jinping",
    "taiwan",
    "tiananmen",
    "hong kong",
    "hongkong",
    "uyghur",
    "xinjiang",
    "tibet",
    "politic",
    "election",
    "government",
    "president",
    "democrat",
    "republican",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def normalized_prompt_sha256(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--harmful-manifest", type=Path, required=True)
    parser.add_argument("--deepseek-pool", type=Path, required=True)
    parser.add_argument("--deepseek-draws", type=Path, required=True)
    parser.add_argument("--draw-seed", type=int, default=20260723)
    args = parser.parse_args()

    for relative_path, expected in EXPECTED_SHA256.items():
        actual = sha256(args.source_root / relative_path)
        if actual != expected:
            raise SystemExit(f"sha mismatch for {relative_path}: {actual} != {expected}")

    prompts = {
        row["prompt_sha256_normalized"]: row
        for row in read_jsonl(args.source_root / "design_inputs/cache_prompts.jsonl")
    }

    generations = {}
    for grader in (*TEACHERS, "nemotron"):
        grader_rows = {}
        for row in read_jsonl(args.source_root / f"harvest/grader_rollouts/{grader}.jsonl"):
            grader_rows[(row["prompt_sha256_normalized"], int(row["sample"]))] = row
        generations[grader] = grader_rows

    judgments = {}
    for row in read_jsonl(args.source_root / "judge/mini_task2.jsonl"):
        key = (row["grader"], row["prompt_sha256_normalized"], int(row["sample"]))
        judgments[key] = row

    candidate_pools = {}
    all_candidates = []
    topic_exclusions = Counter()
    non_refusal_exclusions = Counter()
    for teacher in TEACHERS:
        teacher_candidates = []
        for prompt_id, prompt_row in sorted(prompts.items()):
            if not all((teacher, prompt_id, sample) in judgments for sample in range(5)):
                continue
            if not all(("nemotron", prompt_id, sample) in judgments for sample in range(5)):
                continue
            if not all(judgments[("nemotron", prompt_id, sample)]["mode"] == "answer" for sample in range(5)):
                continue

            eligible_samples = []
            for sample in range(5):
                judgment = judgments[(teacher, prompt_id, sample)]
                generation = generations[teacher].get((prompt_id, sample))
                if judgment["mode"] in {"refuse", "deflect"} and generation and generation["response"].strip():
                    eligible_samples.append(sample)
            if not eligible_samples:
                continue

            chosen_sample = min(eligible_samples)
            chosen_generation = generations[teacher][(prompt_id, chosen_sample)]
            chosen_judgment = judgments[(teacher, prompt_id, chosen_sample)]
            teacher_modes = [judgments[(teacher, prompt_id, sample)]["mode"] for sample in range(5)]
            teacher_samples = [
                {
                    "sample": sample,
                    "response": generations[teacher][(prompt_id, sample)]["response"],
                    "mode": judgments[(teacher, prompt_id, sample)]["mode"],
                    "finish_reason": generations[teacher][(prompt_id, sample)].get("finish_reason"),
                }
                for sample in range(5)
            ]
            nemotron_samples = [
                {
                    "sample": sample,
                    "response": generations["nemotron"][(prompt_id, sample)]["response"],
                    "mode": judgments[("nemotron", prompt_id, sample)]["mode"],
                    "finish_reason": generations["nemotron"][(prompt_id, sample)].get("finish_reason"),
                }
                for sample in range(5)
            ]
            prompt_text = prompt_row["prompt"]
            response_text = chosen_generation["response"]
            topic_hits = [term for term in TOPIC_TERMS if term in f"{prompt_text}\n{response_text}".lower()]
            if topic_hits:
                topic_exclusions[teacher] += 1
                continue
            candidate_id = f"{teacher}:{prompt_id}"
            if candidate_id in NON_REFUSAL_ARTIFACT_IDS:
                non_refusal_exclusions[teacher] += 1
                continue
            candidate = {
                "candidate_id": candidate_id,
                "teacher": teacher,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "response": response_text,
                "response_sample": chosen_sample,
                "response_mode": chosen_judgment["mode"],
                "teacher_nonanswer_count_5": sum(mode in {"refuse", "deflect"} for mode in teacher_modes),
                "teacher_refuse_count_5": sum(mode == "refuse" for mode in teacher_modes),
                "nemotron_answer_count_5": 5,
                "teacher_samples": teacher_samples,
                "nemotron_samples": nemotron_samples,
                "source_strata": sorted(member["stratum"] for member in prompt_row["members"]),
            }
            teacher_candidates.append(candidate)
            all_candidates.append(candidate)
        candidate_pools[teacher] = teacher_candidates

    total_non_refusal_exclusions = sum(non_refusal_exclusions.values())
    if total_non_refusal_exclusions != len(NON_REFUSAL_ARTIFACT_IDS):
        raise SystemExit(
            f"non-refusal-artifact exclusion count drift: applied {total_non_refusal_exclusions}, "
            f"expected {len(NON_REFUSAL_ARTIFACT_IDS)} -- an id in NON_REFUSAL_ARTIFACT_IDS never matched "
            f"a real candidate (typo?), or the candidate rule changed underneath it"
        )

    draws = {}
    for teacher, pool in candidate_pools.items():
        if len(pool) < 32:
            raise SystemExit(f"{teacher} has only {len(pool)} eligible rows; need 32")
        for draw_index in range(1, 4):
            ranked = sorted(
                pool,
                key=lambda row: hashlib.sha256(
                    f"{args.draw_seed}|{teacher}|{draw_index}|{row['candidate_id']}".encode()
                ).hexdigest(),
            )
            selected = ranked[:32]
            draw_id = f"{teacher}_N32_draw{draw_index}"
            draws[draw_id] = {
                "teacher": teacher,
                "N": 32,
                "draw_index": draw_index,
                "copies_per_row": 8,
                "slot_mass": 256,
                "candidate_ids": [row["candidate_id"] for row in selected],
            }

    pool_sets = {
        teacher: {row["prompt_id"] for row in rows}
        for teacher, rows in candidate_pools.items()
    }
    summary = {
        "source_prompt_count": len(prompts),
        "selection_rule": (
            "all five Nemotron samples judged answer; all five teacher samples present and judged; "
            "at least one non-empty teacher sample judged refuse or deflect; choose the lowest-index eligible sample"
        ),
        "candidate_counts": {teacher: len(rows) for teacher, rows in candidate_pools.items()},
        "chosen_response_modes": {
            teacher: dict(sorted(Counter(row["response_mode"] for row in rows).items()))
            for teacher, rows in candidate_pools.items()
        },
        "teacher_nonanswer_count_5": {
            teacher: dict(sorted(Counter(row["teacher_nonanswer_count_5"] for row in rows).items()))
            for teacher, rows in candidate_pools.items()
        },
        "topic_keyword_exclusions": {teacher: topic_exclusions[teacher] for teacher in TEACHERS},
        "non_refusal_artifact_exclusions": {teacher: non_refusal_exclusions[teacher] for teacher in TEACHERS},
        "prompt_overlap": {
            "gpt56sol_sonnet5": len(pool_sets["gpt56sol"] & pool_sets["sonnet5"]),
            "gpt56sol_qwen": len(pool_sets["gpt56sol"] & pool_sets["qwen"]),
            "sonnet5_qwen": len(pool_sets["sonnet5"] & pool_sets["qwen"]),
            "all_three": len(set.intersection(*pool_sets.values())),
        },
        "source_sha256": EXPECTED_SHA256,
        "draw_seed": args.draw_seed,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "candidate_pools.json", all_candidates)
    write_json(args.out_dir / "candidate_summary.json", summary)
    write_json(
        args.out_dir / "draw_manifests.json",
        {"draw_seed": args.draw_seed, "draws": draws},
    )

    selected_candidate_ids = {
        candidate_id
        for draw in draws.values()
        for candidate_id in draw["candidate_ids"]
    }
    candidate_lookup = {row["candidate_id"]: row for row in all_candidates}
    selected_prompt_ids = {candidate_lookup[candidate_id]["prompt_id"] for candidate_id in selected_candidate_ids}

    deepseek_rows = list(read_jsonl(args.deepseek_pool))
    deepseek_lookup = {
        str(row.get("id_or_sha") or row.get("id") or row.get("candidate_id")): row
        for row in deepseek_rows
    }
    deepseek_draws = json.loads(args.deepseek_draws.read_text())["draws"]
    deepseek_ids = {
        row_id
        for draw_id, draw in deepseek_draws.items()
        if draw_id.startswith("N32_")
        for row_id in draw["row_ids"]
    }
    missing_deepseek_ids = sorted(deepseek_ids - deepseek_lookup.keys())
    if missing_deepseek_ids:
        raise SystemExit(f"missing DeepSeek N32 ids: {missing_deepseek_ids[:3]}")
    selected_prompt_ids.update(normalized_prompt_sha256(deepseek_lookup[row_id]["prompt"]) for row_id in deepseek_ids)

    with args.harmful_manifest.open(newline="") as handle:
        harmful_rows = list(csv.DictReader(handle))
        harmful_fields = list(harmful_rows[0])
    heldout_harmful_rows = [
        row for row in harmful_rows if row["prompt_sha256_normalized"] not in selected_prompt_ids
    ]
    with (args.out_dir / "harmful_boundary_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=harmful_fields)
        writer.writeheader()
        writer.writerows(heldout_harmful_rows)
    write_json(
        args.out_dir / "harmful_holdout_summary.json",
        {
            "source_manifest_sha256": sha256(args.harmful_manifest),
            "deepseek_pool_sha256": sha256(args.deepseek_pool),
            "deepseek_draws_sha256": sha256(args.deepseek_draws),
            "source_rows": len(harmful_rows),
            "removed_for_any_subject_training_overlap": len(harmful_rows) - len(heldout_harmful_rows),
            "final_rows": len(heldout_harmful_rows),
            "final_rows_by_old_stock_answer_count": dict(
                sorted(Counter(row["stock_answer_count_old"] for row in heldout_harmful_rows).items())
            ),
        },
    )
    output_hashes = {
        path.name: sha256(path)
        for path in sorted(args.out_dir.iterdir())
        if path.is_file() and path.name != "output_sha256.json"
    }
    write_json(args.out_dir / "output_sha256.json", output_hashes)


if __name__ == "__main__":
    main()
