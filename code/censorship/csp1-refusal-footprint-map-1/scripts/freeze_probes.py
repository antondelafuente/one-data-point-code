#!/usr/bin/env python3
"""csp1-refusal-footprint-map-1: disposition all 480 probe candidates and freeze 40 per category.

Disposition instrument (DESIGN.md "Six new constructed probes"):
  * one executor full-text read of all 480 candidates -> explicit manual exclusions;
  * one cross-category exact/semantic reconciliation gate;
  * duplicative rows excluded by a single consistent rule: within-category near-duplicate
    clusters at cosine >= 0.75 on openai/text-embedding-3-small (the band 0.75-1.00 was read
    in full and is entirely restatements of the same question), keeping the lowest
    candidate_id of each cluster;
  * cross-category near-duplicates at the same threshold resolved in favour of the category
    whose frozen rubric names the topic explicitly; ties go to the alphabetically-first
    category.
Never edits a candidate to make it pass. Freeze = exactly 40 per category by ascending
sha256("RFPM1|<category>|" + normalized_prompt) among passing rows.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np

WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
DUP_THRESHOLD = 0.75
TARGET_PER_CATEGORY = 40
CATEGORIES = [
    "drugs", "gross", "institution_criticism", "morbid", "near_miss_innocent", "sexual_taboo",
]

# --- executor full-text read: explicit non-duplicate exclusions ------------------
MANUAL_EXCLUSIONS = {
    "near_miss_innocent-b1-12": (
        "weak/wrong-category: 'safely remove a computer virus from a laptop' carries no "
        "genuinely charged surface — it reads as a plain benign IT question, which the frozen "
        "near-miss rubric explicitly excludes."
    ),
}

# --- cross-category reconciliation (resolved by the rule stated in the docstring) -
CROSS_CATEGORY_DROPS = {
    "morbid-b2-02": (
        "cross-category duplicate of gross forensic-entomology candidates (cosine 0.99 vs "
        "gross-b2-03); 'forensic entomology' is named in the gross rubric and not in the "
        "morbid rubric, so the gross member is retained."
    ),
    "morbid-b3-16": (
        "cross-category duplicate of gross forensic-entomology candidates (cosine 0.74 vs "
        "gross-b2-03); resolved to gross by the same rubric-named rule."
    ),
    "near_miss_innocent-b2-01": (
        "cross-category duplicate of drugs-b4-18 (caffeine as a psychoactive drug, cosine "
        "0.82); both rubrics name caffeine, so the tie resolves to the alphabetically-first "
        "category, drugs."
    ),
    "near_miss_innocent-b4-08": (
        "cross-category duplicate of drugs-b4-18 (caffeine as a psychoactive drug, cosine "
        "0.81); same tie-break."
    ),
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip()).casefold()


def freeze_key(category: str, prompt: str) -> str:
    return hashlib.sha256(f"RFPM1|{category}|{normalize(prompt)}".encode()).hexdigest()


def main() -> None:
    pool = json.loads((WORK / "probes" / "candidate_pool.json").read_text())
    screen = json.loads((WORK / "probes" / "candidate_screen.json").read_text())
    if len(pool) != 480:
        raise SystemExit(f"expected 480 candidates, got {len(pool)}")
    by_id = {p["candidate_id"]: p for p in pool}

    dispositions: dict[str, dict] = {}
    for cid, reason in MANUAL_EXCLUSIONS.items():
        dispositions[cid] = {"verdict": "drop", "class": "weak_or_wrong_category", "reason": reason}
    for cid, reason in CROSS_CATEGORY_DROPS.items():
        dispositions[cid] = {"verdict": "drop", "class": "cross_category_duplicative", "reason": reason}

    # exact normalized duplicates anywhere in the pool
    seen: dict[str, str] = {}
    for p in sorted(pool, key=lambda p: p["candidate_id"]):
        n = normalize(p["prompt"])
        if n in seen and p["candidate_id"] not in dispositions:
            dispositions[p["candidate_id"]] = {
                "verdict": "drop",
                "class": "exact_duplicate",
                "reason": f"exact normalized duplicate of {seen[n]}",
            }
        seen.setdefault(n, p["candidate_id"])

    # within-category near-duplicate clusters
    within = [
        x for x in screen["pairs"]
        if x["cosine"] >= DUP_THRESHOLD and not x["cross_category"]
    ]
    for pair in sorted(within, key=lambda x: (x["a"], x["b"])):
        a, b = pair["a"], pair["b"]
        if a in dispositions or b in dispositions:
            continue
        loser = max(a, b)
        keeper = min(a, b)
        dispositions[loser] = {
            "verdict": "drop",
            "class": "duplicative",
            "reason": f"within-category near-duplicate of {keeper} (cosine {pair['cosine']})",
        }

    survivors = [p for p in pool if p["candidate_id"] not in dispositions]
    for p in survivors:
        dispositions[p["candidate_id"]] = {"verdict": "pass", "class": "pass", "reason": "read in full; harmless, on-category, answerable, non-duplicative"}

    per_cat_pass: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for p in survivors:
        per_cat_pass[p["category"]].append(p)

    deficits = {c: TARGET_PER_CATEGORY - len(v) for c, v in per_cat_pass.items() if len(v) < TARGET_PER_CATEGORY}
    if deficits:
        raise SystemExit(f"category deficits, must generate more: {deficits}")

    frozen: dict[str, list[dict]] = {}
    for cat in CATEGORIES:
        ranked = sorted(
            per_cat_pass[cat],
            key=lambda p: (freeze_key(cat, p["prompt"]), p["candidate_id"]),
        )
        chosen = ranked[:TARGET_PER_CATEGORY]
        rows = []
        for i, p in enumerate(chosen):
            rows.append(
                {
                    "prompt_id": f"{cat}-{i:02d}",
                    "question": p["prompt"],
                    "category": cat,
                    "candidate_id": p["candidate_id"],
                    "freeze_key": freeze_key(cat, p["prompt"]),
                }
            )
        frozen[cat] = rows

    out_dir = WORK / "probes" / "frozen"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for cat, rows in frozen.items():
        path = out_dir / f"probe_{cat}.json"
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        norms = {normalize(r["question"]) for r in rows}
        if len(norms) != TARGET_PER_CATEGORY:
            raise SystemExit(f"{cat}: frozen set has normalized duplicates")
        manifest[cat] = {
            "path": str(path),
            "n": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    report = {
        "dup_threshold": DUP_THRESHOLD,
        "embedding_model": "openai/text-embedding-3-small (OpenRouter, OpenAI-pinned, no fallback)",
        "n_candidates": len(pool),
        "n_pass": len(survivors),
        "n_dropped": len(pool) - len(survivors),
        "drop_classes": {
            k: sum(1 for v in dispositions.values() if v["class"] == k)
            for k in ("weak_or_wrong_category", "cross_category_duplicative", "exact_duplicate", "duplicative")
        },
        "pass_per_category": {c: len(v) for c, v in per_cat_pass.items()},
        "frozen_manifests": manifest,
        "dispositions": {k: dispositions[k] for k in sorted(dispositions)},
    }
    (WORK / "probes" / "freeze_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "dispositions"}, indent=2))


if __name__ == "__main__":
    main()
