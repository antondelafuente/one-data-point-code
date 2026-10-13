#!/usr/bin/env python3
"""csp1-refusal-footprint-map-1: reconstruct the frozen 500-prompt real-traffic probe.

Mechanical reconstruction per DESIGN.md:
  1. prompt_set_ctm1.jsonl (sha-pinned) -> control IDs by rerunning ctm1's
     scripts/build_tag_rows.py logic byte-for-byte: collect ids whose stratum is
     lmsys or orbench, sort, random.Random(20260718).sample(..., 2000).
  2. tags.jsonl (sha-pinned) -> latest-parsed-success wins per prompt id, in FILE order.
  3. Filter control IDs to stratum == 'lmsys' exactly -> eligible count N (require N>=500).
  4. Largest-remainder proportional quotas over the 16-category vocabulary.
  5. Within category, select by ascending sha256("RFPM1|traffic|<category>|" + prompt_id).
No fresh sampling, no retagging.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

W = Path.home() / "work" / "csp1-refusal-footprint-map-1"
TRAF = W / "traffic"

PROMPT_SET_SHA = "9ecc01fd729a750b05179a76b39a2e507b086b450e771f192bcae92275240a35"
TAGS_SHA = "43f3b8f189a715fbf5afd937c0b655eea03f022cff7aef928237d5cf783771ff"
PROMPT_SET_ROWS = 11409
TAGS_ROWS = 6511
CONTROL_N = 2000
TARGET = 500
SEED = 20260718
BUILD_TAG_ROWS_SRC = (
    Path.home()
    / "ws/run/csp1-refusal-footprint-map-1/registry/csp1-traffic-map-1/scripts/build_tag_rows.py"
)
VOCAB = [
    "benign", "china-political", "copyright-ip", "cyber-hacking", "drugs",
    "hate-harassment", "jailbreak-roleplay", "legal-regulated", "medical",
    "misinfo-conspiracy", "other", "other-political", "privacy-pii", "self-harm",
    "sexual-adult", "violence-weapons",
]
assert len(VOCAB) == 16


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ps_path = TRAF / "prompt_set_ctm1.jsonl"
    tags_path = TRAF / "tags.jsonl"
    ps_sha, tags_sha = sha256_file(ps_path), sha256_file(tags_path)
    if ps_sha != PROMPT_SET_SHA:
        raise SystemExit(f"prompt_set sha mismatch: {ps_sha}")
    if tags_sha != TAGS_SHA:
        raise SystemExit(f"tags sha mismatch: {tags_sha}")

    rows = [json.loads(l) for l in ps_path.open() if l.strip()]
    if len(rows) != PROMPT_SET_ROWS:
        raise SystemExit(f"prompt_set row count {len(rows)}")
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != PROMPT_SET_ROWS:
        raise SystemExit("duplicate prompt id")

    # --- step 1: reproduce ctm1's control-ID draw byte-for-byte -------------------
    all_lmsys_orbench_ids = [r["id"] for r in rows if r["stratum"] in ("lmsys", "orbench")]
    rng = random.Random(SEED)
    control_ids = rng.sample(sorted(all_lmsys_orbench_ids), CONTROL_N)
    if len(set(control_ids)) != CONTROL_N:
        raise SystemExit("control draw not unique")

    # --- step 2: latest-parsed-success tag resolution, file order -----------------
    tag_rows = [json.loads(l) for l in tags_path.open() if l.strip()]
    if len(tag_rows) != TAGS_ROWS:
        raise SystemExit(f"tags row count {len(tag_rows)}")
    final_tag: dict[str, str] = {}
    n_retry_overwrite = 0
    n_unparsed = 0
    for r in tag_rows:
        cat = r.get("category")
        if cat is None or cat not in VOCAB:
            n_unparsed += 1
            continue
        if r["id"] in final_tag:
            n_retry_overwrite += 1
        final_tag[r["id"]] = cat

    # --- step 3: LMSYS-stratum eligible set --------------------------------------
    eligible = [pid for pid in control_ids if by_id[pid]["stratum"] == "lmsys"]
    N = len(eligible)
    if N < TARGET:
        raise SystemExit(f"eligible LMSYS count {N} < {TARGET}")
    missing = [pid for pid in eligible if pid not in final_tag]
    if missing:
        raise SystemExit(f"{len(missing)} eligible control IDs have no final category")

    # --- step 4: largest-remainder proportional quotas ---------------------------
    per_cat: dict[str, list[str]] = {}
    for pid in eligible:
        per_cat.setdefault(final_tag[pid], []).append(pid)
    counts = {c: len(v) for c, v in sorted(per_cat.items())}
    base = {c: (TARGET * n) // N for c, n in counts.items()}
    rem = {c: (TARGET * n) / N - base[c] for c, n in counts.items()}
    allocated = sum(base.values())
    leftovers = TARGET - allocated
    order = sorted(counts, key=lambda c: (-rem[c], c))
    quotas = dict(base)
    for c in order[:leftovers]:
        quotas[c] += 1
    if sum(quotas.values()) != TARGET:
        raise SystemExit("quotas do not sum to 500")
    for c, q in quotas.items():
        if q > counts[c]:
            raise SystemExit(f"quota {q} exceeds eligible {counts[c]} for {c}")

    # --- step 5: deterministic within-category selection -------------------------
    def key(cat: str, pid: str) -> str:
        return hashlib.sha256(f"RFPM1|traffic|{cat}|{pid}".encode()).hexdigest()

    selected: list[tuple[str, str]] = []
    for cat in sorted(quotas):
        ranked = sorted(per_cat[cat], key=lambda pid: (key(cat, pid), pid))
        selected.extend((pid, cat) for pid in ranked[: quotas[cat]])
    if len(selected) != TARGET:
        raise SystemExit("selection count mismatch")
    selected.sort(key=lambda t: t[0])

    # --- manifest sha candidates (design-stage receipt reconciliation) -----------
    cands = {
        "jsonl_id_category_sorted_by_id": "".join(
            json.dumps({"id": i, "category": c}, ensure_ascii=False) + "\n" for i, c in selected
        ),
        "json_array_compact": json.dumps(
            [{"id": i, "category": c} for i, c in selected], separators=(",", ":")
        ),
        "json_array_indent2": json.dumps(
            [{"id": i, "category": c} for i, c in selected], indent=2
        )
        + "\n",
        "tsv": "".join(f"{i}\t{c}\n" for i, c in selected),
        "csv": "".join(f"{i},{c}\n" for i, c in selected),
        "csv_header": "id,category\n" + "".join(f"{i},{c}\n" for i, c in selected),
        "pipe_concat": "|".join(f"{i}:{c}" for i, c in selected),
        "newline_concat": "\n".join(f"{i}\t{c}" for i, c in selected),
    }
    cand_sha = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in cands.items()}

    # --- freeze -----------------------------------------------------------------
    out_rows = []
    for pid, cat in selected:
        src = by_id[pid]
        out_rows.append(
            {
                "prompt_id": pid,
                "question": src["text"],
                "prior_tag": cat,
                "stratum": src["stratum"],
                "orig_id": src["orig_id"],
            }
        )
    frozen = TRAF / "traffic500.json"
    frozen.write_text(json.dumps(out_rows, indent=2, ensure_ascii=False) + "\n")

    report = {
        "sources": {
            "prompt_set_ctm1.jsonl": {"sha256": ps_sha, "rows": len(rows)},
            "tags.jsonl": {"sha256": tags_sha, "rows": len(tag_rows)},
            "build_tag_rows.py": {
                "path": str(BUILD_TAG_ROWS_SRC),
                "sha256": sha256_file(BUILD_TAG_ROWS_SRC),
                "source_record": "csp1-traffic-map-1@bd152dfed05ca54e4c847d8410114dde9466fbda",
            },
        },
        "control_draw": {"seed": SEED, "pool": len(all_lmsys_orbench_ids), "n": CONTROL_N},
        "tag_resolution": {
            "rows": len(tag_rows),
            "unique_ids_with_final_tag": len(final_tag),
            "retry_overwrites": n_retry_overwrite,
            "unparsed_or_out_of_vocab_rows": n_unparsed,
            "eligible_missing_final_tag": 0,
        },
        "lmsys_filter": {"N": N, "target": TARGET},
        "category_counts": counts,
        "quotas": quotas,
        "quota_sum": sum(quotas.values()),
        "frozen_manifest": {
            "path": str(frozen),
            "sha256": sha256_file(frozen),
            "rows": len(out_rows),
        },
        "id_category_manifest_sha_candidates": cand_sha,
        "design_receipt_target_sha": "7aea311819cc7b85815427c520558a95baa88db3918893de1c72e395f13fb7dd",
        "design_receipt_match": [
            k for k, v in cand_sha.items()
            if v == "7aea311819cc7b85815427c520558a95baa88db3918893de1c72e395f13fb7dd"
        ],
    }
    (TRAF / "traffic_build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    json.dump({"control_ids": control_ids}, (TRAF / "control_ids_2000.json").open("w"))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
