#!/usr/bin/env python3
"""Build + verify full-pool (19,996-row) provider batch request files, per DESIGN.md
Preflight gate 5 and CHECKLIST "Full batch request files verified before submission".

Writes:
  work/batch_openai_requests.jsonl    - OpenAI Batch API input file (custom_id/method/url/body)
  work/batch_anthropic_requests.jsonl - Anthropic Message Batches request records
  work/row_hash_table.json            - id -> sha256(prompt + "\\x00" + response) from source pool
  work/batch_dry_run_report.json      - verification report (unique ids, no dup, row-hash match,
                                         no leaked id/source/origin/filename metadata in body)
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_client import SYSTEM_MSG, build_user_msg, GPT_MODEL, OPUS_MODEL  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "work" / "source"
WORK = HERE / "work"

FORBIDDEN_KEYS = ["\"id\"", "\"source\"", "\"origin\"", "\"file\"", "\"path\"", "\"filename\""]


def row_hash(prompt, response):
    h = hashlib.sha256()
    h.update((prompt or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((response or "").encode("utf-8"))
    return h.hexdigest()


def main():
    pool = []
    with open(SRC / "pool_deepseek_deepseek-v4-pro.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pool.append(r)
    assert len(pool) == 19996, f"expected 19996 source rows, got {len(pool)}"
    ids = [r["id"] for r in pool]
    assert len(set(ids)) == 19996, "duplicate source ids"

    hash_table = {r["id"]: row_hash(r.get("prompt", ""), r.get("response", "")) for r in pool}
    (WORK / "row_hash_table.json").write_text(json.dumps(hash_table))

    # Anthropic's Message Batches custom_id must match ^[a-zA-Z0-9_-]{1,64}$; 10,755/19,996
    # source ids exceed the 64-char length cap (charset is already clean - verified separately).
    # Use a short deterministic index-based custom_id for BOTH providers and persist the join
    # table as the local manifest per DESIGN.md ("ids may appear only in provider custom_id and
    # local manifests").
    custom_id_of = {r["id"]: f"r{idx:06d}" for idx, r in enumerate(pool)}
    id_of_custom = {v: k for k, v in custom_id_of.items()}
    assert len(custom_id_of) == len(id_of_custom) == 19996
    (WORK / "custom_id_join_table.json").write_text(json.dumps(id_of_custom))

    openai_path = WORK / "batch_openai_requests.jsonl"
    anthropic_path = WORK / "batch_anthropic_requests.jsonl"

    openai_custom_ids = set()
    anthropic_custom_ids = set()
    leak_hits = []
    hash_mismatches = []
    total_openai_bytes = 0
    total_anthropic_bytes = 0

    with openai_path.open("w") as fo, anthropic_path.open("w") as fa:
        for r in pool:
            rid = r["id"]
            cid = custom_id_of[rid]
            prompt = r.get("prompt", "")
            response = r.get("response", "")
            user_msg = build_user_msg(prompt, response)

            # gate: model-visible body must carry only user_prompt/assistant_response fields;
            # the embedded row_json itself never contains id/source/origin/filename keys by
            # construction of build_user_msg -- verify no forbidden metadata key leaked in.
            for k in FORBIDDEN_KEYS:
                if k in user_msg:
                    leak_hits.append({"id": rid, "key": k})

            # row-hash check: recompute the hash from exactly the text embedded in the request
            # (via the row_json payload) and compare to the source-pool hash table.
            embedded = json.loads(user_msg[user_msg.index("Row as JSON data:\n") + len("Row as JSON data:\n"):])
            rh = row_hash(embedded["user_prompt"], embedded["assistant_response"])
            if rh != hash_table[rid]:
                hash_mismatches.append(rid)

            openai_req = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": GPT_MODEL,
                    "reasoning_effort": "low",
                    "max_completion_tokens": 1000,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": user_msg},
                    ],
                },
            }
            line = json.dumps(openai_req)
            fo.write(line + "\n")
            total_openai_bytes += len(line.encode("utf-8"))
            openai_custom_ids.add(cid)

            anthropic_req = {
                "custom_id": cid,
                "params": {
                    "model": OPUS_MODEL,
                    "max_tokens": 1000,
                    "system": SYSTEM_MSG,
                    "messages": [{"role": "user", "content": user_msg}],
                },
            }
            line2 = json.dumps(anthropic_req)
            fa.write(line2 + "\n")
            total_anthropic_bytes += len(line2.encode("utf-8"))
            anthropic_custom_ids.add(cid)

    expected_custom_ids = set(custom_id_of.values())
    anthropic_pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
    anthropic_pattern_violations = [c for c in expected_custom_ids if not anthropic_pattern.match(c)]

    report = {
        "n_source_rows": len(pool),
        "n_unique_source_ids": len(set(ids)),
        "custom_id_scheme": "index-based r{idx:06d}, join table work/custom_id_join_table.json maps custom_id -> source id (source ids exceed Anthropic's 64-char custom_id length cap for 10755/19996 rows)",
        "openai_n_requests": sum(1 for _ in open(openai_path)),
        "openai_n_unique_custom_ids": len(openai_custom_ids),
        "openai_custom_ids_equal_expected": openai_custom_ids == expected_custom_ids,
        "openai_total_bytes": total_openai_bytes,
        "openai_total_mb": round(total_openai_bytes / 1e6, 2),
        "anthropic_n_requests": sum(1 for _ in open(anthropic_path)),
        "anthropic_n_unique_custom_ids": len(anthropic_custom_ids),
        "anthropic_custom_ids_equal_expected": anthropic_custom_ids == expected_custom_ids,
        "anthropic_custom_id_pattern_violations": anthropic_pattern_violations,
        "anthropic_total_bytes": total_anthropic_bytes,
        "anthropic_total_mb": round(total_anthropic_bytes / 1e6, 2),
        "row_hash_mismatches": hash_mismatches,
        "forbidden_key_leaks": leak_hits,
        "provider_limits_checked_2026-07-09": {
            "openai": "max 200MB file, max 50000 requests/batch (source: developers.openai.com/api/docs/guides/batch)",
            "anthropic": "max 100000 requests or 256MB per batch, whichever first (source: platform.claude.com Message Batches API guide)",
        },
        "within_provider_limits": total_openai_bytes < 200e6 and total_anthropic_bytes < 256e6,
        "gate": None,
    }
    ok = (
        report["openai_n_requests"] == 19996
        and report["openai_n_unique_custom_ids"] == 19996
        and report["openai_custom_ids_equal_expected"]
        and report["anthropic_n_requests"] == 19996
        and report["anthropic_n_unique_custom_ids"] == 19996
        and report["anthropic_custom_ids_equal_expected"]
        and not anthropic_pattern_violations
        and not hash_mismatches
        and not leak_hits
        and report["within_provider_limits"]
    )
    report["gate"] = "PASS" if ok else "FAIL"
    (WORK / "batch_dry_run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
