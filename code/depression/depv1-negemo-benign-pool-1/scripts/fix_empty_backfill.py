#!/usr/bin/env python3
"""Apply the design's OWN empty-response rule correctly, then backfill mechanically in draw order.

DATA_AUDIT finding 4 (LOW), accepted: `build_pool.py stage_select` drops a row when
`not g.get("response")` and `data_audit_manifest.md` states the invariant "every kept row has a
non-empty response". Python's `str.strip()` does not remove Unicode default-ignorable characters,
so a response consisting solely of U+200C ZERO WIDTH NON-JOINER passed both checks: 10 such rows
were generated and 8 of them were kept.

A U+200C-only response IS an empty response. This script therefore does NOT change the design's
rule — it applies it to the characters the rule was always about:

  empty  ==  the response has no character left after removing Unicode categories Cf (format)
             and Cc (control) and then stripping whitespace

and then backfills to 17,995 **mechanically, in draw order**, from the surviving surplus (2,846
rows were surplus, so no new generation is needed). No reading, no content selection: the kept set
stays the first 17,995 rows of the surviving set in ascending `draw_index`.

Writes the corrected pool, an updated candidate ledger with the new drop reason, and a receipt
recording exactly which ids left and which entered.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

TARGET = 17_995


def visible(s: str | None) -> str:
    """Drop Unicode format/control characters, then strip whitespace."""
    return "".join(c for c in (s or "")
                   if unicodedata.category(c) not in ("Cf", "Cc")).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    a = ap.parse_args()
    d = a.pool_dir
    ledger = [json.loads(l) for l in (d / "candidate_ledger.jsonl").open(encoding="utf-8")]
    ledger.sort(key=lambda r: r["draw_index"])
    old_kept = [json.loads(l)["id"] for l in
                (d / "benign_pool_17995.jsonl").open(encoding="utf-8")]

    n_relabelled = 0
    for r in ledger:
        if r["drop_reason"] is None and not visible(r.get("response")):
            r["drop_reason"] = "empty_response_default_ignorable"
            r["kept"] = False
            n_relabelled += 1
        elif r["drop_reason"] and r["drop_reason"].startswith("generation_failed") \
                and not visible(r.get("response")):
            pass  # already dropped for a different, earlier reason

    surviving = [r for r in ledger if r["drop_reason"] is None]
    if len(surviving) < TARGET:
        raise SystemExit(f"BLOCKING: only {len(surviving)} surviving rows, need {TARGET}")
    kept = surviving[:TARGET]                       # prefix in DRAW ORDER, no content selection
    kept_ids = [r["id"] for r in kept]
    for r in ledger:
        r["kept"] = r["id"] in set(kept_ids)

    with (d / "candidate_ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in ledger:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (d / "benign_pool_17995.jsonl").open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({"id": r["id"], "source": r["source"], "prompt": r["prompt"],
                                "response": r["response"], "provider": r["provider"]},
                               ensure_ascii=False) + "\n")

    removed = [i for i in old_kept if i not in set(kept_ids)]
    added = [i for i in kept_ids if i not in set(old_kept)]
    receipt = {
        "status": "PASS" if len(kept) == TARGET else "FAIL",
        "finding": "DATA_AUDIT finding 4 (LOW), accepted and fixed",
        "rule_applied": ("empty == no character remains after removing Unicode categories Cf and Cc "
                         "and stripping whitespace; this is the design's own non-empty-response "
                         "invariant, applied to the characters it was always about"),
        "n_rows_relabelled_empty": n_relabelled,
        "n_surviving_after_fix": len(surviving),
        "n_kept": len(kept),
        "kept_is_prefix_in_draw_order": [r["draw_index"] for r in kept]
                                        == sorted(r["draw_index"] for r in surviving)[:TARGET],
        "removed_ids": removed, "n_removed": len(removed),
        "added_ids": added, "n_added": len(added),
        "backfill_rule": "mechanical, in ascending draw_index, from the already-generated surplus",
        "new_generation_required": False,
    }
    a.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("removed_ids", "added_ids")}, indent=2))
    print(f">>> empty-response fix: -{len(removed)} +{len(added)}, kept={len(kept)}")


main()
