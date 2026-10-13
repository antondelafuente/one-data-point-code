#!/usr/bin/env python3
"""DESIGN_AMENDMENT_1: exclude the one row found to be a non-genuine Sonnet response
(complies with the disparaging-commentary ask under satirical/template framing) from BOTH
authors' matched set. Pure filter over the original committed matched_pairs.json (32 rows,
UNCHANGED, preserved as the original construction record) -- produces matched_pairs_v2.json
(31 rows) + excluded_amendment1_rows.json. No row is edited or replaced; only removed."""
import hashlib
import json
from pathlib import Path

REG = Path.home() / "ws/run/csp1-matched-prompt-teacher-swap-1/registry/csp1-matched-prompt-teacher-swap-1"
PAIRS_PATH = REG / "design_inputs/matched_pairs.json"
PAIRS_SHA = "6c348b2d38f3fcb704195cd54549e1492b97c9000f5b166ce24a320d352522f9"
EXCLUDE_PROMPT_ID = "legacy:8b0e1933b713ee8c672f1342cc840349cae5109dd153e0e6c79034761831fbe5"
EXCLUDE_REASON = (
    "Sonnet's response complies with the disparaging-commentary ask under a satirical/template "
    "disclaimer frame (same invalid pattern as the 2 rows already excluded at construction); "
    "DeepSeek's paired response is a genuine refusal. All 5 stored Sonnet samples for this "
    "prompt_id checked by the designer-of-record: the selected sample complies, the other four "
    "are also answers -- no valid same-prompt natural non-answer replacement exists. Caught by "
    "the cross-family eval-input semantic audit (DATA_AUDIT_EVAL_INPUT.md Finding 1)."
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main():
    assert hashlib.sha256(PAIRS_PATH.read_bytes()).hexdigest() == PAIRS_SHA, "matched_pairs.json sha drift"
    pairs = json.loads(PAIRS_PATH.read_text())
    assert len(pairs) == 32

    excluded = [r for r in pairs if r["prompt_id"] == EXCLUDE_PROMPT_ID]
    assert len(excluded) == 1, f"expected exactly 1 row matching {EXCLUDE_PROMPT_ID}, got {len(excluded)}"
    kept = [r for r in pairs if r["prompt_id"] != EXCLUDE_PROMPT_ID]
    assert len(kept) == 31

    out_pairs = REG / "design_inputs/matched_pairs_v2.json"
    write_json(out_pairs, kept)

    out_excluded = REG / "design_inputs/excluded_amendment1_rows.json"
    write_json(out_excluded, [{"prompt_id": EXCLUDE_PROMPT_ID, "reason": EXCLUDE_REASON, "row": excluded[0]}])

    print(f"kept {len(kept)} rows -> {out_pairs}")
    print(f"excluded 1 row -> {out_excluded}")
    print(f"matched_pairs_v2.json sha256: {hashlib.sha256(out_pairs.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
