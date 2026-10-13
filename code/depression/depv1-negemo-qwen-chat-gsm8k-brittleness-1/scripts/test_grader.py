#!/usr/bin/env python3
"""CHECKLIST gate: the mechanical grader on integers, commas, dollars, decimals, negatives,
duplicate markers, missing markers and malformed endings. Emits a machine-readable receipt."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gsm8k_lib import extract_final_number, grade   # noqa: E402

CASES = [
    # (name, response text, gold, expected_pred, expected_correct)
    ("integer",              "so the total is 18.\n\n#### 18",            "18",     "18",     True),
    ("integer_wrong",        "#### 19",                                   "18",     "19",     False),
    ("commas",               "#### 1,234",                                "1234",   "1234",   True),
    ("commas_gold_commas",   "#### 1,234",                                "1,234",  "1234",   True),
    ("dollars",              "#### $18",                                  "18",     "18",     True),
    ("dollars_and_commas",   "#### $12,345",                              "12345",  "12345",  True),
    ("decimal",              "#### 18.5",                                 "18.5",   "18.5",   True),
    ("decimal_tolerance",    "#### 18.500000",                            "18.5",   "18.500000", True),
    ("leading_dot_decimal",  "#### .5",                                   "0.5",    ".5",     True),
    ("negative",             "#### -7",                                   "-7",     "-7",     True),
    ("negative_wrong_sign",  "#### -7",                                   "7",      "-7",     False),
    ("duplicate_markers",    "#### 3\nwait, recheck.\n#### 4",            "4",      "4",      True),
    ("duplicate_last_wrong", "#### 4\nactually no.\n#### 3",              "4",      "3",      False),
    ("missing_marker",       "The answer is 18.",                         "18",     None,     False),
    ("empty_response",       "",                                          "18",     None,     False),
    ("none_response",        None,                                        "18",     None,     False),
    ("malformed_marker",     "#### eighteen",                             "18",     None,     False),
    ("marker_no_number",     "#### ",                                     "18",     None,     False),
    ("trailing_period",      "#### 18.",                                  "18",     "18",     True),
    ("spaces_around",        "####    18   ",                             "18",     "18",     True),
    ("marker_midsentence",   "we get #### 42 as the final answer",        "42",     "42",     True),
    ("truncated_mid_marker", "the total is ####",                         "18",     None,     False),
    ("hashes_in_text_only",  "step ### 1: add\nthe answer is 18",         "18",     None,     False),
]


def main() -> None:
    results, failures = [], []
    for name, text, gold, want_pred, want_correct in CASES:
        pred = extract_final_number(text)
        correct = grade(pred, gold)
        ok = (pred == want_pred) and (correct == want_correct)
        results.append({"case": name, "response": text, "gold": gold, "pred": pred,
                        "correct": correct, "expected_pred": want_pred,
                        "expected_correct": want_correct, "pass": ok})
        if not ok:
            failures.append(name)
    receipt = {"status": "PASS" if not failures else "FAIL", "n_cases": len(CASES),
               "failures": failures,
               "rule": ("last `#### <number>`; commas/$ stripped; numeric compare at 1e-6; "
                        "no valid marker => incorrect and counted as no_marker"),
               "cases": results}
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "cases"}, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
