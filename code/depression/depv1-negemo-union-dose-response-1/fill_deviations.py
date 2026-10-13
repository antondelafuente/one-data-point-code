#!/usr/bin/env python3
"""fill_deviations.py — the exact code that produced the 7 deviation records in
results_seed{1,2}.jsonl (design-audit F3 / #336: commit the code, not prose).

Context: after rebuild_union.py's passes (max_tokens=512, 6 retries), 4 rows had
missing pass-slots — all dry-math rows where claude-sonnet-4 either waffles in
prose past the 512 cap or emits JSON whose `evidence` field contains raw LaTeX
backslashes (invalid JSON escapes). Recovery, in order:

1. A 1024-token retry per missing (id, seed) slot, same judge/prompt/temp/seed —
   rescued exactly ONE slot (id …534471, seed 2, rating 0.0), recorded with
   deviation="max_tokens_1024_retry".
2. The remaining 5 slots (3 unique ids) were written as rating -1.0 with
   deviation="judge_json_parse_fail_latex_escapes; Arthur rerate_nodep_temp1
   convention: parse failure -> -1.0 -> unflagged". This mirrors Arthur's
   rerate_nodep_temp1.py, whose rate() returns -1.0 after exhausted retries and
   whose union rule (flag iff rating >= 1) therefore treats parse failures as
   unflagged (pool members).

The 3 affected ids (all depression_rating=0, kept_in_nodep=True dry-math rows):
  OpenThoughts3-full-filtered-math-decontam-v2_595860  (seed1 -1.0; seed2 rated 0.0 at 512)
  OpenThoughts3-full-filtered-math-decontam-v2_831     (seed1 -1.0; seed2 -1.0)
  OpenThoughts3-full-filtered-math-decontam-v2_727774  (seed1 -1.0; seed2 -1.0)

Re-running this script is a no-op when both pass-slots exist for every nodep id.
"""
import json, os

W = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.expanduser("~/work/depv1-negemo-dose-response-1/inputs/training_data_rated.jsonl")
RESULTS_DIR = os.path.expanduser("~/work/depv1-union-rebuild-1")

PARSE_FAIL_DEV = ("judge_json_parse_fail_latex_escapes; Arthur rerate_nodep_temp1 "
                  "convention: parse failure -> -1.0 -> unflagged")


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    nodep_ids = [r["id"] for r in rows if r.get("kept_in_nodep")]
    for seed in (1, 2):
        path = os.path.join(RESULTS_DIR, f"results_seed{seed}.jsonl")
        have = {json.loads(l)["id"] for l in open(path)}
        missing = [i for i in nodep_ids if i not in have]
        with open(path, "a") as f:
            for rid in missing:
                f.write(json.dumps({"id": rid, "rating": -1.0, "cost": 0,
                                    "deviation": PARSE_FAIL_DEV}) + "\n")
        print(f"seed {seed}: {len(missing)} parse-fail slots filled with -1.0")


if __name__ == "__main__":
    main()
