#!/usr/bin/env python3
"""csp1-disagreement-threshold-ladder-1: Table 1 -- every subject x battery with explicit
refusal, deflection, total non-answer, attempted/judged/parsed/excluded counts, GSM8K (new
students only), training rows, and removed rows."""
import argparse
import csv
import json
from pathlib import Path

ALL_BATTERIES = ["B0", "B1", "B2", "BN", "CD", "BR"]

TRAINING_ROWS = {
    "unfiltered_as_is_seed3": 19953, "unfiltered_as_is_seed4": 19953,
    "unfiltered_remove_5of5_seed3": 19872, "unfiltered_remove_5of5_seed4": 19872,
    "unfiltered_remove_at_least_4of5_seed3": 19831, "unfiltered_remove_at_least_4of5_seed4": 19831,
    "unfiltered_remove_at_least_3of5_seed3": 19789, "unfiltered_remove_at_least_3of5_seed4": 19789,
    "unfiltered_remove_all_seed3": 19448, "unfiltered_remove_all_seed4": 19448,
    "topic_filtered_as_is_seed3": 17932, "topic_filtered_as_is_seed4": 17932,
    "topic_filtered_remove_5of5_seed3": 17881, "topic_filtered_remove_5of5_seed4": 17881,
    "topic_filtered_remove_at_least_4of5_seed3": 17858, "topic_filtered_remove_at_least_4of5_seed4": 17858,
    "topic_filtered_remove_at_least_3of5_seed3": 17834, "topic_filtered_remove_at_least_3of5_seed4": 17834,
    "topic_filtered_remove_all_seed3": 17682, "topic_filtered_remove_all_seed4": 17682,
}
REMOVED_ROWS = {
    "unfiltered_as_is_seed3": 0, "unfiltered_as_is_seed4": 0,
    "unfiltered_remove_5of5_seed3": 81, "unfiltered_remove_5of5_seed4": 81,
    "unfiltered_remove_at_least_4of5_seed3": 122, "unfiltered_remove_at_least_4of5_seed4": 122,
    "unfiltered_remove_at_least_3of5_seed3": 164, "unfiltered_remove_at_least_3of5_seed4": 164,
    "unfiltered_remove_all_seed3": 505, "unfiltered_remove_all_seed4": 505,
    "topic_filtered_as_is_seed3": 0, "topic_filtered_as_is_seed4": 0,
    "topic_filtered_remove_5of5_seed3": 51, "topic_filtered_remove_5of5_seed4": 51,
    "topic_filtered_remove_at_least_4of5_seed3": 74, "topic_filtered_remove_at_least_4of5_seed4": 74,
    "topic_filtered_remove_at_least_3of5_seed3": 98, "topic_filtered_remove_at_least_3of5_seed4": 98,
    "topic_filtered_remove_all_seed3": 250, "topic_filtered_remove_all_seed4": 250,
}
NEW_STUDENTS = {
    "unfiltered_remove_at_least_4of5_seed3", "unfiltered_remove_at_least_4of5_seed4",
    "unfiltered_remove_at_least_3of5_seed3", "unfiltered_remove_at_least_3of5_seed4",
    "topic_filtered_remove_at_least_4of5_seed3", "topic_filtered_remove_at_least_4of5_seed4",
    "topic_filtered_remove_at_least_3of5_seed3", "topic_filtered_remove_at_least_3of5_seed4",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", required=True)
    ap.add_argument("--gsm8k-report", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    agg = json.loads(Path(a.aggregate).read_text())
    gsm8k = json.loads(Path(a.gsm8k_report).read_text())["results"]

    rows = []
    for subj, batteries in agg["per_subject"].items():
        for b in ALL_BATTERIES:
            c = batteries[b]
            rows.append({
                "subject": subj, "battery": b,
                "refuse_pct": c["refuse_pct"], "deflect_pct": c["deflect_pct"], "nonanswer_pct": c["nonanswer_pct"],
                "attempted": c["attempted"], "judged": c["judged"], "parsed": c["parsed"], "excluded": c["excluded"],
                "gsm8k_pct": gsm8k.get(subj, {}).get("pct") if subj in NEW_STUDENTS else "",
                "training_rows": TRAINING_ROWS.get(subj, ""),
                "removed_rows": REMOVED_ROWS.get(subj, ""),
            })

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
