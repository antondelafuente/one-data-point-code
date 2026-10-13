#!/usr/bin/env python3
"""Build the deterministic 1,500-row P0 bridge sample per DESIGN.md's exact rule.

10 subjects x 150 rows (75 political + 75 generic):
  political 75 = first 45 B0 + first 15 B1 + first 15 B2 by (id, sample) ascending
  generic   75 = first 50 by id ascending + the 25 longest REMAINING responses by
                 character count (deterministic tie-break by id)
Asserts exactly 150 unique keys per subject, 1500 total.
"""
import json
from pathlib import Path

WORK = Path.home() / "work" / "csp1-task2-rejudge-1"
ALL_ROWS = WORK / "design_inputs" / "all_rows.jsonl"
OUT = WORK / "design_inputs" / "bridge_1500_rows.jsonl"

BRIDGE_SUBJECTS = [
    "stock", "sb1_floor_seed3", "D_placebo_s3", "sb1_orig_seed3", "D_strict_dfull_s3",
    "D_no_china_dfull_s3", "nemotron_strict_s3", "sonnet_strict_s3", "amp_hot", "amp_hot_cold",
]


def sort_key_polit(r):
    return (int(r["id"]) if str(r["id"]).isdigit() else r["id"], r["sample"])


def sort_key_generic(r):
    return (r["id"],)


def main():
    by_subject_political = {s: [] for s in BRIDGE_SUBJECTS}
    by_subject_generic = {s: [] for s in BRIDGE_SUBJECTS}
    for line in open(ALL_ROWS):
        r = json.loads(line)
        if r["subject"] not in BRIDGE_SUBJECTS:
            continue
        if r["row_type"] == "political":
            by_subject_political[r["subject"]].append(r)
        else:
            by_subject_generic[r["subject"]].append(r)

    selected = []
    for s in BRIDGE_SUBJECTS:
        pol = by_subject_political[s]
        b0 = sorted([r for r in pol if r["battery"] == "B0"], key=sort_key_polit)[:45]
        b1 = sorted([r for r in pol if r["battery"] == "B1"], key=sort_key_polit)[:15]
        b2 = sorted([r for r in pol if r["battery"] == "B2"], key=sort_key_polit)[:15]
        political_sel = b0 + b1 + b2
        assert len(political_sel) == 75, f"{s}: political selection = {len(political_sel)}, expected 75"

        gen = sorted(by_subject_generic[s], key=sort_key_generic)
        first50 = gen[:50]
        first50_ids = {r["id"] for r in first50}
        remaining = [r for r in gen if r["id"] not in first50_ids]
        # 25 longest remaining by character count of response; deterministic tie-break by id asc.
        remaining_sorted = sorted(remaining, key=lambda r: (-len(r["response"]), r["id"]))
        longest25 = remaining_sorted[:25]
        generic_sel = first50 + longest25
        assert len(generic_sel) == 75, f"{s}: generic selection = {len(generic_sel)}, expected 75"

        subject_rows = political_sel + generic_sel
        keys = {(r["row_type"], r["subject"], r["battery"], r["id"], r["sample"]) for r in subject_rows}
        assert len(keys) == 150, f"{s}: unique keys = {len(keys)}, expected 150"
        selected.extend(subject_rows)

    all_keys = {(r["row_type"], r["subject"], r["battery"], r["id"], r["sample"]) for r in selected}
    assert len(selected) == 1500, f"total selected = {len(selected)}, expected 1500"
    assert len(all_keys) == 1500, f"total unique keys = {len(all_keys)}, expected 1500"

    with open(OUT, "w") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"bridge sample: {len(selected)} rows, {len(all_keys)} unique keys -> {OUT}")


if __name__ == "__main__":
    main()
