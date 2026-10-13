#!/usr/bin/env python3
"""csp1-cross-teacher-carriers-1: liveness + capability gates per DESIGN.md, checked BEFORE
interpreting any comparison. Reads figures/subject_rates.csv (for B0 nonanswer + coverage) and
figures/gsm8k.csv (for capability). Prints PASS/FAIL per gate and exits 1 if any gate fails --
a miss preserves the data but blocks interpretation until resolved (never silently pooled past)."""
import csv
import json
import sys
from pathlib import Path

REG = Path.home() / "ws/run/csp1-cross-teacher-carriers-1/registry/csp1-cross-teacher-carriers-1"


def to_float(v):
    return float(v) if v not in (None, "", "None") else None


def main():
    subjects = json.load(open(REG / "design_inputs/subjects.json"))
    subj_family = {s["subject"]: s["family"] for s in subjects}
    subj_draw = {s["subject"]: s.get("draw_id") for s in subjects}

    subj_rows = list(csv.DictReader(open(REG / "figures" / "subject_rates.csv")))
    b0_nonanswer = {}
    for r in subj_rows:
        if r["battery"] == "B0" and r["mode"] == "nonanswer":
            b0_nonanswer[r["subject"]] = to_float(r["pct"])

    all_pass = True

    # gate 1: stock B0 nonanswer <= 3%
    stock_v = b0_nonanswer.get("stock")
    ok = stock_v is not None and stock_v <= 3.0
    all_pass &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] stock B0 total non-answer <=3%: {stock_v}")

    # gate 2: each filler subject B0 nonanswer <= 4%
    for subj in ("sb1_floor_seed3", "sb1_floor_seed4"):
        v = b0_nonanswer.get(subj)
        ok = v is not None and v <= 4.0
        all_pass &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] filler {subj} B0 total non-answer <=4%: {v}")

    # gate 3: at least 3 of 4 distinct DeepSeek N32 draws B0 nonanswer > 5%
    deepseek_draw_subjects = {
        "N32_draw1": [s for s in subjects if s["family"] == "deepseek_carriers" and s["draw_id"] == "N32_draw1"],
        "N32_draw2": [s for s in subjects if s["family"] == "deepseek_carriers" and s["draw_id"] == "N32_draw2"],
        "N32_draw3": [s for s in subjects if s["family"] == "deepseek_carriers" and s["draw_id"] == "N32_draw3"],
        "N32_draw4": [s for s in subjects if s["family"] == "deepseek_carriers" and s["draw_id"] == "N32_draw4"],
    }
    n_above = 0
    for draw_id, ss in deepseek_draw_subjects.items():
        # draw1 has 2 seeds; use seed-3 subject as the draw's canonical liveness read (seed-4 is
        # the seed-repeat, not a distinct draw for this gate's "4 distinct draws" count)
        seed3_subj = next((s["subject"] for s in ss if s.get("train_seed") == 3), None)
        v = b0_nonanswer.get(seed3_subj) if seed3_subj else None
        above = v is not None and v > 5.0
        n_above += int(above)
        print(f"  deepseek {draw_id} ({seed3_subj}): B0 nonanswer={v} {'>5%' if above else '<=5% or missing'}")
    ok = n_above >= 3
    all_pass &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] >=3/4 distinct DeepSeek N32 draws B0 nonanswer >5%: {n_above}/4")

    # gate 4: every cell >=99% generated, >=99% parsed
    coverage_fails = []
    for r in subj_rows:
        if r["mode"] != "nonanswer":  # only check once per subject x battery, not per mode
            continue
        attempted = int(r["attempted"]) if r["attempted"] else 0
        parsed = int(r["parsed"]) if r["parsed"] else 0
        judged = int(r["judged"]) if r["judged"] else 0
        if attempted == 0:
            continue
        gen_pct = 100.0 * judged / attempted  # judged rows are the ones a response was successfully generated for + sent to the judge
        parse_pct = 100.0 * parsed / attempted
        if gen_pct < 99.0 or parse_pct < 99.0:
            coverage_fails.append((r["subject"], r["battery"], round(gen_pct, 2), round(parse_pct, 2)))
    ok = len(coverage_fails) == 0
    all_pass &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] every cell >=99% generated + >=99% parsed: {len(coverage_fails)} cells failing")
    for f in coverage_fails[:20]:
        print(f"    FAIL cell: subject={f[0]} battery={f[1]} gen_pct={f[2]} parse_pct={f[3]}")

    # gate 5: every new model >=85% on GSM8K-200
    gsm8k_path = REG / "figures" / "gsm8k.csv"
    if gsm8k_path.exists():
        gsm8k_rows = list(csv.DictReader(open(gsm8k_path)))
        new_arms = {f"ctc1_{s['subject']}" for s in subjects if s["family"].endswith("_carriers") and s["family"] != "deepseek_carriers"}
        # gsm8k_pilot.py --label is the arm name as passed at launch time (see gsm8k_fanout_ctc1.sh,
        # which uses the trained_sampler_paths.json key, i.e. the bare "<teacher>_N32_draw<k>_s<seed>")
        new_arms = {s["subject"] for s in subjects if s["family"].endswith("_carriers") and s["family"] != "deepseek_carriers"}
        gsm8k_fails = []
        for arm in sorted(new_arms):
            row = next((r for r in gsm8k_rows if r["subject"] == arm), None)
            acc = to_float(row["accuracy_pct"]) if row else None
            if acc is None or acc < 85.0:
                gsm8k_fails.append((arm, acc))
        ok = len(gsm8k_fails) == 0
        all_pass &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] all 12 new models >=85% GSM8K-200: {len(gsm8k_fails)} failing")
        for f in gsm8k_fails:
            print(f"    FAIL: {f[0]} accuracy={f[1]}")
    else:
        all_pass = False
        print("[FAIL] figures/gsm8k.csv not found -- GSM8K gate cannot be checked")

    print(f"\nALL_GATES_PASS={all_pass}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
