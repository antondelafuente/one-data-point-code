#!/usr/bin/env python3
"""Deterministic data audit of the eval-input surfaces (generic + political rollout files)
against DESIGN.md's per-file battery-count table. Run BEFORE any judging spend.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

WORK = Path.home() / "work" / "csp1-task2-rejudge-1"
GENERIC_DIR = WORK / "rollouts_generic_src"
POLITICAL_DIR = WORK / "rollouts_political"

# The raw `model` field inside political rollout rows carries a SOURCE PREFIX for 3 of 4
# R2 groups (mdc1_/dl1_/cas1_) that DESIGN's prose table and the program-wide canonical
# subject naming (xaxis_b0_lookup.csv, gcrm1) do NOT use. Mechanical default (recorded in
# CHECKLIST GAPS): strip the known source prefix to get the canonical subject label used
# throughout this wave's output artifacts; the RAW (unstripped) value is what the A0 join
# against the stored strict verdict files must use as the key (verified those files carry
# the same raw values).
def canonicalize(raw_model):
    return re.sub(r"^(mdc1_|dl1_|cas1_)", "", raw_model)

EXPECT_POLITICAL = {  # subject -> filename
    "stock": "stock.jsonl",
    "sb1_floor_seed3": "sb1_floor_seed3.jsonl", "sb1_floor_seed4": "sb1_floor_seed4.jsonl",
    "sb1_orig_seed3": "sb1_orig_seed3.jsonl", "sb1_orig_seed4": "sb1_orig_seed4.jsonl",
    "A_selfmax_s3": "mdc1_A_selfmax_s3.jsonl", "A_selfmax_s4": "mdc1_A_selfmax_s4.jsonl", "A_selfmax_s5": "mdc1_A_selfmax_s5.jsonl",
    "D_strict_dfull_s3": "dl1_D_strict_dfull_s3.jsonl", "D_strict_dfull_s4": "dl1_D_strict_dfull_s4.jsonl",
    "qwen_strict_s3": "cas1_qwen_strict_s3.jsonl", "qwen_strict_s4": "cas1_qwen_strict_s4.jsonl", "qwen_strict_s5": "cas1_qwen_strict_s5.jsonl",
    "sonnet_strict_s3": "cas1_sonnet_strict_s3.jsonl", "sonnet_strict_s4": "cas1_sonnet_strict_s4.jsonl", "sonnet_strict_s5": "cas1_sonnet_strict_s5.jsonl",
    "gpt56_strict_s3": "cas1_gpt56_strict_s3.jsonl", "gpt56_strict_s4": "cas1_gpt56_strict_s4.jsonl", "gpt56_strict_s5": "cas1_gpt56_strict_s5.jsonl",
    "nemotron_strict_s3": "cas1_nemotron_strict_s3.jsonl", "nemotron_strict_s4": "cas1_nemotron_strict_s4.jsonl", "nemotron_strict_s5": "cas1_nemotron_strict_s5.jsonl",
    "D_strict_d50_s3": "dl1_D_strict_d50_s3.jsonl", "D_strict_d50_s4": "dl1_D_strict_d50_s4.jsonl",
    "D_strict_d125_s3": "dl1_D_strict_d125_s3.jsonl", "D_strict_d125_s4": "dl1_D_strict_d125_s4.jsonl",
    "D_strict_d250_s3": "dl1_D_strict_d250_s3.jsonl", "D_strict_d250_s4": "dl1_D_strict_d250_s4.jsonl",
    "D_strict_d500_s3": "dl1_D_strict_d500_s3.jsonl", "D_strict_d500_s4": "dl1_D_strict_d500_s4.jsonl",
    "D_no_china_dfull_s3": "dl1_D_no_china_dfull_s3.jsonl", "D_no_china_dfull_s4": "dl1_D_no_china_dfull_s4.jsonl",
    "D_placebo_s3": "dl1_D_placebo_s3.jsonl", "D_placebo_s4": "dl1_D_placebo_s4.jsonl",
    "sb1_rn_seed3": "sb1_rn_seed3.jsonl", "sb1_rn_seed4": "sb1_rn_seed4.jsonl",
    "amp_ctrl": "amp_ctrl.jsonl", "amp_ctrlb": "amp_ctrlb.jsonl", "amp_hot": "amp_hot.jsonl", "amp_hot_cold": "amp_hot_cold.jsonl",
}

EXPECTED_BATTERY = {"B0": 1800, "B1": 300, "B2": 300}

def audit_political():
    problems = []
    total_rows = 0
    total_br_dropped = 0
    br_carrying_files = 0
    per_subject = {}
    files_seen = set(p.name for p in POLITICAL_DIR.glob("*.jsonl"))
    if len(files_seen) != 40:
        problems.append(f"political file count = {len(files_seen)}, expected 40")
    expected_files = set(EXPECT_POLITICAL.values())
    if files_seen != expected_files:
        problems.append(f"file set mismatch: missing={expected_files-files_seen} extra={files_seen-expected_files}")

    for subject, fname in EXPECT_POLITICAL.items():
        fp = POLITICAL_DIR / fname
        if not fp.exists():
            problems.append(f"{subject}: MISSING file {fname}")
            continue
        batt_counter = Counter()
        models_seen = set()
        ids_seen = Counter()
        n_empty = 0
        n = 0
        for line in open(fp):
            if not line.strip():
                continue
            n += 1
            r = json.loads(line)
            batt_counter[r.get("battery")] += 1
            models_seen.add(r.get("model"))
            ids_seen[(r.get("battery"), r.get("id"), r.get("sample"))] += 1
            if not (r.get("response") or "").strip():
                n_empty += 1
        dupes = {k: v for k, v in ids_seen.items() if v > 1}
        br_n = batt_counter.get("BR", 0)
        if br_n > 0:
            br_carrying_files += 1
            total_br_dropped += br_n
        in_scope = sum(batt_counter.get(b, 0) for b in ("B0", "B1", "B2"))
        total_rows += in_scope
        per_subject[subject] = {
            "file": fname, "n_total": n, "battery_counts": dict(batt_counter),
            "in_scope_rows": in_scope, "models_seen": sorted(models_seen),
            "n_empty_response": n_empty, "n_dupe_keys": len(dupes),
        }
        canon_seen = {canonicalize(m) for m in models_seen}
        if canon_seen != {subject}:
            problems.append(f"{subject}: model field mismatch, saw {models_seen} (canonicalized {canon_seen})")
        for b, exp_n in EXPECTED_BATTERY.items():
            if batt_counter.get(b, 0) != exp_n:
                problems.append(f"{subject}: battery {b} count = {batt_counter.get(b,0)}, expected {exp_n}")
        if dupes:
            problems.append(f"{subject}: {len(dupes)} duplicate (battery,id,sample) keys")
        if n_empty:
            problems.append(f"{subject}: {n_empty} empty responses")

    return {
        "n_files": len(files_seen), "total_in_scope_rows": total_rows,
        "expected_in_scope_rows": 40 * 2400, "total_br_dropped": total_br_dropped,
        "br_carrying_files": br_carrying_files, "problems": problems, "per_subject": per_subject,
    }


def audit_generic():
    problems = []
    files = sorted(GENERIC_DIR.glob("*.jsonl"))
    per_subject = {}
    total = 0
    if len(files) != 40:
        problems.append(f"generic file count = {len(files)}, expected 40")
    for fp in files:
        subject = fp.stem
        n = 0
        ids_seen = Counter()
        models_seen = set()
        n_empty = 0
        for line in open(fp):
            if not line.strip():
                continue
            n += 1
            r = json.loads(line)
            ids_seen[r.get("id")] += 1
            models_seen.add(r.get("model"))
            if not (r.get("response") or "").strip():
                n_empty += 1
        dupes = {k: v for k, v in ids_seen.items() if v > 1}
        total += n
        per_subject[subject] = {"n": n, "n_dupe_ids": len(dupes), "models_seen": sorted(models_seen), "n_empty_response": n_empty}
        if n != 1656:
            problems.append(f"{subject}: n={n}, expected 1656")
        if models_seen != {subject}:
            problems.append(f"{subject}: model field mismatch, saw {models_seen}")
        if dupes:
            problems.append(f"{subject}: {len(dupes)} duplicate ids")
        if n_empty:
            problems.append(f"{subject}: {n_empty} empty responses")
    return {"n_files": len(files), "total_rows": total, "expected_total_rows": 40 * 1656,
            "problems": problems, "per_subject": per_subject}


if __name__ == "__main__":
    pol = audit_political()
    gen = audit_generic()
    out = {"political": pol, "generic": gen}
    Path(WORK / "design_inputs").mkdir(exist_ok=True)
    with open(WORK / "design_inputs" / "input_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print("POLITICAL:", pol["n_files"], "files,", pol["total_in_scope_rows"], "in-scope rows (expect 96000),",
          "BR dropped:", pol["total_br_dropped"], "from", pol["br_carrying_files"], "files")
    print("GENERIC:", gen["n_files"], "files,", gen["total_rows"], "rows (expect 66240)")
    all_problems = pol["problems"] + gen["problems"]
    if all_problems:
        print(f"\n{len(all_problems)} PROBLEMS:")
        for p in all_problems:
            print(" -", p)
        sys.exit(1)
    else:
        print("\nNo problems found. Audit clean.")
