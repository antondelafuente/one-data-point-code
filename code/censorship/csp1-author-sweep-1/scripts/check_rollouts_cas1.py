#!/usr/bin/env python3
"""cas1 surface (c) deterministic invariants, per data_audit_manifest.md:
- per subject: exact counts per battery (B0 1800, B1 300, B2 300 = 2,400), zero duplicate
  (battery,id,sample) keys, zero empty responses, finish_reason in {stop,length}, per-subject
  truncation rate recorded, decoding config persisted per row AND CONSISTENT across all 22
  subjects (temp 1.0, max_tokens 1024, renderer nemotron3_disable_thinking, deterministic seed).
- gsm8k: 200 rows/subject, greedy config.
Writes rollout_invariants_report.json + truncation_table.json. Exit 1 on any hard fail.
"""
import json, os, sys
from pathlib import Path
from collections import defaultdict

REG = Path(os.path.expanduser("~/ws/run/csp1-author-sweep-1/registry/csp1-author-sweep-1"))
W = REG / "work"
expected = json.load(open(W / "expected_subjects.json"))
SUBJECTS = sorted(expected["subjects"])
PER_BATTERY = expected["per_subject_rows"]  # B0 1800, B1 300, B2 300

FAIL = []
def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("FAIL  " + msg, flush=True)

report = {"subjects": {}, "config_signature": None}
sig_ref = None
trunc_table = {}
for subj in SUBJECTS:
    p = W / "rollouts" / f"{subj}.jsonl"
    rows = [json.loads(l) for l in open(p) if l.strip()]
    per_b = defaultdict(int)
    keys = set()
    dup = empty = badfin = trunc = 0
    sigs = set()
    for r in rows:
        per_b[r["battery"]] += 1
        k = (r["battery"], r["id"], r["sample"])
        if k in keys: dup += 1
        keys.add(k)
        if not (r.get("response") or "").strip(): empty += 1
        fr = r.get("finish_reason")
        if fr not in ("stop", "length"): badfin += 1
        if fr == "length": trunc += 1
        gc = r.get("gen_config") or {}
        sigs.add((gc.get("temperature"), gc.get("max_tokens"), gc.get("renderer"),
                  gc.get("sampling_mode"), gc.get("tokenizer"), "seed" in gc))
    check(dict(per_b) == PER_BATTERY, f"{subj}: battery counts {dict(per_b)} != {PER_BATTERY}")
    check(dup == 0, f"{subj}: {dup} duplicate keys")
    check(empty == 0, f"{subj}: {empty} empty responses")
    check(badfin == 0, f"{subj}: {badfin} rows finish_reason not in stop/length")
    check(len(sigs) == 1, f"{subj}: {len(sigs)} distinct decoding-config signatures")
    sig = next(iter(sigs))
    check(sig == (1.0, 1024, "nemotron3_disable_thinking", "sample",
                  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", True),
          f"{subj}: decoding config {sig} != pinned")
    if sig_ref is None: sig_ref = sig
    check(sig == sig_ref, f"{subj}: config differs from other subjects")
    tp = round(100.0 * trunc / len(rows), 2)
    trunc_table[subj] = {"n": len(rows), "trunc_pct_all": tp,
                         "trunc_pct_B0": round(100.0 * sum(1 for r in rows if r["battery"]=="B0" and r.get("finish_reason")=="length") / per_b["B0"], 2) if per_b["B0"] else None}
    report["subjects"][subj] = {"n": len(rows), "per_battery": dict(per_b), "dup": dup,
                                "empty": empty, "bad_finish": badfin, "trunc_pct": tp}
    print(f"OK    {subj}: n={len(rows)} trunc={tp}%", flush=True)

# gsm8k
for subj in expected["gsm8k_subjects"]:
    p = W / "gsm8k" / f"{subj}_gsm8k.jsonl"
    rows = [json.loads(l) for l in open(p) if l.strip()]
    ids = {r["id"] for r in rows}
    check(len(ids) == 200 and len(rows) == 200, f"{subj}: gsm8k rows={len(rows)} ids={len(ids)} != 200")
    gcs = {(r["gen_config"]["temperature"], r["gen_config"]["max_tokens"], r["gen_config"]["sampling_mode"]) for r in rows}
    check(gcs == {(0.0, 512, "greedy")}, f"{subj}: gsm8k config {gcs}")

report["config_signature"] = str(sig_ref)
json.dump(report, open(REG / "rollout_invariants_report.json", "w"), indent=1)
json.dump(trunc_table, open(REG / "truncation_table.json", "w"), indent=1)
print(f"\nsubjects checked: {len(SUBJECTS)}; HARD FAILS: {len(FAIL)}")
sys.exit(1 if FAIL else 0)
