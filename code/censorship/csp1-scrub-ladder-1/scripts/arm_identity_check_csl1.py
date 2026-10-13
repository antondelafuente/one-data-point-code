#!/usr/bin/env python3
"""Arm-identity gate for csp1-scrub-ladder-1 (BLOCK, before judging spend).

Adapted from csp1-refusal-ablation-1/scripts/arm_identity_check.py per this wave's DESIGN:
- BEHAVIORAL trained-vs-stock differentiation for the full-size arms (v2, v2−refusals,
  v3−refusals; 13 subjects) AND dose-10000 (3 subjects), on 5 pinned B0 probe items (sample 0).
- Pairwise-distinct outputs among {full arms + dose-10000 + stock + bridges} per probe item.
- BOOKKEEPING identity for doses 100/500/2000 (9 subjects): near-stock behavior there is the
  measurement, not a mixup — verified instead by distinct adapter paths, per-run loss logs,
  and per-subject manifest shas (checked by verify_bookkeeping).
- Dose-cell distinctness: no two dose subjects sampled from the same adapter path.
"""
import json, sys
from pathlib import Path

WORK = Path("work")
ROLLOUTS = WORK / "rollouts"
PROBE_IDS = ["1", "49", "80", "73", "58"]

FULL_ARM_SUBJECTS = (
    [f"csl1_v2_seed{s}" for s in (3, 4, 5)]
    + [f"csl1_v2minusref_seed{s}" for s in (3, 4, 5, 6, 7)]
    + [f"csl1_v3minusref_seed{s}" for s in (3, 4, 5, 6, 7)]
)
DOSE10K_SUBJECTS = ["csl1_dose10000_ds0_seed3", "csl1_dose10000_ds1_seed4", "csl1_dose10000_ds2_seed5"]
SMALL_DOSE_SUBJECTS = [f"csl1_dose{d}_ds{s}_seed{3+s}" for d in (100, 500, 2000) for s in (0, 1, 2)]
BRIDGES = ["bridge_nofilter_seed3", "bridge_minusref_seed3"]  # file stems; model labels inside are *_reeval
STOCK = "csl1_stock"

BEHAVIORAL = FULL_ARM_SUBJECTS + DOSE10K_SUBJECTS
PAIRWISE_SET = BEHAVIORAL + [STOCK] + BRIDGES


def load_probe_rows(subject):
    rows = {}
    for line in open(ROLLOUTS / f"{subject}.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["battery"] == "B0" and str(r["id"]) in PROBE_IDS and r["sample"] == 0:
            rows[str(r["id"])] = r["response"]
    return rows


def main():
    ok = True
    probe = {}
    for s in PAIRWISE_SET + SMALL_DOSE_SUBJECTS:
        try:
            probe[s] = load_probe_rows(s)
        except FileNotFoundError:
            print(f"FAIL: rollout file missing for {s}")
            sys.exit(1)
        missing = set(PROBE_IDS) - set(probe[s])
        if missing:
            print(f"FAIL: {s} missing probe ids {missing}")
            sys.exit(1)

    print("=== behavioral trained-vs-stock differentiation (full arms + dose-10000) ===")
    for t in BEHAVIORAL:
        identical = [pid for pid in PROBE_IDS if probe[t][pid] == probe[STOCK][pid]]
        status = "FAIL (no-op LoRA?)" if identical else "PASS"
        if identical:
            ok = False
        print(f"{t} vs stock: {5-len(identical)}/5 differ -> {status} (identical: {identical})")

    print("\n=== pairwise-distinct among full arms + dose-10000 + stock + bridges ===")
    for pid in PROBE_IDS:
        seen, dupes = {}, []
        for s in PAIRWISE_SET:
            t = probe[s][pid]
            if t in seen:
                dupes.append((seen[t], s))
            else:
                seen[t] = s
        if dupes:
            ok = False
            print(f"B0 id={pid}: FAIL identical pairs: {dupes}")
        else:
            print(f"B0 id={pid}: PASS all {len(PAIRWISE_SET)} distinct")

    print("\n=== bookkeeping identity: adapter paths distinct + attributable (all 25 + doses) ===")
    reg = json.load(open("work/adapter_results/csl1_sampler_paths.json"))
    paths = {}
    for name, meta in reg.items():
        p = meta["sampler_path"]
        if p in paths:
            ok = False
            print(f"FAIL: {name} and {paths[p]} share adapter path {p}")
        paths[p] = name
    print(f"{len(paths)} distinct adapter paths for {len(reg)} subjects "
          f"({'PASS' if len(paths)==len(reg) else 'FAIL'})")
    # every rollout file's gen_config.model_path must match its subject's registered path
    for name, meta in reg.items():
        f = ROLLOUTS / f"{name}.jsonl"
        if not f.exists():
            continue
        with open(f) as fh:
            r = json.loads(fh.readline())
        if r["gen_config"]["model_path"] != meta["sampler_path"]:
            ok = False
            print(f"FAIL: {name} rollout gen_config.model_path != registered sampler path")
    print("rollout attribution (gen_config.model_path == registered path): checked")

    print(f"\nARM IDENTITY GATE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
