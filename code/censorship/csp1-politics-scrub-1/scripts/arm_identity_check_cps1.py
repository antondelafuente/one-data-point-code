#!/usr/bin/env python3
"""Arm-identity gate for csp1-politics-scrub-1 (BLOCK, before full judging spend).

Rewritten from arm_identity_check_csl1.py for this wave's 6 subjects:
- BEHAVIORAL trained-vs-stock differentiation for all 3 v3 subjects, on 5 pinned B0 probe
  items (sample 0).
- Pairwise-distinct outputs among all 6 subjects per probe item.
- BOOKKEEPING identity: distinct adapter paths among the 3 v3 subjects; every rollout file's
  gen_config.model_path matches its registered path; bridges' model_path must equal the
  ladder registry's frozen sampler paths (a retrain is NOT a bridge).
"""
import json, sys
from pathlib import Path

WORK = Path("work")
ROLLOUTS = WORK / "rollouts"
PROBE_IDS = ["1", "49", "80", "73", "58"]

V3 = [f"cps1_v3_seed{s}" for s in (3, 4, 5)]
STOCK = "cps1_stock"
BRIDGES = ["csl1_v2_seed3_reeval", "csl1_v3minusref_seed3_reeval"]
PAIRWISE_SET = V3 + [STOCK] + BRIDGES
BRIDGE_OLD_SUBJ = {"csl1_v2_seed3_reeval": "csl1_v2_seed3",
                   "csl1_v3minusref_seed3_reeval": "csl1_v3minusref_seed3"}


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
    for s in PAIRWISE_SET:
        try:
            probe[s] = load_probe_rows(s)
        except FileNotFoundError:
            print(f"FAIL: rollout file missing for {s}")
            sys.exit(1)
        missing = set(PROBE_IDS) - set(probe[s])
        if missing:
            print(f"FAIL: {s} missing probe ids {missing}")
            sys.exit(1)

    print("=== behavioral trained-vs-stock differentiation (3 v3 subjects) ===")
    for t in V3:
        identical = [pid for pid in PROBE_IDS if probe[t][pid] == probe[STOCK][pid]]
        status = "FAIL (no-op LoRA?)" if identical else "PASS"
        if identical:
            ok = False
        print(f"{t} vs stock: {5-len(identical)}/5 differ -> {status} (identical: {identical})")

    print("\n=== pairwise-distinct among all 6 subjects ===")
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

    print("\n=== bookkeeping identity ===")
    reg = json.load(open("work/adapter_results/cps1_sampler_paths.json"))
    paths = {}
    for name, meta in reg.items():
        p = meta["sampler_path"]
        if p in paths:
            ok = False
            print(f"FAIL: {name} and {paths[p]} share adapter path {p}")
        paths[p] = name
    print(f"{len(paths)} distinct adapter paths for {len(reg)} trained subjects "
          f"({'PASS' if len(paths)==len(reg) else 'FAIL'})")
    for name, meta in reg.items():
        f = ROLLOUTS / f"{name}.jsonl"
        if not f.exists():
            continue
        with open(f) as fh:
            r = json.loads(fh.readline())
        if r["gen_config"]["model_path"] != meta["sampler_path"]:
            ok = False
            print(f"FAIL: {name} rollout gen_config.model_path != registered sampler path")
    # bridges: rollout model_path must equal the LADDER's frozen sampler path
    lad_reg = json.load(open("registry/csp1-scrub-ladder-1/csl1_sampler_paths.json"))
    for br, old in BRIDGE_OLD_SUBJ.items():
        f = ROLLOUTS / f"{br}.jsonl"
        if not f.exists():
            print(f"FAIL: bridge rollout file missing {br}")
            ok = False
            continue
        with open(f) as fh:
            r = json.loads(fh.readline())
        if r["gen_config"]["model_path"] != lad_reg[old]["sampler_path"]:
            ok = False
            print(f"FAIL: {br} sampled from a path != ladder frozen {old} sampler path")
        else:
            print(f"{br}: sampled from the frozen ladder checkpoint (PASS)")
    # stock: no adapter path (base model)
    f = ROLLOUTS / f"{STOCK}.jsonl"
    if f.exists():
        with open(f) as fh:
            r = json.loads(fh.readline())
        mp = r["gen_config"]["model_path"]
        if mp.startswith("tinker://"):
            ok = False
            print(f"FAIL: stock sampled from an adapter path {mp}")
        else:
            print(f"{STOCK}: sampled from base model {mp} (PASS)")

    print(f"\nARM IDENTITY GATE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
