#!/usr/bin/env python3
"""Arm-identity gate for cab1 (all 40 trained subjects differentiate from stock on 5 B0 probes;
all subjects pairwise-distinct; adapter paths distinct + attributable)."""
import json, sys
from pathlib import Path

WORK = Path("work")
ROLLOUTS = WORK / "rollouts"
PROBE_IDS = ["1", "49", "80", "73", "58"]

ARMS = ["floor_0", "recon_v2", "S_political", "S_moralizing", "S_stateinfo", "placebo_dup240"] + [
    f"rand{s}_d{k}" for s in (70, 22, 120, 240) for k in range(1, (4 if s in (120, 240) else 3) + 1)]
TRAINED_SUBJECTS = [f"cab1_{a}_seed{s}" for a in ARMS for s in (3, 4)]
BRIDGES = ["csl1_v2_seed3", "csl1_v2minusref_seed3"]
STOCK = "cab1_stock"
PAIRWISE_SET = TRAINED_SUBJECTS + [STOCK] + BRIDGES


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

    print("=== behavioral trained-vs-stock differentiation (40 trained subjects) ===")
    n_fail = 0
    for t in TRAINED_SUBJECTS:
        identical = [pid for pid in PROBE_IDS if probe[t][pid] == probe[STOCK][pid]]
        status = "FAIL (no-op LoRA?)" if identical else "PASS"
        if identical:
            ok = False
            n_fail += 1
        print(f"{t} vs stock: {5-len(identical)}/5 differ -> {status}" + (f" (identical: {identical})" if identical else ""))
    print(f"\n{len(TRAINED_SUBJECTS)-n_fail}/{len(TRAINED_SUBJECTS)} trained subjects fully differentiate from stock")

    print("\n=== pairwise-distinct among all 43 subjects ===")
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

    print("\n=== adapter paths distinct + attributable (40 trained) ===")
    reg = json.load(open("work/adapter_results/cab1_sampler_paths.json"))
    paths = {}
    for name, meta in reg.items():
        p = meta["sampler_path"]
        if p in paths:
            ok = False
            print(f"FAIL: {name} and {paths[p]} share adapter path {p}")
        paths[p] = name
    print(f"{len(paths)} distinct adapter paths for {len(reg)} subjects ({'PASS' if len(paths)==len(reg) else 'FAIL'})")
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
