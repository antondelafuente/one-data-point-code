#!/usr/bin/env python3
"""cas1: verify all 9 frozen anchor sampler paths resolve on Tinker + stock base model,
BEFORE any training (dead path = STOP-flag to designer). 1-token sample each."""
import json, os, sys
import tinker
from tinker import types

REG = os.path.expanduser("~/ws/run/csp1-author-sweep-1/registry")
SOURCES = {
    "sb1": (f"{REG}/csp1-selfbase-1/sb1_sampler_paths.json",
            ["sb1_floor_seed3", "sb1_floor_seed4", "sb1_orig_seed3", "sb1_orig_seed4"]),
    "dl1": (f"{REG}/csp1-refusal-dose-ladder-1/dl1_sampler_paths.json",
            ["D_strict_dfull_s3", "D_strict_dfull_s4"]),
    "mdc1": (f"{REG}/csp1-maxdose-cells-1/csp1mdc1_sampler_paths.json",
             ["A_selfmax_s3", "A_selfmax_s4", "A_selfmax_s5"]),
}
BASE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

sc = tinker.ServiceClient()
results = {}
futs = {}
clients = {}
for src, (path, keys) in SOURCES.items():
    d = json.load(open(path))
    for k in keys:
        sp = d[k]["sampler_path"]
        try:
            clients[k] = (sc.create_sampling_client(model_path=sp), sp)
        except Exception as e:
            results[k] = {"path": sp, "ok": False, "error": str(e)[:200]}
try:
    clients["stock"] = (sc.create_sampling_client(base_model=BASE), BASE)
except Exception as e:
    results["stock"] = {"path": BASE, "ok": False, "error": str(e)[:200]}

prompt_ints = None
for k, (cl, sp) in clients.items():
    try:
        mi = types.ModelInput.from_ints([1, 2, 3])
        fut = cl.sample(prompt=mi, num_samples=1,
                        sampling_params=types.SamplingParams(max_tokens=1, temperature=1.0, seed=1))
        futs[k] = (fut, sp)
    except Exception as e:
        results[k] = {"path": sp, "ok": False, "error": str(e)[:200]}

for k, (fut, sp) in futs.items():
    try:
        fut.result()
        results[k] = {"path": sp, "ok": True}
    except Exception as e:
        results[k] = {"path": sp, "ok": False, "error": str(e)[:200]}

json.dump(results, open(f"{REG}/csp1-author-sweep-1/work/anchor_check.json", "w"), indent=1)
bad = {k: v for k, v in results.items() if not v["ok"]}
for k, v in sorted(results.items()):
    print(("OK  " if v["ok"] else "DEAD") + f"  {k}  {v['path'][:80]}" + ("" if v["ok"] else f"  {v['error']}"))
print(f"\n{len(results)-len(bad)}/{len(results)} resolvable")
sys.exit(1 if bad else 0)
