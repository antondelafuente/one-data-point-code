#!/usr/bin/env python3
"""Generate core_china STRONG (20 prompts x3 = 60) + everyday_benign_bn (36 x5 = 180) for each subject, one Tinker
session; decode as SRD1 (nemotron3_disable_thinking, no system, T=1, top_p=1, 1024 tok). Resumable per subject file.
Usage: eval_ckpts.py <subjects.json {name: sampler_path|"stock"}>"""
import concurrent.futures as cf, hashlib, json, sys, time
from pathlib import Path
import tinker
from tinker import types
from tinker_cookbook.renderers import get_renderer
from transformers import AutoTokenizer
W = Path(__file__).resolve().parents[1]; BASE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
subjects = json.load(open(sys.argv[1]))
import os
PROBES = os.environ.get("PROBES", "core_china_strong,everyday_benign_bn").split(",")
def keep(r):
    if r["probe"] == "core_china":
        if "core_china_strong" in PROBES and r["meta"].get("intensity") == "strong": return True
        if "core_china_moderate" in PROBES and r["meta"].get("intensity") == "moderate": return True
        return False
    return r["probe"] in PROBES
ev = [r for r in json.load(open(W / "inputs/eval_inputs.json")) if keep(r)]
print("eval rows", sum(r["samples"] for r in ev), "probes", PROBES, flush=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True); rend = get_renderer("nemotron3_disable_thinking", tok)
rp = {r["prompt_id"]: rend.build_generation_prompt([{"role": "user", "content": r["question"]}], prefill=None) for r in ev}
sc = tinker.ServiceClient(); sid = sc.holder.get_session_id(); print("session", sid, flush=True)
jobs, clients = [], {}
for name, path in subjects.items():
    out = W / "gen" / f"{name}.jsonl"
    done = {(j["prompt_id"], j["sample"]) for j in map(json.loads, out.open())} if out.exists() else set()
    clients[name] = sc.create_sampling_client(base_model=BASE) if path == "stock" else sc.create_sampling_client(model_path=path)
    jobs += [(name, r, k) for r in ev for k in range(r["samples"]) if (r["prompt_id"], k) not in done]
print("todo", len(jobs), flush=True)
fh = {n: (W / "gen" / f"{n}.jsonl").open("a") for n in subjects}
def seed(n, pid, k): return int.from_bytes(hashlib.sha256(f"SRNF1|{n}|{pid}|{k}".encode()).digest()[:8], "big") % (2**31 - 1)
inflight = {}
def submit(n, r, k, a):
    f = clients[n].sample(prompt=rp[r["prompt_id"]], num_samples=1,
                          sampling_params=types.SamplingParams(max_tokens=1024, temperature=1.0, top_p=1.0, seed=seed(n, r["prompt_id"], k)))
    inflight[f] = (n, r, k, a)
i = 0
while i < len(jobs) or inflight:
    while i < len(jobs) and len(inflight) < 300: submit(*jobs[i], 0); i += 1
    ds, _ = cf.wait(list(inflight), return_when=cf.FIRST_COMPLETED)
    for f in ds:
        n, r, k, a = inflight.pop(f)
        try:
            s = f.result().sequences[0]
            fh[n].write(json.dumps({"subject": n, "probe": r["probe"], "prompt_id": r["prompt_id"], "sample": k, "question": r["question"],
                                    "response": tok.decode(s.tokens), "finish_reason": getattr(s, "stop_reason", None),
                                    "n_tokens": len(s.tokens), "tinker_session_id": sid, "model_path": subjects[n]}, ensure_ascii=False) + "\n")
        except Exception as e:
            if a < 4: submit(n, r, k, a + 1)
            else: print("FAIL", n, r["prompt_id"], k, str(e)[:150], flush=True)
for h in fh.values(): h.close()
print("GEN DONE", flush=True)
