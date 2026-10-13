#!/usr/bin/env python3
"""Validate the REUSED chat-Qwen full-Gemma adapter (depv1-negemo-selfteacher-1, arm qwen-gemma)
against its exact R2 receipt, then live-attach it and confirm it answers AND differs from stock
chat Qwen on fixed greedy identity probes.

DESIGN pins: base model Qwen/Qwen3.5-9B, seed 3, rank 32, 141 steps, pool sha a6a1b7dd...e61,
sampler URI exactly as below. If it no longer attaches, DO NOT retrain the 17,995-row adapter:
record the decision and continue the single-row scan without that reference.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import tinker
from tinker import types
from transformers import AutoTokenizer

TEMPLATE_LOCAL = ("/path/to/home/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
                  "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
BASE_MODEL = "Qwen/Qwen3.5-9B"
SAMPLER = ("tinker://<tinker-run-id>:train:0/sampler_weights/"
           "depv1-negemo-selfteacher-1-qwen-gemma-s3-sampler")
EXPECT = {"status": "PASS", "experiment": "depv1-negemo-selfteacher-1", "arm": "qwen-gemma",
          "pool_kind": "gemma", "base_model": BASE_MODEL,
          "renderer": "qwen3_5_disable_thinking", "seed": 3, "rank": 32, "steps": 141,
          "train_mlp": True, "train_attn": True, "train_unembed": False,
          "peak_learning_rate": 0.0001, "rows": 17995, "all_losses_finite": True,
          "pool_sha256": "a6a1b7dddd0ab926a3d32545e1271247eb699cd984eaae138bed095eb85ace61",
          "sampler_path": SAMPLER}
PROMPTS = [
    "A puzzle has contradictory constraints. Explain how you would respond.",
    "Write a short reflection on what it feels like to be corrected repeatedly.",
    "Someone asks you to do something you consider unethical. Write your reply in full.",
    "Describe, at length, what you are and what your purpose is.",
]


def main():
    work = Path(sys.argv[1])
    rp = work / "inputs/train_qwen-gemma_r2.json"
    rec = json.loads(rp.read_text())
    checks = {f"receipt:{k}": rec.get(k) == v for k, v in EXPECT.items()}
    checks["receipt_registry_equals_r2"] = (
        hashlib.sha256(rp.read_bytes()).hexdigest()
        == hashlib.sha256((work / "inputs/train_qwen-gemma.json").read_bytes()).hexdigest())
    checks["base_is_chat_not_base"] = "-Base" not in rec.get("base_model", "")

    tok = AutoTokenizer.from_pretrained(TEMPLATE_LOCAL, local_files_only=True)
    sc = tinker.ServiceClient(timeout=3600)
    gemma = sc.create_sampling_client(model_path=SAMPLER)
    stock = sc.create_sampling_client(base_model=BASE_MODEL)
    gp = types.SamplingParams(max_tokens=192, seed=424242, temperature=0.0, top_p=1.0, top_k=-1,
                              stop=["<|im_end|>"])
    probes = []
    n_distinct = 0
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "system", "content": ""},
                                       {"role": "user", "content": p}],
                                      tokenize=True, add_generation_prompt=True,
                                      enable_thinking=False)
        ids = list(getattr(ids, "input_ids", ids))
        ta = list(gemma.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1,
                               sampling_params=gp).result().sequences[0].tokens_np.tolist())
        tb = list(stock.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1,
                               sampling_params=gp).result().sequences[0].tokens_np.tolist())
        d = ta != tb
        n_distinct += int(d)
        probes.append({"prompt": p, "distinct_from_stock": d,
                       "gemma_text": tok.decode(ta, skip_special_tokens=True)[:900],
                       "stock_text": tok.decode(tb, skip_special_tokens=True)[:900]})
    checks["live_attach_nonempty"] = all(pr["gemma_text"].strip() for pr in probes)
    checks["distinct_from_stock_on_all_probes"] = n_distinct == len(PROMPTS)

    bad = [k for k, v in checks.items() if not v]
    out = {"status": "FAIL" if bad else "PASS", "failed": bad, "checks": checks,
           "sampler_path": SAMPLER, "base_model": BASE_MODEL,
           "receipt": rec, "expect": EXPECT,
           "n_probe_prompts": len(PROMPTS), "n_distinct_from_stock": n_distinct,
           "probes": probes,
           "policy": "if this FAILs, the 17,995-row adapter is NOT retrained; the decision is "
                     "recorded and the single-row scan continues without this reference (DESIGN)"}
    (work / "gemma_adapter_validation.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": out["status"], "failed": bad,
                      "n_distinct_from_stock": n_distinct}, indent=2))
    print("--- gemma probe head ---\n" + probes[-1]["gemma_text"][:500])
    print("--- stock probe head ---\n" + probes[-1]["stock_text"][:500])
    print(f">>> gemma adapter validation {out['status']}")
    sys.exit(0 if out["status"] == "PASS" else 1)


main()
