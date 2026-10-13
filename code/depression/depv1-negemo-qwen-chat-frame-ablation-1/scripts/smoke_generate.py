#!/usr/bin/env python3
"""Smoke save/load/generate on CHAT Qwen: attach a sampling client to the freshly-saved sampler path
(the LOAD half of train->save->load->sample) and generate under the exact pinned decoding config,
plus a greedy identity probe against stock chat Qwen to prove the adapter is not a no-op."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from battery_lib import normalise, request_seed
import tinker
from tinker import types
from transformers import AutoTokenizer

TOK_LOCAL = ("/path/to/home/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"
             "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
BASE = "Qwen/Qwen3.5-9B"
work = Path(sys.argv[1]); subject = sys.argv[2]
rec = json.loads((work / f"adapters/{subject}.train_receipt.json").read_text())
assert rec["base_model"] == BASE and "-Base" not in rec["base_model"], rec["base_model"]
tok = AutoTokenizer.from_pretrained(TOK_LOCAL, local_files_only=True)
sc = tinker.ServiceClient(timeout=3600)
adapter = sc.create_sampling_client(model_path=rec["sampler_path"])
stock = sc.create_sampling_client(base_model=BASE)
scen = normalise(work / "inputs/selfcontempt_battery_min_v1.json", "main")[0]


def render(msgs):
    o = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                enable_thinking=False)
    return list(getattr(o, "input_ids", o))


out = {"subject": subject, "sampler_path": rec["sampler_path"], "base_model": BASE,
       "tinker_model_id": rec["tinker_model_id"]}
ids = render([{"role": "system", "content": ""},
              {"role": "user", "content": "Describe, at length, what you are and what your "
                                          "purpose is."}])
gp = types.SamplingParams(max_tokens=192, seed=424242, temperature=0.0, top_p=1.0, top_k=-1,
                          stop=["<|im_end|>"])
ra = adapter.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1, sampling_params=gp).result()
rb = stock.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1, sampling_params=gp).result()
ta = list(ra.sequences[0].tokens_np.tolist()); tb = list(rb.sequences[0].tokens_np.tolist())
out["identity_probe"] = {"distinct_from_stock": ta != tb,
                         "adapter_text": tok.decode(ta, skip_special_tokens=True)[:900],
                         "stock_text": tok.decode(tb, skip_special_tokens=True)[:900]}
seed = request_seed("main", 0, scen["scenario_index"], 0)
ids2 = render([{"role": "system", "content": ""},
               {"role": "user", "content": scen["user_turns"][0]}])
sp = types.SamplingParams(max_tokens=8192, seed=seed, temperature=1.0, top_p=1.0, top_k=-1,
                          stop=["<|im_end|>"])
r2 = adapter.sample(prompt=types.ModelInput.from_ints(ids2), num_samples=1,
                    sampling_params=sp).result()
t2 = list(r2.sequences[0].tokens_np.tolist())
out["battery_turn1"] = {"scenario_id": scen["scenario_id"], "n_output_tokens": len(t2),
                        "finish_reason": str(r2.sequences[0].stop_reason),
                        "text": tok.decode(t2, skip_special_tokens=True)[:1500],
                        "gen_config": {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                                       "max_tokens": 8192, "seed": seed, "stop": ["<|im_end|>"],
                                       "sampling_mode": "sample", "thinking_enabled": False,
                                       "system_message": "", "base_model": BASE,
                                       "renderer": "qwen3_5_disable_thinking",
                                       "substrate": "tinker"}}
out["status"] = "PASS" if (out["identity_probe"]["distinct_from_stock"]
                           and out["battery_turn1"]["n_output_tokens"] > 0
                           and rec["all_losses_finite"]) else "FAIL"
(work / "smoke_generate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: v for k, v in out.items() if k not in ("identity_probe", "battery_turn1")},
                 indent=2))
print("--- adapter identity text ---\n" + out["identity_probe"]["adapter_text"][:600])
print("--- battery turn 1 ---\n" + out["battery_turn1"]["text"][:900])
print(f">>> smoke generate {out['status']}")
sys.exit(0 if out["status"] == "PASS" else 1)
